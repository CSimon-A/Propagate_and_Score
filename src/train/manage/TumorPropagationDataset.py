# src/manage/TumorPropagationDataset.py
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.cropping import crop_to_fixed_bbox
import config
from utils.utils import robust_minmax, threshold, load_nifti
from utils.downsample import downsample
from utils.levelset import build_4d_field, make_temporal_interpolator, extract_mask
from utils.utils import max_voxels, top_k_voxels, top_Xperc_voxels

class TumorPropagationDataset(Dataset):
    """
    For each tumor and time t=1..3, returns:
      x: [3, D, H, W] -> (BRAVO, FLAIR, mask_t)
      y: [1, D, H, W] -> next mask (mask_{t+1})
    Optionally crops a random patch of size PATCH_SIZE.
    """
    def __init__(self, 
                 data_dir, 
                 downsample, 
                 factor, 
                 scale_mode, 
                 max_tp,
                 tx, 
                 levelset_steps,
                 dataset_transform,
                 t0_spec,
                 threshold,
                 decomposition_steps = 9,
                 decomposition_mode = 'exp_e',
                 decomposition_decay =0.5
                 ):
        
        self.samples = []
        self.downsample = downsample
        self.factor = factor
        self.scale_mode = scale_mode
        self.max_tp = max_tp
        self.tx = tx
        self.levelset_steps = levelset_steps
        self.dataset_transform = dataset_transform
        self.thresh = threshold
        self.t0_kind, self.t0_val = t0_spec
        self.decomposition_steps = decomposition_steps  # t0.1 .. t0.9
        self.decomposition_mode  = decomposition_mode   # 'linear' | 'exponential' | 'exp_e'
        self.decomposition_decay = decomposition_decay  # only for 'exponential'

        pattern = re.compile(r't([1-4])_\d{3}\.nii\.gz')
        for pid in sorted(os.listdir(data_dir)):
            patient_dir = os.path.join(data_dir, pid)
            tumors_dir = os.path.join(patient_dir, 'tumors_mni')
            if not os.path.isdir(tumors_dir): continue
            bravo = os.path.join(patient_dir, 't1_bravo_bet_mni.nii.gz')
            flair = os.path.join(patient_dir, 't2_flair_bet_mni.nii.gz')
            for fn in sorted(os.listdir(tumors_dir)):
                m = pattern.match(fn)
                if not m: continue
                t = int(m.group(1))
                if t >= self.tx or t >= self.max_tp: continue  # only t=1,2,3
                mask_t  = os.path.join(tumors_dir, fn)

                if self.dataset_transform == 'T2E':
                    self.samples.append((bravo, flair, mask_t, t, False))
                    if self.t0_kind != 'none' and t == 1:
                        # Create synthetic t0.* variants from the T1 mask (m_t_path)
                        if self.decomposition_steps and self.decomposition_steps > 0:
                            start_i, end_i = 1, self.decomposition_steps - 1    # excludes both extremes
                            for i in range(start_i, end_i + 1):
                                self.samples.append((bravo, flair, mask_t, 0, True, i, self.decomposition_steps))
                        else:
                            # Single synthetic t0 (basic erosion-like core)
                            self.samples.append((bravo, flair, mask_t, 0, True, 1, 1))
                else:
                    mask_tp = os.path.join(tumors_dir, f't{t+1}' + fn[2:])
                    if not os.path.exists(mask_tp): continue

                    if self.levelset_steps == 0:
                        # no levelset, just add the pair
                        self.samples.append((bravo, flair, mask_t, mask_tp, t, 0, False))
                    else:
                        for step in range(self.levelset_steps + 1):
                            self.samples.append((bravo, flair, mask_t, mask_tp, t, step, False))

                    if self.t0_kind != 'none' and t == 1:
                        # use the same mask_t path (t1); we’ll derive t0 in __getitem__
                        synth_t0 = True
                        if self.levelset_steps == 0:
                            self.samples.append((bravo, flair, mask_t, mask_t, 0, 0, synth_t0))
                        else:
                            for step in range(self.levelset_steps + 1):
                                self.samples.append((bravo, flair, mask_t, mask_t, 0, step, synth_t0))

    def __len__(self):
        return len(self.samples)
    
    def _prep_image(self, vol):
        # z-score
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        if self.scale_mode == "min-max":
            vol = robust_minmax(vol, 1, 99)
        elif self.scale_mode == "tanh":
            vol = (np.tanh(vol) * 0.5) + 0.5
        elif self.scale_mode == "sigmoid":
            vol = 1 / (1 + np.exp(-vol))
        return vol

    def __getitem__(self, idx):
        if self.dataset_transform == 'T2E':
            item = self.samples[idx]

            # Support 5-tuple (normal) and 7-tuple (synthetic t0.* with step info)
            if len(item) == 5:
                b_path, f_path, m_t_path, t, synth_t0 = item
                step_idx, total_steps = None, None
            else:
                b_path, f_path, m_t_path, t, synth_t0, step_idx, total_steps = item

            tumors_dir = os.path.dirname(m_t_path)
            suffix = os.path.basename(m_t_path)[2:]  # "_003.nii.gz"

            # load modalities
            b = self._prep_image(load_nifti(b_path)[0])
            f = self._prep_image(load_nifti(f_path)[0])

            mask_vols = []
            if synth_t0:
                # Build progression: t0.i, t0.(i+1), ..., t0.s, then real t1..tx
                mask_vols = []

                t1 = load_nifti(m_t_path)[0]  # real T1 volume (not binarized here)
                i = int(step_idx) if step_idx is not None else 1
                s = int(total_steps) if total_steps is not None else 1
                i = max(0, min(i, s))  # clamp safely

                # map j in [i..s] to a monotonically increasing fraction in [0..1]
                def frac_keep(j, s, mode, decay):
                    if s <= 0:
                        return 1.0
                    if mode == 'linear':
                        return j / s
                    elif mode == 'exponential':
                        # rises toward 1 faster; ensures j=s -> 1
                        return min(1.0, max(0.0, 1.0 - (decay ** j)))
                    else:  # 'exp_e'
                        # normalized 1 - exp(-j) so that j=s -> 1 exactly
                        denom = 1.0 - np.exp(-s)
                        return float((1.0 - np.exp(-j)) / denom) if denom > 0 else 1.0

                j_start, j_end = max(i, 1), s - 1     # exclude 0% and 100%
                if j_start > j_end:                   # ensure at least one step
                    j_start = j_end = max(1, s - 1)

                for j in range(j_start, j_end + 1):
                    frac = float(np.clip(frac_keep(j, s, self.decomposition_mode, self.decomposition_decay), 0.0, 1.0))
                    m_j = top_Xperc_voxels(t1, frac).astype(np.uint8)
                    mask_vols.append(m_j)


                # then append real t1..tx (binarized)
                for tp in range(1, self.tx + 1):
                    p = os.path.join(tumors_dir, f"t{tp}{suffix}")
                    mask_vols.append(
                        threshold(load_nifti(p)[0], thr=self.thresh, binary=True).astype(np.uint8)
                    )


            else:
                # normal case: current time is t, future is t..tx
                for tp in range(t, self.tx + 1):
                    p = os.path.join(tumors_dir, f"t{tp}{suffix}")
                    mask_vols.append(threshold(load_nifti(p)[0], thr=self.thresh, binary=True).astype(np.uint8))

            # First-event map: index of first True across time; -1 if never tumor
            stacked = np.stack(mask_vols, axis=-1)      # [D,H,W,N]
            ever_tumor = stacked.any(axis=-1)
            first_idx = np.argmax(stacked, axis=-1)
            first_idx[~ever_tumor] = -1 # background
            event_map = first_idx.astype(np.float32)

            # downsample
            if self.downsample:
                b_ds, f_ds = downsample(b, f, None, None, factor=self.factor, order=1)[:2]
                event_map_ds = downsample(event_map, None, None, None, factor=self.factor, order=0)[0]
                target_ds = tuple(s // self.factor for s in config.BBOX_SIZE)
            else:
                b_ds, f_ds, event_map_ds = b, f, event_map
                target_ds = config.BBOX_SIZE

            # current mask for cropping is "already tumor" at current time (index 0)
            m_ds_b = (event_map_ds == 0).astype(np.float32)
            output = event_map_ds

        else:
            b_path, f_path, m_t_path, m_tp, t, step, synth_t0 = self.samples[idx]

            b = self._prep_image(load_nifti(b_path)[0])
            f = self._prep_image(load_nifti(f_path)[0])

            if synth_t0:
                # m_t is actually t1; synthesize t0 and set y=t1
                t1 = load_nifti(m_t_path)[0]

                if self.t0_kind == 'max':
                    m = max_voxels(t1).astype(np.float32)  # t0

                elif self.t0_kind == 'topk':
                    m = top_k_voxels(t1, self.t0_val).astype(np.float32)

                elif self.t0_kind == 'perc':
                    m = top_Xperc_voxels(t1, self.t0_val).astype(np.float32)
                
                y = threshold(t1, thr=self.thresh, binary=True)  #t1

            else:
                m = load_nifti(m_t_path)[0]
                y = load_nifti(m_tp)[0]

            if self.levelset_steps > 0:
                float_masks = [threshold(img, thr=self.thresh, binary=False) for img in (m, y)]
                vol4d = build_4d_field(float_masks)
                P_cont = make_temporal_interpolator(vol4d, times=(t, t+1), kind="linear")

                #TODO: Error now, linear makes it so the timepoint following t is far from t while
                #      the timepoint before t+1 is close very close to it
                #      should probably use cubic interpolation and load all 4 timepoints to get a regular transitions

                m_timepoint = t + (step / (self.levelset_steps + 1))
                y_timepoint = t + ((step + 1) / (self.levelset_steps + 1))
                m = extract_mask(P_cont, t=m_timepoint)
                y = extract_mask(P_cont, t=y_timepoint)

            if self.downsample:
                b_ds, f_ds, m_ds, y_ds = downsample(b, f, m, y, factor=self.factor, order=1)
                target_ds = tuple(s // self.factor for s in config.BBOX_SIZE)
            else:
                b_ds, f_ds, m_ds, y_ds = b, f, m, y
                target_ds = config.BBOX_SIZE

            m_ds_b = threshold(m_ds, thr=self.thresh, binary=True)
            y_ds_b = threshold(y_ds, thr=self.thresh, binary=True)

            output = y_ds_b

        # now crop all four volumes to the same box, centered on the union of m & y
        vols = [b_ds, f_ds, m_ds_b, output]

        aff = load_nifti(b_path)[1]

        (b_c, f_c, m_c, y_c), _, _, new_aff, _ = crop_to_fixed_bbox(vols, target_ds, aff)

        # stack & to tensor
        x = np.stack([b_c, f_c, m_c], axis=0)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y_c).unsqueeze(0)

        return xt, yt, new_aff

    

