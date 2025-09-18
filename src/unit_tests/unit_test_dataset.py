#src % python -m unit_tests.unit_test_dataset


import os
import nibabel as nib
import numpy as np

import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset

import train.config as config
from train.utils.utils import robust_minmax, threshold, load_nifti
from train.utils.utils import top_Xperc_voxels

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
                 decomposition_steps,
                 decomposition_mode,
                 decomposition_decay,
                 seed,
                 noise_amp
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
        self.decomposition_steps = decomposition_steps
        self.decomposition_mode  = decomposition_mode
        self.decomposition_decay = decomposition_decay
        self.seed = seed
        self.noise_amp = noise_amp

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
                                self.samples.append((bravo, flair, mask_t, 0, True, i, self.decomposition_steps)) # i in [1..s-1], s = decomp
                        else:
                            # Single synthetic t0 (basic erosion-like core)
                            self.samples.append((bravo, flair, mask_t, 0, True, 1, 1))

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
    
    def _make_rng(self, idx):
        return np.random.default_rng(self.seed + idx)

    def __getitem__(self, idx):
        rng = self._make_rng(idx)

        
        item = self.samples[idx]

        # Support 5-tuple (normal) and 7-tuple (synthetic t0.* with step info)
        if len(item) == 5:
            b_path, f_path, m_t_path, t, synth_t0 = item
            step_idx, total_steps = None, None
        else:
            b_path, f_path, m_t_path, t, synth_t0, step_idx, total_steps = item # step_idx in [1..s-1], total_steps = s

        tumors_dir = os.path.dirname(m_t_path)
        suffix = os.path.basename(m_t_path)[2:]  # "_003.nii.gz"

        fname = os.path.basename(m_t_path)          # e.g., "t1_003.nii.gz"
        m = re.search(r'_(\d{3})\.nii\.gz$', fname) # capture the 3 digits
        num = m.group(1)  

        # load modalities
        b = self._prep_image(load_nifti(b_path)[0])
        f = self._prep_image(load_nifti(f_path)[0])

        mask_vols = []
        if synth_t0:
            # Build progression: t0.i, t0.(i+1), ..., t0.s, then real t1..tx
            mask_vols = []

            t1 = load_nifti(m_t_path)[0]  # real T1 volume (not binarized here)
            t1 = threshold(t1, thr=self.thresh, binary=False)

            print(f"Sample {idx}: synthetic t0 from {m_t_path} (real t1)")
            i = int(step_idx) if step_idx is not None else 1
            s = int(total_steps) if total_steps is not None else 1
            i = max(0, min(i, s))  # clamp safely
            print (f"  -> step {i}/{s} (total steps={s})")

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
            print (f"  -> generate steps j in [{j_start}..{j_end}]")

            for j in range(j_start, j_end + 1):
                print(f"  Sample {idx}: synthetic t0 step {j}/{s} (total steps={s})")
                frac = float(np.clip(frac_keep(j, s, self.decomposition_mode, self.decomposition_decay), 0.0, 1.0))
                print(f"    -> keep {frac*100:.1f}% of tumor voxels")
                noise = rng.uniform(-self.noise_amp, self.noise_amp)
                print(f"    -> noise {noise:+.3f}")
                frac_noisy = float(np.clip(frac + noise, 0.02, 0.98)) # avoid extremes
                print(f"    -> noisy keep {frac_noisy*100:.1f}% of tumor voxels")
                m_j = top_Xperc_voxels(t1, frac_noisy).astype(np.uint8)
                print(f"    -> actual {m_j.sum()} voxels ({(m_j.sum()/m_j.size)*100:.3f}%)")
                # nib.save(m_j, os.path.join('outputs_nifti', f"sample_{num}-{step_idx}-{j:02d}.nii.gz"))
                mask_vols.append(m_j)


            # then append real t1..tx (binarized)
            for tp in range(1, self.tx + 1):
                print(f"  Sample {idx}: real t{tp}")
                p = os.path.join(tumors_dir, f"t{tp}{suffix}")
                print (f"    -> load {p}")
                mask_vols.append(
                    threshold(load_nifti(p)[0], thr=self.thresh, binary=True).astype(np.uint8)
                )


        else:
            # normal case: current time is t, future is t..tx
            for tp in range(t, self.tx + 1):
                print(f"  Sample {idx}: real t{tp}")
                p = os.path.join(tumors_dir, f"t{tp}{suffix}")
                print (f"    -> load {p}")
                mask_vols.append(
                    threshold(load_nifti(p)[0], thr=self.thresh, binary=True).astype(np.uint8))

        # First-event map: index of first True across time; -1 if never tumor
        stacked = np.stack(mask_vols, axis=-1)      # [D,H,W,N]
        ever_tumor = stacked.any(axis=-1)
        first_idx = np.argmax(stacked, axis=-1)
        first_idx[~ever_tumor] = -1 # background
        event_map = first_idx.astype(np.float32)

        b, f, event_map
        target_ds = config.BBOX_SIZE

        # current mask for cropping is "already tumor" at current time (index 0)
        m_ds_b = (event_map == 0).astype(np.float32)

        # now crop all four volumes to the same box, centered on the union of m & y
        aff = load_nifti(b_path)[1]

        # stack & to tensor
        x = np.stack([b, f, m_ds_b], axis=0)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(event_map).unsqueeze(0)

        return xt, yt, aff



def save_sample_outputs_as_nifti(dataset, indices, out_dir="outputs_nifti"):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for k, idx in enumerate(indices):
        # dataset[idx] returns (xt, yt, new_aff)
        _, yt, aff = dataset[idx]
        vol = yt.squeeze(0).numpy().astype(np.int16)  # → (D,H,W) binary mask
        out_nii = nib.Nifti1Image(vol, affine=aff)
        nib.save(out_nii, os.path.join(out_dir, f"sample_{k:02d}.nii.gz"))
        saved += 1
    print(f"Saved {saved} NIfTI file(s) to {out_dir}")


def build_dataloaders():
    full_ds = TumorPropagationDataset(
        '/Users/sa-mbp/Documents/Sherbrooke/SCIL/propagate_and_score/data/original_data',
        downsample=True,
        factor=2,
        scale_mode='min-max',
        max_tp=4,
        tx=2,
        levelset_steps=0,
        dataset_transform='T2E',
        t0_spec=('topk', 9),
        threshold=0.01,
        decomposition_steps=5,
        decomposition_mode='exp_e',
        decomposition_decay=0.5,
        seed=42,
        noise_amp=0.05,
    )

    indices = list(range(len(full_ds)))
    print(f"Total samples in dataset: {len(full_ds)}")

    subset_size = 10
    indices = indices[:subset_size]

    train_idx = indices  # no split; keep the 10

    # Save the 10 outputs now
    save_sample_outputs_as_nifti(full_ds, train_idx, out_dir="outputs_nifti")

    return full_ds


if __name__ == "__main__":

    full_ds = build_dataloaders()