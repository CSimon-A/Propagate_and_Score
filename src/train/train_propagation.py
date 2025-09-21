# src/train_propagation.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Stdlib ────────────────────────────────────────────────────────────────
import sys
import argparse
import logging
import random

# ── Third‑party ───────────────────────────────────────────────────────────
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# ── Local ────────────────────────────────────────────────────────────────
import config
from utils.losses import BoundaryLoss, HausdorffLoss, compute_tte_loss
from manage.TumorPropagationDataset import TumorPropagationDataset as Dataset

# ------------------------------
# ARGUMENT PARSER
# ------------------------------
def parse_t0(s: str):
    if s is None:
        return ("none", None)
    s = s.strip().lower()
    if s in ("none", "off", "false", "0"):
        return ("none", None)
    if s == "max":
        return ("max", None)
    if s.startswith("topk:"):
        try:
            k = int(s.split(":", 1)[1])
        except ValueError:
            raise argparse.ArgumentTypeError("topk expects an integer, e.g. topk:9")
        if k <= 0:
            raise argparse.ArgumentTypeError("topk must be > 0")
        return ("topk", k)
    if s.startswith("perc:"):
        v = s.split(":", 1)[1].strip()
        try:
            # Allow perc:1%, perc:5, perc:0.05
            if v.endswith("%"):
                p = float(v[:-1]) / 100.0
            else:
                p = float(v)
                if p > 1.0:  # interpret as percent if >1
                    p /= 100.0
        except ValueError:
            raise argparse.ArgumentTypeError("perc expects a number, e.g. perc:1% or perc:0.01")
        if not (0.0 < p <= 1.0):
            raise argparse.ArgumentTypeError("perc must be in (0,1] or (0,100]%")
        return ("perc", p)
    raise argparse.ArgumentTypeError("Use: max | topk:N | perc:X or perc:X%")


# ------------------------------
# LOGGER SETUP
# ------------------------------
def setup_logger(save_dir, filename="train.log", level=logging.INFO):
    _orig_stdout = sys.stdout

    logfile = os.path.join(save_dir, filename)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(_orig_stdout),
        ]
    )

    logger = logging.getLogger(__name__)

    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass

    sys.stdout = StreamToLogger(logger, level)
    return logger


# ------------------------------
# # SAVE TRAINING PARAMETERS
# ------------------------------
def save_training_params(args, config, save_dir, filename="train_params.txt"):
    import os
    import sys

    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        f.write("Command: " + " ".join(sys.argv) + "\n")
        f.write("\nArguments:\n")
        for key, val in sorted(vars(args).items()):
            f.write(f"{key}: {val}\n")

        f.write("\nConfig Constants:\n")
        for name in dir(config):
            if name.isupper():
                f.write(f"{name}: {getattr(config, name)}\n")

    print(f"Saved training parameters to {path}")


# ------------------------------
# DATALOADER SETUP
# ------------------------------
def build_dataloaders(args):
    # Create the full dataset
    full_ds = Dataset(
        args.data_dir,
        downsample=args.downsample,
        factor=args.factor,
        scale_mode=args.scale_mode,
        max_tp=args.max_timepoint,
        tx= args.tx,
        levelset_steps=args.levelset_steps,
        dataset_transform=args.dataset_transform,
        t0_spec=args.t0,
        threshold=args.threshold,
        decomposition_steps = args.decomposition_steps,
        decomposition_mode = args.decomposition_mode,
        decomposition_decay = args.decomposition_decay,
        seed = args.seed,
        noise_amp = args.noise_amp
        )
    
    indices = list(range(len(full_ds)))
    print(f"Total samples in dataset: {len(full_ds)}")
    if args.subset_frac < 1.0:
        subset_size = max(1, int(len(indices) * args.subset_frac))
        indices = random.sample(indices, subset_size)

    if len(indices) > 1:
        train_idx, val_idx = train_test_split(indices, test_size=args.val_size, random_state=args.seed)
    else:
        train_idx, val_idx = indices, []

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True) if val_idx else None

    return train_loader, val_loader, full_ds, train_idx, val_idx


# ------------------------------
# MODEL SETUP
# ------------------------------
def build_model(args):
    if args.model == 'small':
        from models.Unet3D_smaller import Unet3D_smaller as Model
    elif args.model == 'larger':
        from models.Unet3D_larger import Unet3D_larger as Model
    elif args.model == 'larger_skip':
        from models.Unet3D_larger_skip import Unet3D_larger_skip as Model
    elif args.model == 'Unet3D':
        from models.Unet3D import Unet3D as Model

    model = Model(config.INPUT_CHANNELS, config.OUTPUT_CHANNELS).to(config.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    return model, opt, scaler


# ------------------------------
# LOSS SETUP
# ------------------------------
def build_losses(args):
    if args.dataset_transform == 'T2E':
        return torch.nn.L1Loss(reduction='none')
    
    else:
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight).to(config.DEVICE))
        bd  = BoundaryLoss()
        hd  = HausdorffLoss()
        return bce, bd, hd


def train(args):
    writer = SummaryWriter(log_dir=args.save_dir)

    train_loader, val_loader, _, _, _ = build_dataloaders(args)
    model, opt, scaler = build_model(args)
    losses = build_losses(args)

    
    if args.num_saves <= 1:
        save_epochs = {args.epochs}
    else:
        tmp = np.linspace(1, args.epochs, num=args.num_saves)
        save_epochs = {int(round(e)) for e in tmp}
        save_epochs.add(args.epochs)

    best_val_loss = float('inf')
    worse_than_best_count = 0


    for epoch in tqdm(range(1, args.epochs+1), desc="Epochs"):
        # reset peak stats if GPU
        if config.DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats(0)

        # training
        model.train()
        train_loss = 0.0
        num_ok_batches = 0
        num_skipped = 0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            opt.zero_grad()
            with autocast():
                logits = model(x)

                if args.dataset_transform == 'T2E':
                    # apply masked L1 regression
                    pred_time = logits.squeeze(1)
                    true_time = y.squeeze(1).float()
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        loss = compute_tte_loss(pred_time, true_time, args.bg_weight)

                    if not torch.isfinite(loss):
                        print("WARN: non-finite loss. stats:",
                              "logits:", logits.min().item(), logits.max().item(),
                              "pred_time:", pred_time.min().item(), pred_time.max().item(),
                              "true_time:", true_time.min().item(), true_time.max().item())
                        opt.zero_grad(set_to_none=True)
                        num_skipped += 1
                        scaler.update()
                        continue

                    writer.add_scalar('Loss/train_L1', loss.item(), epoch)
                    print(f"Epoch {epoch}/{args.epochs}  Train L1={loss:.4f}")


                else:
                    bce, bd, hd = losses
                    bce_loss = bce(logits, y)
                    bd_loss = bd(logits, y)
                    hd_loss = hd(logits, y)
                    loss = args.bce_weight * bce_loss + args.bd_weight * bd_loss + args.hd_weight * hd_loss

                    writer.add_scalar('Loss/train_BCE', bce_loss.item(), epoch)
                    writer.add_scalar('Loss/train_BD', bd_loss.item(), epoch)
                    writer.add_scalar('Loss/train_HD', hd_loss.item(), epoch)
                    
                    print(f"Epoch {epoch}/{args.epochs}: "
                            f"BCE={bce_loss.item():.4f}, "
                            f"Boundary={bd_loss.item():.4f}, "
                            f"Hausdorff={hd_loss.item():.4f}")

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            bad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad = True; break
            if bad:
                print("WARN: non-finite grad; skipping step")
                opt.zero_grad(set_to_none=True)
                num_skipped += 1
                scaler.update()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            train_loss += loss.detach().item()
            num_ok_batches += 1
        train_loss /= max(1, num_ok_batches)

        print(f"Epoch {epoch}/{args.epochs} Train Loss={train_loss:.4f} ({num_ok_batches} batches, {num_skipped} skipped)")
        writer.add_scalar('Loss/train_total', train_loss, epoch)

        # validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            num_ok_batches = 0
            num_skipped = 0
            with torch.no_grad(), autocast():
                for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False):
                    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

                    logits = model(x)

                    if args.dataset_transform == 'T2E':
                        # apply masked L1 regression
                        pred_time = logits.squeeze(1)
                        true_time = y.squeeze(1).float()

                        with torch.cuda.amp.autocast(enabled=False):
                            loss = compute_tte_loss(pred_time, true_time, args.bg_weight)

                        if not torch.isfinite(loss):
                            print("WARN: non-finite loss. stats:",
                                "logits:", logits.min().item(), logits.max().item(),
                                "pred_time:", pred_time.min().item(), pred_time.max().item(),
                                "true_time:", true_time.min().item(), true_time.max().item())
                            num_skipped += 1
                            continue

                    else:
                        bce, bd, hd = losses
                        bce_loss = bce(logits, y)
                        bd_loss = bd(logits, y)
                        hd_loss = hd(logits, y)
                        loss = args.bce_weight * bce_loss + args.bd_weight * bd_loss + args.hd_weight * hd_loss
                    
                    batch_loss = loss
                    val_loss += batch_loss.detach().item()
                    num_ok_batches += 1
            val_loss /= max(1, num_ok_batches)

            print(f"Epoch {epoch}/{args.epochs} Val Loss={val_loss:.4f} ({num_ok_batches} batches, {num_skipped} skipped)")

            if args.dataset_transform == 'T2E':
                writer.add_scalar('Loss/val_total_L1', val_loss, epoch)
            else:
                writer.add_scalar('Loss/val_total', val_loss, epoch)

            # smarter early stopping
            if val_loss < best_val_loss - args.es_delta:
                best_val_loss = val_loss
                worse_than_best_count = 0
                print(f"New best val loss: {best_val_loss:.6f}")
                ckpt_path = os.path.join(args.save_dir, 'propagation_unet_best.pth')
                torch.save(model.state_dict(), ckpt_path)
            else:
                worse_than_best_count += 1
                print(f"Val loss did not improve (current: {val_loss:.6f}, best: {best_val_loss:.6f})")
                print(f"Worse-than-best count: {worse_than_best_count}/{args.es_patience}")
                if worse_than_best_count >= args.es_patience:
                    print("Early stopping: too many epochs worse than best val loss.")
                    break


        # — measure GPU memory before cache clear —
        if config.DEVICE == 'cuda':
            alloc = torch.cuda.memory_allocated(0)
            alloc_max = torch.cuda.max_memory_allocated()
            resv  = torch.cuda.memory_reserved(0)
            tot   = torch.cuda.get_device_properties(0).total_memory
            print(f"[Epoch {epoch}] GPU alloc: {alloc/1024**3:.2f} GB, "
                  f"max alloc: {alloc_max/1024**3:.2f} GB, "
                  f"reserved: {resv/1024**3:.2f} GB / {tot/1024**3:.2f} GB")

        torch.cuda.empty_cache()

        # — save one example prediction from training and validation as NIfTI —
        if epoch in save_epochs:
            model.eval()
            with torch.no_grad(), autocast():
                # TRAIN example
                x_train, y_train, new_aff_batch = next(iter(train_loader))
                # new_aff_batch has shape [batch, 4, 4]; extract the first (and only) affine:
                new_aff_train = new_aff_batch[0].cpu().numpy() if torch.is_tensor(new_aff_batch) else new_aff_batch[0]

                x_train = x_train.to(config.DEVICE)
                logits_train = model(x_train)

                if args.dataset_transform == 'T2E':
                    prob_train = logits_train.squeeze(1).cpu().numpy()[0]  # [D,H,W]
                else:
                    prob_train = torch.sigmoid(logits_train).cpu().numpy()[0, 0]

                nib.save(nib.Nifti1Image(prob_train.astype(np.float32), new_aff_train),
                        os.path.join(args.save_dir, f'epoch_{epoch:03d}_pred_train.nii.gz'))

                y_np_train = y_train.cpu().numpy()[0, 0]
                nib.save(nib.Nifti1Image(y_np_train.astype(np.float32), new_aff_train),
                        os.path.join(args.save_dir, f'epoch_{epoch:03d}_gt_train.nii.gz'))

            if args.want_all_checkpoints:
                if val_loader:
                    x_val, y_val, new_aff_batch_val = next(iter(val_loader))
                    new_aff_val = new_aff_batch_val[0].cpu().numpy() if torch.is_tensor(new_aff_batch_val) else new_aff_batch_val[0]

                    x_val = x_val.to(config.DEVICE)
                    logits_val = model(x_val)

                    if args.dataset_transform == 'T2E':
                        prob_val = logits_val.squeeze(1).cpu().numpy()[0]
                    else:
                        prob_val = torch.sigmoid(logits_val).cpu().numpy()[0, 0]  # [D,H,W]

                    nib.save(nib.Nifti1Image(prob_val.astype(np.float32), new_aff_val),
                            os.path.join(args.save_dir, f'epoch_{epoch:03d}_pred_val.nii.gz'))

                    y_np_val = y_val.cpu().numpy()[0, 0]
                    nib.save(nib.Nifti1Image(y_np_val.astype(np.float32), new_aff_val),
                            os.path.join(args.save_dir, f'epoch_{epoch:03d}_gt_val.nii.gz'))
                    
                ckpt_path = os.path.join(
                    args.save_dir,
                    f'propagation_unet_epoch_{epoch:03d}.pth'
                    )
                torch.save(model.state_dict(), ckpt_path)

    # Manual logging of hyperparameters and final val loss
    hparams = {
        'model': args.model,
        'data_dir': args.data_dir,
        'downsample': args.downsample,
        'factor': args.factor,
        'scale_mode': args.scale_mode,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'val_size': args.val_size,
        'subset_frac': args.subset_frac,
        'pos_weight': args.pos_weight,
        'bce_weight': args.bce_weight,
        'bd_weight': args.bd_weight,
        'hd_weight': args.hd_weight,
        'bg_weight': args.bg_weight,
        'es_patience': args.es_patience,
        'es_delta': args.es_delta,
        'dataset_transform': args.dataset_transform,
        'tx': args.tx,
        'levelset_steps': args.levelset_steps,
        'num_saves': args.num_saves,
        't0_spec': args.t0,
        'save_dir': args.save_dir,
        'threshold': args.threshold,
        'decomposition_steps': args.decomposition_steps,
        'decomposition_mode': args.decomposition_mode,
        'decomposition_decay': args.decomposition_decay,
        'seed': args.seed,
        'noise_amp': args.noise_amp,
        'want_all_checkpoints': args.want_all_checkpoints,
        'device': config.DEVICE,
        'input_channels': config.INPUT_CHANNELS,
        'output_channels': config.OUTPUT_CHANNELS,
        'bbox_size': config.BBOX_SIZE,
        'input_dims': (config.D, config.H, config.W),
        'max_timepoint': args.max_timepoint
    }

    # Write as individual text summaries under "hparams/"
    for key, val in hparams.items():
        writer.add_text(f'hparams/{key}', str(val))

    # Final val loss as scalar
    writer.add_scalar('hparam_metrics/final_val_loss', best_val_loss)
    
    model.eval()

    with torch.no_grad(), autocast():
        x_train, y_train, new_aff_batch = next(iter(train_loader))
        new_aff_train = new_aff_batch[0].cpu().numpy() if torch.is_tensor(new_aff_batch) else new_aff_batch[0]

        x_np = x_train.cpu().numpy()[0]  # shape [3, D, H, W]

        nib.save(nib.Nifti1Image(x_np[0], new_aff_train),
                os.path.join(args.save_dir, f'epoch_{epoch:03d}_train_input_bravo.nii.gz'))
        nib.save(nib.Nifti1Image(x_np[1], new_aff_train),
                os.path.join(args.save_dir, f'epoch_{epoch:03d}_train_input_flair.nii.gz'))
        nib.save(nib.Nifti1Image(x_np[2], new_aff_train),
                os.path.join(args.save_dir, f'epoch_{epoch:03d}_train_input_mask_t.nii.gz'))
    

        x_train = x_train.to(config.DEVICE)
        logits_train = model(x_train)

        if args.dataset_transform == 'T2E':
            prob_train = logits_train.squeeze(1).cpu().numpy()[0]  # [D,H,W]
        else:
            prob_train = torch.sigmoid(logits_train).cpu().numpy()[0, 0]

        nib.save(nib.Nifti1Image(prob_train.astype(np.float32), new_aff_train),
                os.path.join(args.save_dir, f'epoch_{epoch:03d}_train_pred.nii.gz'))

        y_np_train = y_train.cpu().numpy()[0, 0]
        nib.save(nib.Nifti1Image(y_np_train.astype(np.float32), new_aff_train),
                os.path.join(args.save_dir, f'epoch_{epoch:03d}_train_gt.nii.gz'))
    
    # optionally save val inputs
    if val_loader:
        with torch.no_grad(), autocast():
            x_val, y_val, new_aff_batch_val = next(iter(val_loader))
            new_aff_val = new_aff_batch_val[0].cpu().numpy() if torch.is_tensor(new_aff_batch_val) else new_aff_batch_val[0]

            x_val_np = x_val.cpu().numpy()[0]

            nib.save(nib.Nifti1Image(x_val_np[0], new_aff_val),
                    os.path.join(args.save_dir, f'epoch_{epoch:03d}_val_input_bravo.nii.gz'))
            nib.save(nib.Nifti1Image(x_val_np[1], new_aff_val),
                    os.path.join(args.save_dir, f'epoch_{epoch:03d}_val_input_flair.nii.gz'))
            nib.save(nib.Nifti1Image(x_val_np[2], new_aff_val),
                    os.path.join(args.save_dir, f'epoch_{epoch:03d}_val_input_mask_t.nii.gz'))
        
            x_val = x_val.to(config.DEVICE)
            logits_val = model(x_val)

            if args.dataset_transform == 'T2E':
                prob_val = logits_val.squeeze(1).cpu().numpy()[0]
            else:
                prob_val = torch.sigmoid(logits_val).cpu().numpy()[0, 0]  # [D,H,W]

            nib.save(nib.Nifti1Image(prob_val.astype(np.float32), new_aff_val),
                    os.path.join(args.save_dir, f'epoch_{epoch:03d}_pred_val.nii.gz'))

            y_np_val = y_val.cpu().numpy()[0, 0]
            nib.save(nib.Nifti1Image(y_np_val.astype(np.float32), new_aff_val),
                    os.path.join(args.save_dir, f'epoch_{epoch:03d}_gt_val.nii.gz'))

    print(f"Saved example inputs NIfTI for epoch {epoch}")

    writer.close()


if __name__ == '__main__':
    torch.set_num_threads(1)

    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', 
                   type=str,
                   default=config.DATA_DIR,
                   help=f'directory with training data (default: {config.DATA_DIR})')
    p.add_argument('--dataset-transform', 
                   type=str, 
                   choices=['None', 'T2E'],
                   default=config.DATASET_TRANSFORM,
                   help=f'Dataset type to use (default: {config.DATASET_TRANSFORM})')
    p.add_argument('--max-timepoint', 
                   type=int,
                   default=config.MAX_TIMEPOINTS,
                   help=f'maximum timepoint in dataset (default: {config.MAX_TIMEPOINTS})')
    p.add_argument('--tx',
                   type=int, 
                   default=config.TX,
                   help=f'Latest future timepoint to include as a target (default: {config.TX})')
    
    p.add_argument('--decomposition-steps',
                   type=int,
                   default=config.DECOMPOSITION_STEPS,
                   help=f'number of morphological decomposition steps to apply to input mask t1 (default: {config.DECOMPOSITION_STEPS})')
    p.add_argument('--decomposition-mode',
                   type=str,
                   choices=['linear', 'exponential', 'exp_e'],
                   default=config.DECOMPOSITION_MODE,
                   help=f'morphological operation for decomposition (default {config.DECOMPOSITION_MODE})')
    p.add_argument('--decomposition-decay',
                   type=float,
                   default=config.DECOMPOSITION_DECAY,
                   help=f'decay factor for decomposition steps for exp mode (default: {config.DECOMPOSITION_DECAY})')
    p.add_argument('--noise-amp',
                   type=float,
                   default=config.NOISE_AMP,
                   help=f'amplitude of uniform noise to add to decomposition fractions (default: {config.NOISE_AMP})')  
    
    p.add_argument('--downsample', 
                   action='store_true',
                   default=config.DOWNSAMPLE,
                   help=f'downsample input data by factor 2 (default: {config.DOWNSAMPLE})')
    p.add_argument('--factor', 
                   type=int, 
                   default=config.FACTOR,
                   help=f'downsampling factor (default: {config.FACTOR})')
    p.add_argument('--scale-mode', 
                   type=str,
                   choices=['min-max', 'tanh', 'sigmoid', 'none'], 
                   default=config.SCALE_MODE,
                   help=f'how to scale input data to [0,1] (default: {config.SCALE_MODE})')
    p.add_argument('--model',
                   type=str,
                   choices=['small', 'larger', 'larger_skip', 'Unet3D'],
                   default=config.MODEL,
                   help=f'Which U‑Net variant to train (default: {config.MODEL})')
    p.add_argument('--epochs', 
                   type=int, 
                   default=config.EPOCHS,
                   help=f'number of training epochs (default: {config.EPOCHS})')
    p.add_argument('--batch-size', 
                   type=int, 
                   default=config.BATCH_SIZE,
                   help=f'batch size for training (default: {config.BATCH_SIZE})')
    p.add_argument('--lr', 
                   type=float, 
                   default=config.LR,
                   help=f'learning rate for Adam optimizer (default: {config.LR})')
    p.add_argument('--val-size', 
                   type=float, 
                   default=config.VAL_SIZE,
                   help=f'fraction of data for validation (default: {config.VAL_SIZE})')
    p.add_argument('--seed', 
                   type=int, 
                   default=config.SEED,
                   help=f'random seed for train/val split (default: {config.SEED})')
    p.add_argument('--save-dir',
                   type=str, 
                   default=config.SAVE_DIR,
                   help=f'directory to save model checkpoints and logs (default: {config.SAVE_DIR})')
    p.add_argument('--subset-frac',  
                   type=float, 
                   default=config.SUBSET_FRAC,
                   help=f'use only this fraction of samples (e.g. 0.1 for 10%) before train/val split (default: {config.SUBSET_FRAC})')
    p.add_argument('--num-saves', 
                   type=int, 
                   default=config.NUM_SAVES,
                   help=f'how many prediction snapshots to save (evenly spaced; last epoch always included) (default: {config.NUM_SAVES})')
    p.add_argument('--pos-weight', 
                   type=float, 
                   default=config.POS_WEIGHT,
                   help=f'positive weight for BCE loss (default: {config.POS_WEIGHT})')
    p.add_argument('--bce-weight', 
                   type=float, 
                   default=config.BCE_WEIGHT,
                   help=f'weight for BCE loss (default: {config.BCE_WEIGHT})')
    p.add_argument('--bd-weight', 
                   type=float, 
                   default=config.BD_WEIGHT,
                   help=f'weight for boundary loss (default: {config.BD_WEIGHT})')
    p.add_argument('--hd-weight', 
                   type=float, 
                   default=config.HD_WEIGHT,
                   help=f'weight for Hausdorff loss (default: {config.HD_WEIGHT})')
    p.add_argument('--bg-weight',
                   type=float,
                   default=config.BG_WEIGHT,
                   help=f'weight for background penalty in L1 loss (default: {config.BG_WEIGHT})')
    p.add_argument('--es-patience', 
                   type=int, 
                   default=config.ES_PATIENCE,
                   help=f'# of epochs with val_loss < es_delta before early stopping (default: {config.ES_PATIENCE})')
    p.add_argument('--es-delta', 
                   type=float, 
                   default=config.ES_DELTA,
                   help=f'min change in val_loss to qualify as improvement (default: {config.ES_DELTA})')
    p.add_argument('--levelset-steps', 
                   type=int, 
                   default=config.LEVELSET_STEPS,
                   help=f'number of levelset steps to apply (default: {config.LEVELSET_STEPS})')
    p.add_argument("--t0",
                   type=parse_t0,
                   default=config.T0_SPEC,
                   help=f"t0 decomposition: max | topk:N | perc:X or perc:X% (default: {config.T0_SPEC})")
    p.add_argument('--threshold',
                   type=float,
                   default=config.THRESHOLD,
                   help=f'threshold for binarizing masks (default: {config.THRESHOLD})')
    p.add_argument('--want-all-checkpoints',
                   action='store_true',
                   default=config.WANT_ALL_CHECKPOINTS,
                   help=f'if set, save model checkpoint at every save_epochs (not just best val loss) (default: {config.WANT_ALL_CHECKPOINTS})')

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create a timestamped save directory and log run parameters
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    save_training_params(args, config, args.save_dir)

    logger = setup_logger(args.save_dir)
    
    train(args)
