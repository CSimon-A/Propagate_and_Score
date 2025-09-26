# src/config.py

# input data channels (BRAVO, FLAIR, mask)
INPUT_CHANNELS = 3
# we predict a single‐channel mask at next timepoint
OUTPUT_CHANNELS = 1

# original volume dims
D, H, W = 193, 229, 193

# max 6 downsample steps
BBOX_SIZE = (192, 192, 192)

# training settings
DATA_DIR = 'data'           # directory with training data
DATASET_TRANSFORM = 'T2E'   # dataset type to use (None or T2E)
                            # if T2E, we use TumorPropagationDataset_T2E; otherwise, TumorPropagationDataset
                            # (None is the original dataset with all timepoints and masks)
INC_T0 = True               # include t0 in dataset (default: False)
T0_SPEC = ('topk', 9)       # how to synthesize t0
                                # ("none", None) → don’t synthesize t0
                                # ("max", None) → take the maximum intensity/timepoint
                                # ("topk", k) → take the top k voxels
                                # ("perc", p) → take the top fraction p (0.0–1.0) of voxels
THRESHOLD = 0.01            # threshold for input maps
LEVELSET_STEPS = 0          # number of levelset evolution steps to apply to input masks (0 = no levelset)

DECOMPOSITION_STEPS = 5     # number of times to apply decomposition to input masks (0 = no decomposition)
DECOMPOSITION_MODE = 'linear'  # decomposition mode: 'linear', 'exponential', 'exp_e'
DECOMPOSITION_DECAY = 0.5   # decomposition decay rate
NOISE_AMP = 0.09             # amplitude of uniform noise to add to decomposition fractions# (Rule of thumb: keep NOISE_AMP < 0.5 / DECOMPOSITION_STEPS 
                                #  so noise is less than ~half the step size, avoiding too much overlap)


MAX_TIMEPOINTS = 4           # maximum timepoint in the dataset (e.g. 4 for t1,2,3,4)
TX = 4                      # latest future timepoint to include as a target 
                                # (e.g. tx=3 means the model predicts up to t3, starting from any earlier timepoint (t=1, t=2, etc.))

DOWNSAMPLE = True           # downsample input data by a factor
FACTOR = 2                  # downsampling factor
SCALE_MODE = 'min-max'      # how to scale input data to [0,1]
MODEL = 'small'             # which U-Net variant to train
EPOCHS = 100                # number of training epochs
BATCH_SIZE = 1              # batch size for training
LR = 1e-4                   # learning rate for Adam optimizer
VAL_SIZE = 0.2              # fraction of data for validation
SEED = 42                   # random seed for train/val split
SAVE_DIR = 'saved_models'   # directory to save model checkpoints and logs
SUBSET_FRAC = 1.0           # use only this fraction of samples (e.g. 0.1 for 10%) before train/val split
NUM_SAVES = 4               # how many prediction snapshots to save (evenly spaced; last epoch always included)
POS_WEIGHT = 200.0          # positive weight for BCE loss
BCE_WEIGHT = 1.0            # weight for BCE loss
BD_WEIGHT = 1.0             # weight for boundary loss
HD_WEIGHT = 0.5             # weight for Hausdorff loss
BG_WEIGHT = 1.0             # weight for background penalty in L1 loss
ES_PATIENCE = 3             # # of epochs with val_loss < es_delta before early stopping
ES_DELTA = 1e-4             # min change in val_loss to qualify as improvement

# device (cuda or cpu)
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

WANT_ALL_CHECKPOINTS = False  # if True, save model checkpoint at every save_epochs (not just best val loss)