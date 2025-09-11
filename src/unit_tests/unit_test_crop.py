import numpy as np
from train.utils.cropping import crop_to_fixed_bbox

full = np.zeros((100, 120, 90), np.float32)
full[20:40, 50:70, 10:30] = 1.0
mask = (full > 0).astype(np.uint8)

target = (80, 80, 80)
(c_b, c_f, c_m, _), start_pad, end_pad, _, orig_start = crop_to_fixed_bbox(
    [full, full, mask, mask], target, affine=None
)

pred_crop = c_m.astype(np.float32)

def _uncrop_to_full_ds(cropped, ds_full_shape, start, end, fill=0.0):
    out = np.full(ds_full_shape, fill, dtype=cropped.dtype)
    z0,y0,x0 = map(int, start); z1,y1,x1 = map(int, end)
    Z0,Y0,X0 = max(z0,0), max(y0,0), max(x0,0)
    Z1,Y1,X1 = min(z1,ds_full_shape[0]), min(y1,ds_full_shape[1]), min(x1,ds_full_shape[2])
    cz0,cy0,cx0 = Z0 - z0, Y0 - y0, X0 - x0
    cz1,cy1,cx1 = cz0 + (Z1 - Z0), cy0 + (Y1 - Y0), cx0 + (X1 - X0)
    if Z1 > Z0 and Y1 > Y0 and X1 > X0:
        out[Z0:Z1, Y0:Y1, X0:X1] = cropped[cz0:cz1, cy0:cy1, cx0:cx1]
    return out

orig_start = np.asarray(orig_start, int)
orig_end   = orig_start + np.array(target, int)
restored = _uncrop_to_full_ds(pred_crop, full.shape, orig_start, orig_end, fill=0.0)

assert restored.shape == full.shape
assert np.all(restored[20:40, 50:70, 10:30] == 1.0)
assert restored.sum() == mask.sum()
