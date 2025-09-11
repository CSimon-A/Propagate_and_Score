# src/utils/cropping.py

import numpy as np

def crop_to_fixed_bbox(vols, target, affine=None):
    """
    vols: list of 3D numpy arrays [bravo, flair, mask_t, mask_tp]
    target: (Z, Y, X) tuple
    returns: (crops, start, end)
      crops: list of cropped 3D arrays, same order as vols
      start: [z0, y0, x0], the corner in the padded space
      end:   [z1, y1, x1] = start + target
    """
    # ensure inputs are numpy ints
    target = np.array(target, int)
    shape0 = np.array(vols[0].shape, int)

    # 1) pad volumes so each dimension is at least the target size
    pad_needed = np.maximum(target - shape0, 0)
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before

    padded = []
    for v in vols:
        fill = v.min()
        padded.append(
            np.pad(
                v,
                pad_width=[(pad_before[i], pad_after[i]) for i in range(3)],
                mode="constant",
                constant_values=fill
            )
        )

    # 2) build union mask of the two tumor masks
    m = padded[2] > 0
    y = padded[3] > 0
    uni = m | y
    coords = np.argwhere(uni)

    # 3) find center of the union bbox (or center of volume if no mask)
    if coords.size == 0:
        center = np.array(padded[0].shape) // 2
    else:
        mn, mx = coords.min(axis=0), coords.max(axis=0)
        center = ((mn + mx + 1) // 2).astype(int)

    # 4) initial crop bounds
    half = target // 2
    start = center - half
    end = start + target

    # 5) pad again if crop goes out of bounds in the padded volume
    shape_p = np.array(padded[0].shape, int)
    pad2_before = np.maximum(-start, 0)
    pad2_after  = np.maximum(end - shape_p, 0)
    if pad2_before.any() or pad2_after.any():
        # apply additional padding
        padded2 = []
        for p in padded:
            fill2 = p.min()
            padded2.append(
                np.pad(
                    p,
                    pad_width=[(pad2_before[d], pad2_after[d]) for d in range(3)],
                    mode="constant",
                    constant_values=fill2
                )
            )
        padded = padded2
        # shift the window into padded coordinates
        start = start + pad2_before
        end = start + target

    # 6) finally slice out the fixed box
    crops = [p[start[0]:end[0], start[1]:end[1], start[2]:end[2]] for p in padded]

    # compute new affine and header if provided
    new_affine = None
    total_pad = pad_before + (pad2_before if 'pad2_before' in locals() else np.zeros(3, int))
    orig_start = (start - total_pad).astype(np.float64)
    if affine is not None:
        new_affine = affine.copy()
        # adjust translation component
        new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ orig_start

    return crops, start, end, new_affine, orig_start.astype(int)


