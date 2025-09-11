from scipy.ndimage import zoom

def downsample(b, f, m, y, factor=2, order=1):
    # 1) downsample each volume
    b_ds = zoom(b, zoom=(1/factor,)*3, order=order) if b is not None else None  # trilinear
    f_ds = zoom(f, zoom=(1/factor,)*3, order=order) if f is not None else None
    m_ds = zoom(m, zoom=(1/factor,)*3, order=order) if m is not None else None
    y_ds = zoom(y, zoom=(1/factor,)*3, order=order) if y is not None else None

    return b_ds, f_ds, m_ds, y_ds