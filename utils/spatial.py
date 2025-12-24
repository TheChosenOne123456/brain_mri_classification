'''
定义center_crop_or_pad函数，统一图像尺寸
'''

import SimpleITK as sitk
import numpy as np

def center_crop_or_pad(img: sitk.Image, target_shape):
    """
    target_shape: (D, H, W)
    """
    arr = sitk.GetArrayFromImage(img)  # [D, H, W]
    d, h, w = arr.shape
    td, th, tw = target_shape

    # ---- crop ----
    d_start = max((d - td) // 2, 0)
    h_start = max((h - th) // 2, 0)
    w_start = max((w - tw) // 2, 0)

    arr = arr[
        d_start:d_start + td,
        h_start:h_start + th,
        w_start:w_start + tw
    ]

    # ---- pad ----
    pd = max(td - arr.shape[0], 0)
    ph = max(th - arr.shape[1], 0)
    pw = max(tw - arr.shape[2], 0)

    arr = np.pad(
        arr,
        (
            (pd // 2, pd - pd // 2),
            (ph // 2, ph - ph // 2),
            (pw // 2, pw - pw // 2),
        ),
        mode="constant",
        constant_values=0
    )

    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())

    return out
