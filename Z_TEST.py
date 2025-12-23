import SimpleITK as sitk
from pathlib import Path

img = sitk.ReadImage('../data/brainMRI/脑膜病变图像/脑膜炎主诊/陈祖桥1234 CHEN ZU QIAO 2514007579/4/DWI.nii')
print(img.GetDimension())