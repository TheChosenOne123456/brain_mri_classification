'''
单样本推理脚本：输入一个 NIfTI 文件，输出分类结果并保存 Grad-CAM 热力图
'''

import argparse
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from pathlib import Path
from monai.visualize import GradCAM

from configs.train_config import *
from configs.global_config import *

from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10
from utils.train_and_test import set_seed
from utils.resample import resample_image
from utils.spatial import center_crop_or_pad
from utils.intensity import normalize_intensity

# 忽略特定警告
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)


def preprocess_single_case(nii_path, do_preprocess=False):
    """
    对单个 NIfTI 文件进行预处理
    :param nii_path: 输入文件路径
    :param do_preprocess: 是否执行完整的预处理（重采样、归一化、裁剪）
    """
    if do_preprocess:
        print("  [Preprocess] Executing full preprocessing pipeline...")
        # 1. 重采样 (Resample)
        # resample_image 返回的是 SimpleITK Image 对象
        sitk_img = resample_image(nii_path, target_spacing=TARGET_SPACING, is_label=False)
        
        # 2. 归一化 (Normalize)
        # normalize_intensity 接收 sitk image, 返回 numpy array
        norm_np = normalize_intensity(sitk_img)
        
        # 3. 裁剪/填充 (Pad/Crop)
        # center_crop_or_pad 接收 numpy array, 返回 numpy array
        final_np = center_crop_or_pad(norm_np, TARGET_SHAPE)
        current_affine = np.eye(4)

    else:
        print("  [Preprocess] Loading preprocessed data directly...")
        # 假设输入已经是处理好的 (160, 192, 160)
        nii = nib.load(str(nii_path))
        final_np = nii.get_fdata(dtype=np.float32)
        current_affine = nii.affine

    # 转 Tensor [D, H, W] -> [1, 1, D, H, W] (Batch, Channel, ...)
    tensor = torch.from_numpy(final_np).unsqueeze(0).unsqueeze(0)
    
    return tensor, current_affine, final_np


def save_input_image(image_np, affine, output_path):
    """
    保存模型输入的底图 (Numpy Array)
    """
    nii_img = nib.Nifti1Image(image_np, affine)
    nib.save(nii_img, str(output_path))
    print(f"[Saved] Preprocessed input saved to: {output_path}")


def save_heatmap(heatmap, affine, output_path):
    """
    保存热力图为 NIfTI 文件
    """
    # heatmap 是 [1, 1, D, H, W]，转为 numpy [D, H, W]
    heatmap_np = heatmap.squeeze().cpu().numpy()
    
    # 创建 NIfTI 对象
    nii_img = nib.Nifti1Image(heatmap_np, affine)
    
    # 保存
    nib.save(nii_img, str(output_path))
    print(f"[Saved] Grad-CAM heatmap saved to: {output_path}")


def main(args):
    set_seed(SEED)
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ---------- 1. 准备配置 ----------
    seq_id = args.seq
    seq_idx = seq_id - 1
    seq_name = ALL_SEQUENCES[seq_idx]
    model_name = args.model

    print(f"\n=== Inference Mode ===")
    print(f"Input File    : {input_path.name}")
    print(f"Sequence      : {seq_name}")
    print(f"Model         : {model_name}")
    print(f"Preprocess    : {args.pre}")

    # ---------- 2. 加载模型 ----------
    ckpt_dir = CKPT_DIRS[seq_idx] / model_name
    ckpt_path = ckpt_dir / "model_best.pth"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_name == "cnn3d":
        model = Simple3DCNN(num_classes=NUM_CLASSES)
        target_layer = "conv3" # Simple3DCNN 的最后一层卷积
    elif model_name == "ResNet":
        model = ResNet10(num_classes=NUM_CLASSES)
        target_layer = "layer4" # ResNet 的最后一层残差块
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded from: {ckpt_path}")

    # ---------- 3. 预处理输入 ----------
    input_tensor, img_affine, img_np = preprocess_single_case(input_path, do_preprocess=args.pre)
    input_tensor = input_tensor.to(DEVICE)

    # ---------- 4. 推理预测 ----------
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()

    pred_class = CLASS_NAMES[pred_idx]
    
    print("\n===== Prediction Result =====")
    print(f"Class         : {pred_class} (Label: {pred_idx})")
    print(f"Probability   : {pred_prob:.4f}")
    print(f"Logits        : {logits.cpu().numpy().tolist()}")

    # ---------- 5. 生成 Grad-CAM 热力图 ----------
    print("\n===== Generating Grad-CAM =====")
    
    # 初始化 GradCAM
    cam = GradCAM(nn_module=model, target_layers=target_layer)
    
    # 解释模型预测出的那个类别 (而不是硬编码 1)
    target_class_idx = pred_idx 
    print(f"  [CAM] Explaining class: {CLASS_NAMES[target_class_idx]} ({target_class_idx})")
    
    result = cam(x=input_tensor, class_idx=target_class_idx)
    
    # # 归一化到 0-1 之间，方便可视化
    # result = result.detach().cpu().numpy()
    # res_min, res_max = result.min(), result.max()
    # result = (result - res_min) / (res_max - res_min + 1e-8)
    
    # ---------- 6. 保存结果 ----------
    INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 构造输出文件名
    original_name = input_path.name.replace(".nii.gz", "").replace(".nii", "")

    # 6.1 保存预处理后的输入图像 (作为底图)
    img_output_filename = f"{original_name}_{model_name}_{seq_name}_input.nii.gz"
    img_output_path = INFERENCE_OUTPUT_DIR / img_output_filename
    
    save_input_image(img_np, img_affine, img_output_path)

    # 6.2 保存热力图 (使用相同的 affine，确保对齐)
    cam_output_filename = f"{original_name}_{model_name}_{seq_name}_cam.nii.gz"
    cam_output_path = INFERENCE_OUTPUT_DIR / cam_output_filename
    
    save_heatmap(result, img_affine, cam_output_path)
    
    print("\nDone. Please open both files in ITK-SNAP to visualize.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on a single MRI case")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .nii.gz file",
    )
    parser.add_argument(
        "--seq",
        type=int,
        required=True,
        choices=range(1, NUM_SEQUENCES + 1),
        help=f"Which MRI sequence is this? (1~{NUM_SEQUENCES})",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn3d", "ResNet"],
        help="Which model architecture to use",
    )
    parser.add_argument(
        "--pre",
        action="store_true",
        help="Enable full preprocessing (resample, normalize, crop) if input is raw data",
    )
    
    args = parser.parse_args()
    main(args)