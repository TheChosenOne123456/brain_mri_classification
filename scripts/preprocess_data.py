'''
数据预处理，实现重采样、归一化和裁剪/填充
'''
import argparse
import shutil
from pathlib import Path
import re

from tqdm import tqdm

from configs.global_config import *

from utils.sequences import identify_sequence
from utils.data_scan import collect_cases
from utils.io import load_index, save_index, INDEX_FILE_NAME
from utils.resample import resample_image, save_image
from utils.intensity import normalize_intensity
from utils.spatial import center_crop_or_pad


def main(args):
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / INDEX_FILE_NAME

    # ===== 读取 / 初始化 index =====
    case_index = load_index(index_path)

    if case_index:
        max_case_id = max(case_index.values())
    else:
        max_case_id = 0

    # ===== 定义数据源 =====
    meningitis_dirs = [
        raw_root / subdir for subdir in MENINGITIS_SUBDIRS
    ]

    normal_dirs = [
        raw_root / subdir for subdir in NORMAL_SUBDIRS
    ]

    # ===== 输出目录初始化 =====
    for label_name in ["0_normal", "1_meningitis"]:
        for seq_id in range(1, 4):
            (out_root / label_name / str(seq_id)).mkdir(parents=True, exist_ok=True)

    def process_cases(case_dirs, label_name, desc):
        nonlocal max_case_id

        for case_dir in tqdm(case_dirs, desc=desc):
            # case_key = str(case_dir.resolve())
            # 只取文件夹末尾数字
            folder_name = case_dir.name
            # 提取连续数字，取最后一组
            match = re.findall(r'\d+', folder_name)
            if match:
                case_key = match[-1]
            else:
                case_key = folder_name  

            # ===== 已处理过，跳过 =====
            if case_key in case_index:
                continue

            max_case_id += 1
            case_id = f"{max_case_id:04d}"
            case_index[case_key] = max_case_id

            for nii_file in case_dir.rglob("*.nii*"):
                seq_id = identify_sequence(nii_file)
                if seq_id is None:
                    continue

                # ===== 新增: 重采样 + 强度归一化 =====
                resampled_img = resample_image(nii_file, target_spacing=TARGET_SPACING, is_label=False)
                normalized_img = normalize_intensity(resampled_img)

                fixed_img = center_crop_or_pad(normalized_img, TARGET_SHAPE)

                out_name = f"case_{case_id}_{seq_id}.nii.gz"
                out_path = out_root / label_name / str(seq_id) / out_name

                if not out_path.exists():
                    save_image(fixed_img, out_path)

    # ===== 处理脑膜炎 =====
    meningitis_cases = collect_cases(meningitis_dirs)
    process_cases(meningitis_cases, "1_meningitis", "Processing meningitis cases")

    # ===== 处理正常 =====
    normal_cases = collect_cases(normal_dirs)
    process_cases(normal_cases, "0_normal", "Processing normal cases")

    # ===== 保存 index =====
    save_index(case_index, index_path)

    print(f"Finished. Total cases indexed: {max_case_id}")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess brain MRI data")
    parser.add_argument(
        "--raw_root",
        type=str,
        default=RAW_DATA_PATH,
        help="Path to raw brainMRI directory"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=PROCESSED_DATA_PATH,
        help="Output processed data directory"
    )
    args = parser.parse_args()

    main(args)
