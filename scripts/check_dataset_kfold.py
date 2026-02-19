'''
检查 K-Fold 数据集分布脚本
功能：
1. 直接加载 fold 下的 train.pt / val.pt / test.pt
2. 读取其中的 label 数据进行统计
3. 验证是否符合 global_config 中的配置
'''

import argparse
import torch
import sys
from pathlib import Path
from collections import Counter
import warnings

# 添加项目根目录到 sys.path，确保能导入 config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from configs.global_config import *

def check_fold_distribution(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    
    print(f"\n{'='*60}")
    print(f"Checking Dataset (from .pt files): {dataset_dir.name}")
    print(f"Path: {dataset_dir}")
    print(f"Configs: {NUM_CLASSES} Classes ({CLASS_NAMES})")
    print(f"{'='*60}")

    if not dataset_dir.exists():
        print(f"Error: Directory not found: {dataset_dir}")
        return

    # 遍历所有 Fold
    for k in range(1, K_FOLDS + 1):
        fold_name = f"fold{k}"
        fold_dir = dataset_dir / fold_name
        
        print(f"\n--- {fold_name} ---")
        if not fold_dir.exists():
            print(f"  [Warning] {fold_name} dir not found. Skipping.")
            continue

        stats = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter()
        }

        # 遍历三个集合文件
        for split_name in ["train", "val", "test"]:
            pt_path = fold_dir / f"{split_name}.pt"
            
            if not pt_path.exists():
                print(f"  [Warning] {split_name}.pt not found.")
                continue
            
            try:
                # 压制 torch.load 的 weights_only 警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # 根据 utils/dataset.py，加载得到的是一个字典
                    dataset_dict = torch.load(pt_path)
                
                # 直接从字典中获取标签 Tensor
                if "labels" in dataset_dict:
                    labels_tensor = dataset_dict["labels"]
                    labels_list = labels_tensor.tolist()
                    stats[split_name].update(labels_list)
                else:
                    print(f"  [Error] Key 'labels' not found in {split_name}.pt")
                
            except Exception as e:
                print(f"  [Error] Failed to load {split_name}.pt: {e}")

        # 打印表头
        header_labels = [f"{name}({idx})" for idx, name in enumerate(CLASS_NAMES)]
        header = f"  {'Split':<8} | " + " | ".join([f"{hl:<15}" for hl in header_labels]) + f" | {'Total':<8}"
        print(header)
        print(f"  {'-'*len(header)}")

        # 打印每行数据
        for split_name in ["train", "val", "test"]:
            row_str = f"  {split_name:<8} | "
            total_count = 0
            
            for label_id in range(NUM_CLASSES):
                count = stats[split_name].get(label_id, 0)
                row_str += f"{count:<15} | "
                total_count += count
            
            row_str += f"{total_count:<8}"
            print(row_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check K-Fold dataset class distribution from .pt files")
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Path to the sequence dataset directory (e.g., datasets/seq1_T1)"
    )
    
    args = parser.parse_args()
    
    check_fold_distribution(args.path)