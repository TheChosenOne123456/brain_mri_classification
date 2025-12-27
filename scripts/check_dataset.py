import sys
from pathlib import Path
import torch

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

from configs.global_config import *

def get_distribution(pt_path):
    if not pt_path.exists():
        return None
    
    try:
        # 加载数据集字典
        data = torch.load(pt_path, map_location="cpu")
        labels = data["labels"] # torch.Tensor
        
        total = labels.size(0)
        num_pos = (labels == 1).sum().item()
        num_neg = (labels == 0).sum().item()
        
        return {
            "total": total,
            "pos": num_pos,
            "neg": num_neg
        }
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return None

def main():
    print(f"Checking datasets in: {DATASET_ROOT.resolve()}\n")
    print(f"{'Sequence':<15} | {'Split':<6} | {'Total':<5} | {'Normal (0)':<12} | {'Meningitis (1)':<15}")
    print("-" * 75)

    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
        seq_dir_name = f"seq{seq_id}_{seq_name}"
        seq_dir = DATASET_ROOT / seq_dir_name

        if not seq_dir.exists():
            print(f"{seq_name:<15} | [MISSING DIRECTORY]")
            print("-" * 75)
            continue

        # 检查 train, val, test
        for split in ["train", "val", "test"]:
            pt_file = seq_dir / f"{split}.pt"
            stats = get_distribution(pt_file)

            if stats:
                total = stats['total']
                neg = stats['neg']
                pos = stats['pos']
                
                neg_pct = (neg / total * 100) if total > 0 else 0
                pos_pct = (pos / total * 100) if total > 0 else 0
                
                # 格式化输出
                # 第一行打印序列名，后续行留空
                display_name = f"{seq_id}-{seq_name}" if split == "train" else ""
                
                print(f"{display_name:<15} | {split:<6} | {total:<5} | {neg:<3} ({neg_pct:>4.1f}%) | {pos:<3} ({pos_pct:>4.1f}%)")
            else:
                print(f"{seq_name:<15} | {split:<6} | [FILE NOT FOUND]")
        
        print("-" * 75)

if __name__ == "__main__":
    main()