'''
K-Fold 数据集构建脚本：
无论原始分布如何，保证所有序列使用相同的Case划分。
生成的目录结构：
dataset/seq1_T1/fold1/train.pt
dataset/seq1_T1/fold1/split.json
...
'''
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split

# 导入复用函数
from utils.dataset import build_dataset, collect_cases_by_seq
from configs.global_config import *

def main():
    # 保证划分可复现
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 1. 收集并对齐所有序列的 Case
    print("\n[STEP 1] Collecting and aligning cases...")
    seq_case_maps = {}
    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
        seq_case_maps[seq_id] = collect_cases_by_seq(seq_id)
        print(f"  Seq {seq_id} ({seq_name}): {len(seq_case_maps[seq_id])} cases")
    
    # 取交集
    all_case_ids = sorted(list(set.intersection(*[set(cases.keys()) for cases in seq_case_maps.values()])))
    print(f"  Total complete cases: {len(all_case_ids)}")
    
    if len(all_case_ids) == 0:
        print("[ERROR] No common cases found across sequences.")
        return

    # 转为 numpy array 方便用 sklearn 切分
    all_case_ids_np = np.array(all_case_ids)

    # 2. 初始化 K-Fold
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # 3. 开始切分并保存
    print(f"\n[STEP 2] Building {K_FOLDS}-Fold datasets...")

    # enumerate 从 1 开始，我们习惯用 fold1 ~ fold5
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(all_case_ids_np), start=1):
        print(f"\n--- Processing Fold {fold_idx}/{K_FOLDS} ---")
        
        # 此时有了 Train+Val 的索引 和 Test 的索引
        # 我们还需要从 Train+Val 中切出一部分作为 Validation 用于早停
        #  Val 占 (Train+Val) 的 K_FOLDS_VAL_RATIO 比例
        train_sub_idx, val_sub_idx = train_test_split(
            train_val_idx, 
            test_size=K_FOLDS_VAL_RATIO, 
            random_state=SEED,
            shuffle=True
        )

        fold_train_ids = all_case_ids_np[train_sub_idx]
        fold_val_ids   = all_case_ids_np[val_sub_idx]
        fold_test_ids  = all_case_ids_np[test_idx]

        print(f"  Train: {len(fold_train_ids)} | Val: {len(fold_val_ids)} | Test: {len(fold_test_ids)}")

        # 对每个序列分别构建 dataset
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
            seq_cases = seq_case_maps[seq_id]
            
            # 根据 ID 提取 Case 对象
            train_data = build_dataset([seq_cases[cid] for cid in fold_train_ids])
            val_data   = build_dataset([seq_cases[cid] for cid in fold_val_ids])
            test_data  = build_dataset([seq_cases[cid] for cid in fold_test_ids])

            # 保存路径： datasets/seq1_T1/fold1/
            fold_dir = DATASET_ROOT / (f"seq{seq_id}_{seq_name}") / f"fold{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            torch.save(train_data, fold_dir / "train.pt")
            torch.save(val_data,   fold_dir / "val.pt")
            torch.save(test_data,  fold_dir / "test.pt")
            
            # --- 保存 Split 信息 (关键步骤) ---
            split_info = {
                "fold": fold_idx,
                "sequence": seq_name,
                "train_ids": fold_train_ids.tolist(),
                "val_ids": fold_val_ids.tolist(),
                "test_ids": fold_test_ids.tolist()
            }
            with open(fold_dir / "split.json", "w", encoding='utf-8') as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)
            
        print(f"  Saved fold {fold_idx} data for all sequences.")

    print("\n[SUCCESS] K-Fold datasets building finished.")

if __name__ == "__main__":
    main()