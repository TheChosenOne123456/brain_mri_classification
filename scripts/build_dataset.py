'''
根据预处理后的数据构造.pt数据集
'''
import random
import json

import torch

from utils.dataset import *


# ================== 主流程 ==================
def main():
    random.seed(RANDOM_SEED)

    for seq_id, seq_name in SEQ_IDS.items():
        print(f"\n=== Processing sequence {seq_id}: {seq_name} ===")

        normal_cases = collect_cases(seq_id, label=0)
        meningitis_cases = collect_cases(seq_id, label=1)

        all_cases = normal_cases + meningitis_cases
        random.shuffle(all_cases)

        if len(all_cases) == 0:
            print("  [SKIP] No cases found.")
            continue

        n_total = len(all_cases)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_cases = all_cases[:n_train]
        val_cases = all_cases[n_train : n_train + n_val]
        test_cases = all_cases[n_train + n_val:]

        # 构建数据集
        train_dataset = build_dataset(train_cases)
        val_dataset = build_dataset(val_cases)
        test_dataset = build_dataset(test_cases)

        # 输出目录
        out_dir = DATASET_ROOT / f"seq{seq_id}_{seq_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 保存 pt
        torch.save(train_dataset, out_dir / "train.pt")
        torch.save(val_dataset, out_dir / "val.pt")
        torch.save(test_dataset, out_dir / "test.pt")

        # 保存 split（与 pt 同目录）
        split_info = {
            "sequence_id": seq_id,
            "sequence_name": seq_name,
            "seed": RANDOM_SEED,
            "num_total": len(all_cases),
            "num_train": len(train_cases),
            "num_val": len(val_cases),
            "num_test": len(test_cases),
            "train_case_ids": [c["case_id"] for c in train_cases],
            "val_case_ids": [c["case_id"] for c in val_cases],
            "test_case_ids": [c["case_id"] for c in test_cases],
        }

        with open(out_dir / f"split_seed{RANDOM_SEED}.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

        print(f"  total: {len(all_cases)} | train: {len(train_cases)} | val: {len(val_cases)} | test: {len(test_cases)}")
        print(f"  saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()