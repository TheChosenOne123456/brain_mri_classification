'''
根据预处理后的数据构造.pt数据集（case 级别统一划分）
'''
import random
import json

import torch

from utils.dataset import build_dataset, collect_cases_by_seq
from configs.global_config import *


def main():
    random.seed(SEED)

    # ============================================================
    # 1. 收集所有序列的 case（按 case_id 对齐）
    # ============================================================
    print("\n[STEP 1] Collecting cases from all sequences...")

    seq_case_maps = {}
    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
        seq_case_maps[seq_id] = collect_cases_by_seq(seq_id)
        print(f"  {seq_name}: {len(seq_case_maps[seq_id])} cases")

    # ============================================================
    # 2. 只保留「所有序列都存在」的 case
    # ============================================================
    print("\n[STEP 2] Filtering complete cases...")

    all_case_ids = set.intersection(
        *[set(cases.keys()) for cases in seq_case_maps.values()]
    )

    all_case_ids = sorted(all_case_ids)

    if len(all_case_ids) == 0:
        print("[ERROR] No complete cases found.")
        return

    print(f"  Complete cases: {len(all_case_ids)}")

    # ============================================================
    # 3. case 级别统一划分
    # ============================================================
    random.shuffle(all_case_ids)

    n_total = len(all_case_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_ids = all_case_ids[:n_train]
    val_ids   = all_case_ids[n_train:n_train + n_val]
    test_ids  = all_case_ids[n_train + n_val:]

    print(f"\n[STEP 3] Split cases")
    print(f"  total: {n_total} | train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")

    # ============================================================
    # 4. 按序列构建 pt 数据集（使用同一组 case_id）
    # ============================================================
    print("\n[STEP 4] Building datasets per sequence...")

    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
        print(f"\n=== Processing sequence {seq_id}: {seq_name} ===")

        seq_cases = seq_case_maps[seq_id]

        train_cases = [seq_cases[cid] for cid in train_ids]
        val_cases   = [seq_cases[cid] for cid in val_ids]
        test_cases  = [seq_cases[cid] for cid in test_ids]

        train_dataset = build_dataset(train_cases)
        val_dataset   = build_dataset(val_cases)
        test_dataset  = build_dataset(test_cases)

        out_dir = DATASET_ROOT / f"seq{seq_id}_{seq_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train_dataset, out_dir / "train.pt")
        torch.save(val_dataset, out_dir / "val.pt")
        torch.save(test_dataset, out_dir / "test.pt")

        split_info = {
            "sequence_id": seq_id,
            "sequence_name": seq_name,
            "seed": SEED,
            "num_total": n_total,
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "num_test": len(test_ids),
            "train_case_ids": train_ids,
            "val_case_ids": val_ids,
            "test_case_ids": test_ids,
        }

        with open(out_dir / f"split_seed{SEED}.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

        print(f"  saved to: {out_dir.resolve()}")

    print("\n[SUCCESS] All sequences processed with aligned splits.")


if __name__ == "__main__":
    main()
