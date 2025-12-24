# Z_TEST.py
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import nibabel as nib


# ================== 基本配置 ==================
SEQ_ID = 3
SEQ_NAME = "FLAIR"
MAX_CASES = 10
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

PROCESSED_ROOT = Path("data/processed")
OUT_ROOT = Path("datasets/seq3_FLAIR")


# ================== 工具函数 ==================
def load_nii_as_tensor(nii_path: Path) -> torch.Tensor:
    """
    读取 nii.gz → torch.Tensor [1, D, H, W]
    """
    nii = nib.load(str(nii_path))
    data = nii.get_fdata(dtype=np.float32)

    # z-score 归一化（最简单、稳定）
    mean = data.mean()
    std = data.std()
    if std < 1e-6:
        data = data - mean
    else:
        data = (data - mean) / std

    # [D, H, W] → [1, D, H, W]
    data = torch.from_numpy(data).unsqueeze(0)
    return data


def collect_cases(label_dir: Path, label: int):
    """
    收集某一类下的 FLAIR case
    """
    seq_dir = label_dir / str(SEQ_ID)
    cases = []

    for nii_file in sorted(seq_dir.glob("case_*_3.nii.gz")):
        # case_0001_3.nii.gz → 0001
        case_id = nii_file.name.split("_")[1]
        cases.append({
            "case_id": case_id,
            "nii_path": nii_file,
            "label": label
        })

    return cases


# ================== 主逻辑 ==================
def main():
    random.seed(RANDOM_SEED)

    # 1. 收集 normal / meningitis
    normal_cases = collect_cases(PROCESSED_ROOT / "0_normal", label=0)
    meningitis_cases = collect_cases(PROCESSED_ROOT / "1_meningitis", label=1)

    all_cases = normal_cases + meningitis_cases
    random.shuffle(all_cases)

    # 2. 只取前 MAX_CASES 个
    all_cases = all_cases[:MAX_CASES]

    print(f"[INFO] Total cases used: {len(all_cases)}")
    for c in all_cases:
        print(f"  case {c['case_id']} label={c['label']}")

    # 3. 划分 train / test
    split_idx = int(len(all_cases) * TRAIN_RATIO)
    train_cases = all_cases[:split_idx]
    test_cases = all_cases[split_idx:]

    # 4. 构建 dataset
    def build_dataset(cases):
        images = []
        labels = []
        case_ids = []

        for c in cases:
            img = load_nii_as_tensor(c["nii_path"])
            images.append(img)
            labels.append(c["label"])
            case_ids.append(c["case_id"])

        return {
            "images": torch.stack(images, dim=0),  # [N, 1, D, H, W]
            "labels": torch.tensor(labels, dtype=torch.long),
            "case_ids": case_ids,
            "meta": {
                "sequence_id": SEQ_ID,
                "sequence_name": SEQ_NAME,
                "num_samples": len(cases),
                "intensity_norm": "zscore",
                "created_time": datetime.now().isoformat(),
                "seed": RANDOM_SEED,
            }
        }

    train_dataset = build_dataset(train_cases)
    test_dataset = build_dataset(test_cases)

    # 5. 保存
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(train_dataset, OUT_ROOT / "train.pt")
    torch.save(test_dataset, OUT_ROOT / "test.pt")

    print(f"\n[SUCCESS]")
    print(f"  Train samples: {len(train_cases)}")
    print(f"  Test samples : {len(test_cases)}")
    print(f"  Saved to     : {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()