from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm


# ================== 全局配置 ==================
SEQ_IDS = {
    1: "T1",
    2: "T2",
    3: "FLAIR",
    4: "DWI",
    5: "+C",
}

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

PROCESSED_ROOT = Path("data/processed")
DATASET_ROOT = Path("datasets")


# ================== 工具函数 ==================
def load_nii_as_tensor(nii_path: Path) -> torch.Tensor:
    """
    读取【已预处理】后的 nii.gz
    返回: torch.Tensor [1, D, H, W]
    注意：不做任何空间处理，只读 + 转 tensor
    """
    nii = nib.load(str(nii_path))
    data = nii.get_fdata(dtype=np.float32)
    return torch.from_numpy(data).unsqueeze(0)


def collect_cases(seq_id: int, label: int):
    """
    从 data/processed/{label}/{seq_id}/ 下收集该序列的所有 case
    """
    seq_dir = PROCESSED_ROOT / ("1_meningitis" if label == 1 else "0_normal") / str(seq_id)
    cases = []

    for nii_file in sorted(seq_dir.glob(f"case_*_{seq_id}.nii.gz")):
        case_id = nii_file.name.split("_")[1]
        cases.append({
            "case_id": case_id,
            "nii_path": nii_file,
            "label": label
        })

    return cases


def build_dataset(cases):
    images = []
    labels = []
    case_ids = []

    for c in tqdm(cases, desc="  loading cases", leave=False):
        images.append(load_nii_as_tensor(c["nii_path"]))
        labels.append(c["label"])
        case_ids.append(c["case_id"])

    return {
        "images": torch.stack(images, dim=0),   # [N, 1, D, H, W]
        "labels": torch.tensor(labels, dtype=torch.long),
        "case_ids": case_ids,
        "meta": {
            "num_samples": len(cases),
            "created_time": datetime.now().isoformat(),
            "seed": RANDOM_SEED,
        }
    }