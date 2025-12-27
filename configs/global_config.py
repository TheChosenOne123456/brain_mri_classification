'''
项目的全局设置，用户可以根据需要进行修改
'''
from pathlib import Path

SEED = 42

# ========== Task ==========
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Meningitis"]
# 以上暂时不能改变

# ========== Sequences ==========
ALL_SEQUENCES = ["T1", "T2", "FLAIR"]
NUM_SEQUENCES = len(ALL_SEQUENCES)

# ========== Paths ==========
RAW_DATA_PATH = Path("/home/tbing/projects/data/brainMRI")    # 原始数据根目录
# PROCESSED_DATA_PATH = "data/processed"
PROCESSED_DATA_PATH = Path("data")

MENINGITIS_SUBDIRS = [
    "脑膜病变图像/脑膜炎主诊",
    "脑膜病变图像/脑膜炎次诊",
]

NORMAL_SUBDIRS = [
    "正常头颅MRI",
]

# ========== Preprocessing ==========
TARGET_SPACING = (1.0, 1.0, 1.0)  # (D, H, W)
TARGET_SHAPE = (160, 192, 160)

# ========== Dataset ==========
DATASET_ROOT = Path("datasets")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
