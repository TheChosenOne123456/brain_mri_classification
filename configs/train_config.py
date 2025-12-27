# configs/train_config.py
from pathlib import Path

# ================== 基本训练配置 ==================
SEED = 42
NUM_CLASSES = 2

NUM_EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = "cuda"   # "cuda" or "cpu"
NUM_WORKERS = 4

PATIENCE = 10  # 早停耐心值

# ================== 序列信息（固定顺序！） ==================
# SEQ_IDS = [1, 2, 3]
# SEQ_NAMES = ["T1", "T2", "FLAIR"]

# ================== 数据集路径（与 SEQ_IDS 一一对应） ==================
DATASET_DIRS = [
    Path("datasets/seq1_T1"),
    Path("datasets/seq2_T2"),
    Path("datasets/seq3_FLAIR"),
    # Path("datasets/seq4_DWI"),
    # Path("datasets/seq5_+C"),
]

# ================== 模型保存路径 ==================
CKPT_ROOT = Path("checkpoints")

# 每个序列单独一个目录
CKPT_DIRS = [
    CKPT_ROOT / "seq1_T1",
    CKPT_ROOT / "seq2_T2",
    CKPT_ROOT / "seq3_FLAIR",
    # CKPT_ROOT / "seq4_DWI",
    # CKPT_ROOT / "seq5_+C",
]
