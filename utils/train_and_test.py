import random
import torch
from torch.utils.data import TensorDataset

# ================== 一些工具函数 ==================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pt_dataset(pt_path):
    data = torch.load(pt_path, map_location="cpu")
    x = data["images"]    # [N, 1, D, H, W]
    y = data["labels"]    # [N]
    return TensorDataset(x, y)