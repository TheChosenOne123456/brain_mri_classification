import random
import torch
from torch.utils.data import Dataset
from utils.dataset import load_nii_as_tensor  # 新增 NIfTI 读取方法引入

# ================== 一些工具函数 ==================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PTDataset(Dataset):
    def __init__(self, cases):
        self.cases = cases
        # 暴露出 labels 属性，供外部 train 脚本计算 class_weights 使用
        self.labels = [c["label"] for c in cases]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        # 核心：真正到取数据时才依据路径去拿取和解码大文件
        case_info = self.cases[idx]
        image_tensor = load_nii_as_tensor(case_info["nii_path"])
        label = torch.tensor(case_info["label"], dtype=torch.long)
        
        return image_tensor, label, case_info["case_id"]


def load_pt_dataset(pt_path):
    # 此处读取到的只是个几十K的文本字典
    data = torch.load(pt_path, weights_only=False)
    # 将字典中的路径和索引喂给代理 Dataset
    return PTDataset(data["cases"])
