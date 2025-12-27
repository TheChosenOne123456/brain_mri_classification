import random
import torch
from torch.utils.data import TensorDataset

# ================== 一些工具函数 ==================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, case_ids):
        self.images = images
        self.labels = labels
        self.case_ids = case_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.case_ids[idx]


def load_pt_dataset(pt_path):
    data = torch.load(pt_path)
    return PTDataset(
        data["images"],
        data["labels"],
        data["case_ids"]
    )
