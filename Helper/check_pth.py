# 查看权重文件pth的信息
import torch

# 替换为你的实际模型路径
model_path = "/home/ailab/projects/brain_mri_classification/version1/checkpoints/seq1_T1/FoundationModel/fold4_model_best.pth" 

# 加载 checkpoint
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

# 打印保存的信息
print(f"Fold: {checkpoint['fold']}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['val_loss']}")
print(f"Validation Accuracy: {checkpoint['val_acc']}")
print(f"Validation F1: {checkpoint['val_f1']}")