'''
K-Fold 训练脚本：指定 --fold 参数 (1~5) 进行训练
模型将保存为 fold{k}_model_best.pth
'''
import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from configs.train_config import *
from configs.global_config import *
from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10
from utils.train_and_test import set_seed, load_pt_dataset

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# 辅助函数：计算准确率
def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def main(args):
    set_seed(SEED)
    
    current_fold = args.fold
    seq_id = args.seq
    seq_idx = seq_id - 1
    seq_name = ALL_SEQUENCES[seq_idx]
    model_name = args.model

    print(f"\n=== Training Fold {current_fold}/{K_FOLDS} | Seq: {seq_name} | Model: {model_name} ===")

    # 路径处理
    dataset_dir = DATASET_DIRS[seq_idx] / f"fold{current_fold}"
    ckpt_dir = CKPT_DIRS[seq_idx] / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"Error: Dataset for fold {current_fold} not found at {dataset_dir}")
        print("Please run 'python -m scripts.build_dataset_kfold' first.")
        sys.exit(1)

    # 加载数据
    train_set = load_pt_dataset(dataset_dir / "train.pt")
    val_set   = load_pt_dataset(dataset_dir / "val.pt")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 初始化模型
    if model_name == "cnn3d":
        ModelClass = Simple3DCNN
    elif model_name == "ResNet":
        ModelClass = ResNet10
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 训练循环 &早停
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # 模型保存文件名区分 fold
    best_model_path = ckpt_dir / f"fold{current_fold}_model_best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0 # 显式计算 Train Acc，避免二次遍历
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Fold {current_fold} Ep {epoch}", leave=False)
        for x, y, _ in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total

        # --- Val ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # --- 打印格式 ---
        # Epoch [1/100] train_loss: 0.4240 | train_acc: 0.8257   val_loss: 0.2778 | val_acc: 0.9155
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}   "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        # --- Early Stopping check with MIN_EPOCHS ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "fold": current_fold,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_acc": val_acc
            }, best_model_path)
            # 即使在 MIN_EPOCHS 内，也保存更好的模型
        else:
            # 只有当超过最小训练轮数后，才开始消耗耐心
            if epoch > MIN_EPOCHS:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n[Early Stopping] Fold {current_fold} at epoch {epoch}. "
                          f"Best Val Loss: {best_val_loss:.4f} (Ep {best_epoch})")
                    break
            else:
                 # 保护期内，重置耐心，确保出了保护期是满血状态
                 patience_counter = 0
    
    # 强制在结束时换行
    print(f"\n[Finished] Fold {current_fold} done. Model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, required=True, help="Sequence ID (1-3)")
    parser.add_argument("--model", type=str, required=True, choices=["cnn3d", "ResNet"])
    parser.add_argument("--fold", type=int, required=True, choices=range(1, K_FOLDS + 1), help=f"Fold ID (1-{K_FOLDS})")
    args = parser.parse_args()
    
    main(args)