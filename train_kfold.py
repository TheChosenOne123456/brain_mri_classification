'''
K-Fold 训练脚本：指定 --fold 参数 (1~5) 进行训练
模型将保存为 fold{k}_model_best.pth
[适配新服务器：8x RTX 3080]
'''
import argparse
import sys
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
# 引入 AMP 模块 (兼容新旧版本写法)
try:
    from torch.amp import GradScaler, autocast
    scaler_args = {'device': 'cuda'}
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    scaler_args = {}

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
            # 验证时也开启 autocast 以节省显存
            with autocast(**({'device_type': 'cuda'} if 'device' in scaler_args else {})):
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

    model = ModelClass(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # 启用多卡 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    all_labels = train_set.labels.tolist()
    class_counts = torch.bincount(torch.tensor(all_labels), minlength=NUM_CLASSES)
    
    # 避免除以 0 (虽然理论上不会有空类)
    total_samples = len(all_labels)
    raw_weights = total_samples / (NUM_CLASSES * class_counts.float() + 1e-6)
    class_weights = torch.pow(raw_weights, 0.5)
    
    # 将权重转到 device
    class_weights = class_weights.to(DEVICE)
    
    print(f"Class Weights: {class_weights}") # 打印出来确认一下

    # [修改] 使用加权 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 初始化混合精度 Scaler (使用兼容写法)
    scaler = GradScaler(**scaler_args)

    # 训练循环 &早停
    best_val_loss = float("inf")
    best_val_f1 = 0.0  # 初始化为 0，越大越好
    patience_counter = 0
    best_epoch = 0

    # 模型保存文件名区分 fold
    best_model_path = ckpt_dir / f"fold{current_fold}_model_best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0 
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Fold {current_fold} Ep {epoch}", leave=False)
        for x, y, _ in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            # AMP 前向传播
            # device_type='cuda' 用于新版 torch.amp.autocast，旧版不需要参数但兼容性不同
            # 这里做个简单的兼容处理
            if 'device' in scaler_args:
                actx = autocast(device_type='cuda')
            else:
                actx = autocast()

            with actx:
                logits = model(x)
                loss = criterion(logits, y)
            
            # AMP 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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

        # 收集预测结果和真实标签
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # 验证集也开启 autocast 节省显存
                if 'device' in scaler_args:
                    actx = autocast(device_type='cuda')
                else:
                    actx = autocast()
                
                with actx:
                    logits = model(x)
                    loss = criterion(logits, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # [新增] 计算 Macro F1
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        # --- 打印格式 ---
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}   "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")

        # --- Early Stopping check with MIN_EPOCHS ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            
            # 保存时剥离 DataParallel 包装，否则以后加载会报错
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            
            torch.save({
                "model_state": model_to_save.state_dict(),
                "fold": current_fold,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1 # 新增记录
            }, best_model_path)
            # 即使在 MIN_EPOCHS 内，也保存更好的模型
        else:
            # 只有当超过最小训练轮数后，才开始消耗耐心
            if epoch > MIN_EPOCHS:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n[Early Stopping] Fold {current_fold} at epoch {epoch}. "
                          f"Best Val F1: {best_val_f1:.4f} (Ep {best_epoch})")
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