'''
单通道MRI序列分类模型训练脚本，用参数--seq指定序列
'''

import argparse
# import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.train_config import *
from configs.global_config import *

from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10

from utils.train_and_test import set_seed, load_pt_dataset

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

from tqdm import tqdm


# ================== 主训练流程 ==================
def main(args):
    set_seed(SEED)

    # ---------- 选择序列 ----------
    seq_id = args.seq
    seq_idx = seq_id - 1
    seq_name = ALL_SEQUENCES[seq_idx]

    # ---------- 选择模型 ----------
    model_name = args.model
    if model_name == "cnn3d":
        ModelClass = Simple3DCNN
    elif model_name == "ResNet":
        ModelClass = ResNet10
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\n=== Training on sequence {seq_id}: {seq_name} ===")

    dataset_dir = DATASET_DIRS[seq_idx]
    ckpt_dir = CKPT_DIRS[seq_idx] / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 加载数据 ----------
    train_set = load_pt_dataset(dataset_dir / "train.pt")
    val_set   = load_pt_dataset(dataset_dir / "val.pt")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ---------- 模型 ----------
    model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ---------- 训练 ----------
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{NUM_EPOCHS}]", leave=False)
        for x, y, _ in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # --- validation ---
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
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        # --- early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存当前最佳模型
            torch.save({
                "model_state": model.state_dict(),
                "sequence_id": seq_id,
                "sequence_name": seq_name,
                "model_name": model_name, # 记录模型名
            }, ckpt_dir / "model_best.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
            break

    # ---------- 保存最终模型 ----------
    ckpt_path = ckpt_dir / "model_final.pth"
    torch.save({
        "model_state": model.state_dict(),
        "sequence_id": seq_id,
        "sequence_name": seq_name,
        "model_name": model_name, # 记录模型名
    }, ckpt_path)
    print(f"\n[SUCCESS] Model saved to {ckpt_path.resolve()}")


# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=int,
        required=True,
        choices=range(1, NUM_SEQUENCES + 1),
        help=f"Which MRI sequence to train (1~{NUM_SEQUENCES})",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn3d", "ResNet"],
        help="Which model architecture to use",
    )
    args = parser.parse_args()

    main(args)
