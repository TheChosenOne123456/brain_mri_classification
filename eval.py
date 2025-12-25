# eval.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.train_config import *
from models.cnn3d import Simple3DCNN as Model
from utils.train_and_test import set_seed, load_pt_dataset

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ================== 主评估流程 ==================
def main(args):
    set_seed(SEED)

    # ---------- 选择序列 ----------
    seq_idx = args.seq - 1
    seq_id = SEQ_IDS[seq_idx]
    seq_name = SEQ_NAMES[seq_idx]

    print(f"\n=== Evaluating on sequence {seq_id}: {seq_name} ===")

    dataset_dir = DATASET_DIRS[seq_idx]
    ckpt_dir = CKPT_DIRS[seq_idx]

    # ---------- 加载 test 数据 ----------
    test_set = load_pt_dataset(dataset_dir / "test.pt")
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ---------- 加载模型 ----------
    ckpt_path = ckpt_dir / "model_best.pth"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model = Model(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # ---------- 测试 ----------
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # ---------- 计算指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # ---------- 打印结果 ----------
    print("\n===== Test Results =====")
    print(f"Sequence      : {seq_name}")
    print(f"Test samples  : {len(test_set)}")
    print(f"Test loss     : {avg_loss:.4f}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["Class 0", "Class 1"],
            digits=4,
            zero_division=0,
        )
    )


# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Which MRI sequence to evaluate (1~5)",
    )
    args = parser.parse_args()

    main(args)
