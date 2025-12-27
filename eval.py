'''
根据train.py训练的单通道模型，分别进行评估，用参数--seq指定序列
'''

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.train_config import *
from configs.global_config import *

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
    seq_id = args.seq
    seq_idx = seq_id - 1
    seq_name = ALL_SEQUENCES[seq_idx]

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

    misclassified_cases = []

    with torch.no_grad():
        for x, y, case_ids in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            preds_cpu = preds.cpu().numpy()
            labels_cpu = y.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            # --------- 收集误判 case ---------
            for cid, p, gt in zip(case_ids, preds_cpu, labels_cpu):
                if p != gt:
                    misclassified_cases.append({
                        "case_id": cid,
                        "gt": int(gt),
                        "pred": int(p),
                    })

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

    # ---------- 打印误判 case ----------
    print("\n===== Misclassified Cases =====")
    print(f"Total misclassified: {len(misclassified_cases)}")

    for item in misclassified_cases:
        print(
            f"CaseID: {item['case_id']} | "
            f"GT: {item['gt']} | Pred: {item['pred']}"
        )


# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=int,
        required=True,
        choices=range(1, NUM_SEQUENCES + 1),
        help="Which MRI sequence to evaluate (1~{NUM_SEQUENCES})",
    )
    args = parser.parse_args()

    main(args)
