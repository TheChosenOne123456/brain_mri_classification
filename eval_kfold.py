'''
K-Fold 评估脚本：
功能与 eval.py 类似，但支持 K-Fold 交叉验证模型。
- 如果指定 --fold N，则只评估第 N 折。
- 如果不指定 --fold，则自动评估所有 fold 并计算平均指标。
[适配新服务器：8x RTX 3080]
'''

import argparse
import numpy as np
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_single_fold(seq_idx, model_name, fold_idx, ModelClass):
    """
    评估单个 Fold 的核心函数
    """
    seq_name = ALL_SEQUENCES[seq_idx]
    
    # 路径构造
    dataset_dir = DATASET_DIRS[seq_idx] / f"fold{fold_idx}"
    ckpt_dir = CKPT_DIRS[seq_idx] / model_name
    ckpt_path = ckpt_dir / f"fold{fold_idx}_model_best.pth"

    # 检查存在性
    if not dataset_dir.exists():
        print(f"\n[Warning] Dataset for fold {fold_idx} not found at {dataset_dir}. Skipping.")
        return None
    if not ckpt_path.exists():
        print(f"\n[Warning] Checkpoint for fold {fold_idx} not found at {ckpt_path}. Skipping.")
        return None

    print(f"\n{'='*20} Evaluating Fold {fold_idx} {'='*20}")
    print(f"Sequence: {seq_name} | Model: {model_name}")

    # ---------- 加载 test 数据 ----------
    test_set = load_pt_dataset(dataset_dir / "test.pt")
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ---------- 加载模型 ----------
    model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    # [修改点] 启用多卡 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

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

    # ---------- 打印结果 (保持与 eval.py 格式一致) ----------
    print("\n===== Test Results =====")
    print(f"Sequence      : {seq_name} (Fold {fold_idx})")
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

    if len(misclassified_cases) > 0:
        for item in misclassified_cases:
            print(
                f"CaseID: {item['case_id']} | "
                f"GT: {item['gt']} | Pred: {item['pred']}"
            )
    else:
        print("None")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": avg_loss
    }


# ================== 主流程 ==================
def main(args):
    set_seed(SEED)

    # ---------- 解析参数 ----------
    seq_id = args.seq
    seq_idx = seq_id - 1
    seq_name = ALL_SEQUENCES[seq_idx]
    
    model_name = args.model
    if model_name == "cnn3d":
        ModelClass = Simple3DCNN
    elif model_name == "ResNet":
        ModelClass = ResNet10
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\n>>> Starting K-Fold Evaluation for Sequence {seq_id}: {seq_name} <<<")

    # ---------- 确定要评估的 fold 列表 ----------
    if args.fold is not None:
        folds_to_run = [args.fold]
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        folds_to_run = range(1, K_FOLDS + 1)
        print(f"Mode: All {K_FOLDS} Folds Average")

    metrics_history = []

    # ---------- 循环评估 ----------
    for k in folds_to_run:
        res = evaluate_single_fold(seq_idx, model_name, k, ModelClass)
        if res:
            metrics_history.append(res)
    
    # ---------- 这里如果是多折评估，打印平均值 ----------
    if len(metrics_history) > 1:
        print("\n" + "="*50)
        print(f"   K-FOLDS AVERAGE REPORT ({len(metrics_history)} folds)   ")
        print("="*50)

        avg_acc = np.mean([r['acc'] for r in metrics_history])
        std_acc = np.std([r['acc'] for r in metrics_history])
        
        avg_f1 = np.mean([r['f1'] for r in metrics_history])
        std_f1 = np.std([r['f1'] for r in metrics_history])
        
        avg_prec = np.mean([r['precision'] for r in metrics_history])
        std_prec = np.std([r['precision'] for r in metrics_history])
        
        avg_rec = np.mean([r['recall'] for r in metrics_history])
        std_rec = np.std([r['recall'] for r in metrics_history])

        print(f"Sequence      : {seq_name}")
        print(f"Model         : {model_name}")
        print("-" * 40)
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
        print("-" * 40)
        print(f"{'Accuracy':<15} | {avg_acc:.4f}     | ±{std_acc:.4f}")
        print(f"{'Precision':<15} | {avg_prec:.4f}     | ±{std_prec:.4f}")
        print(f"{'Recall':<15} | {avg_rec:.4f}     | ±{std_rec:.4f}")
        print(f"{'F1-Score':<15} | {avg_f1:.4f}     | ±{std_f1:.4f}")
        print("-" * 40)
    elif len(metrics_history) == 0:
        print("\n[Error] No folds were successfully evaluated.")

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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn3d", "ResNet"],
        help="Which model architecture to use",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to evaluate (1~{K_FOLDS}). If not set, run all folds.",
    )
    args = parser.parse_args()

    main(args)