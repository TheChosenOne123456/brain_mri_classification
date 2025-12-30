'''
根据train.py训练的多个单通道模型，进行投票评估
'''

import argparse
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


# ================== 投票评估流程 ==================
def main(args):
    set_seed(SEED)

    # ---------- 选择模型 ----------
    model_name = args.model
    if model_name == "cnn3d":
        ModelClass = Simple3DCNN
    elif model_name == "ResNet":
        ModelClass = ResNet10
    else:
        raise ValueError(f"Unknown model: {model_name}")

    num_seq = len(ALL_SEQUENCES)
    assert num_seq >= 2, "Voting requires at least 2 sequences."

    print("\n=== Voting Evaluation ===")
    print(f"Model architecture: {model_name}")
    print(f"Sequences: {ALL_SEQUENCES}")

    # ------------------------------------------------
    # 1. 加载 test 数据（以第一个序列为基准）
    # ------------------------------------------------
    test_sets = []
    test_loaders = []

    for dataset_dir in DATASET_DIRS:
        test_set = load_pt_dataset(dataset_dir / "test.pt")
        test_sets.append(test_set)

        test_loaders.append(
            DataLoader(
                test_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )
        )

    num_samples = len(test_sets[0])
    for i, ds in enumerate(test_sets):
        assert len(ds) == num_samples, (
            f"Test set size mismatch at seq {ALL_SEQUENCES[i]}"
        )

    # ------------------------------------------------
    # 2. 加载所有模型
    # ------------------------------------------------
    models = []

    for seq_name, ckpt_dir in zip(ALL_SEQUENCES, CKPT_DIRS):
        ckpt_path = ckpt_dir / model_name / "model_best.pth"
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

        model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        models.append(model)
        print(f"Loaded {model_name} model for {seq_name}")

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------
    # 3. 测试 + 投票
    # ------------------------------------------------
    all_labels = []
    all_preds = []

    misclassified = []        # 投票错
    corrected_by_vote = []    # 被投票纠正

    total_loss = 0.0

    with torch.no_grad():
        for batches in zip(*test_loaders):
            xs = []
            ys = None
            case_ids = None

            for batch in batches:
                x, y, cids = batch
                xs.append(x.to(DEVICE))

                if ys is None:
                    ys = y.to(DEVICE)
                    case_ids = cids

            # 单模型预测
            logits_list = []
            preds_list = []

            for model, x in zip(models, xs):
                logits = model(x)
                logits_list.append(logits)
                preds_list.append(logits.argmax(dim=1))
                total_loss += criterion(logits, ys).item()

            # [num_seq, batch]
            preds_per_seq = torch.stack(preds_list, dim=0)

            # hard voting
            voted_preds, _ = torch.mode(preds_per_seq, dim=0)

            voted_preds_cpu = voted_preds.cpu().numpy()
            labels_cpu = ys.cpu().numpy()
            preds_per_seq_cpu = preds_per_seq.cpu().numpy()

            all_preds.extend(voted_preds_cpu)
            all_labels.extend(labels_cpu)

            # -------- case-level 分析 --------
            for i, cid in enumerate(case_ids):
                gt = labels_cpu[i]
                vote_pred = voted_preds_cpu[i]
                single_preds = preds_per_seq_cpu[:, i]

                single_correct = single_preds == gt

                if vote_pred != gt:
                    misclassified.append({
                        "case_id": cid,
                        "gt": int(gt),
                        "vote_pred": int(vote_pred),
                        "single_preds": single_preds.tolist(),
                    })
                else:
                    # 至少一个单模态错，但投票对 → 被纠正
                    if not single_correct.all():
                        corrected_by_vote.append({
                            "case_id": cid,
                            "gt": int(gt),
                            "vote_pred": int(vote_pred),
                            "single_preds": single_preds.tolist(),
                        })

    avg_loss = total_loss / (len(test_loaders[0]) * num_seq)

    # ------------------------------------------------
    # 4. 指标
    # ------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # ------------------------------------------------
    # 5. 打印结果
    # ------------------------------------------------
    print("\n===== Test Results (Voting) =====")
    print(f"Sequences      : {', '.join(ALL_SEQUENCES)}")
    print(f"Test samples   : {num_samples}")
    print(f"Avg test loss  : {avg_loss:.4f}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")

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

    # ------------------------------------------------
    # 6. Case-level 诊断输出
    # ------------------------------------------------
    print("\n===== Misclassified by Voting =====")
    print(f"Total: {len(misclassified)}")
    for item in misclassified:
        print(
            f"CaseID: {item['case_id']} | "
            f"GT: {item['gt']} | Vote: {item['vote_pred']} | "
            f"Single preds: {item['single_preds']}"
        )

    print("\n===== Corrected by Voting =====")
    print(f"Total: {len(corrected_by_vote)}")
    for item in corrected_by_vote:
        print(
            f"CaseID: {item['case_id']} | "
            f"GT: {item['gt']} | Vote: {item['vote_pred']} | "
            f"Single preds: {item['single_preds']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn3d", "ResNet"],
        help="Which model architecture to use for voting",
    )
    args = parser.parse_args()

    main(args)