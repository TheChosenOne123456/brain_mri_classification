# read_kfold_pth.py
"""
读取并打印 .pth 权重文件中的所有元信息（不包含模型参数）
兼容：
- 非 KFold 模型（model_best.pth / model_final.pth）
- KFold 模型（foldX_best_model.pth）
"""

import torch
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)


def read_pth(pth_path: Path):
    print("=" * 80)
    print(f"[FILE] {pth_path}")

    ckpt = torch.load(pth_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        print("该文件不是标准 checkpoint(dict)")
        return

    # ---------- 自动过滤权重 ----------
    meta = {k: v for k, v in ckpt.items() if k != "model_state"}

    # ---------- 打印 ----------
    for k, v in meta.items():
        print(f"{k:15s}: {v}")

    # ---------- 自动判断 ----------
    if any("fold" in k.lower() for k in meta.keys()):
        print("→ Detected: KFold checkpoint")
    else:
        print("→ Detected: Single-run checkpoint")

    print("=" * 80 + "\n")


def main(path: Path):
    if path.is_file():
        read_pth(path)
    elif path.is_dir():
        pths = sorted(path.glob("*.pth"))
        if not pths:
            print(f"No .pth files found in {path}")
            return
        for pth in pths:
            read_pth(pth)
    else:
        raise FileNotFoundError(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read meta info from .pth files")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to .pth file or directory containing .pth files",
    )
    args = parser.parse_args()

    main(Path(args.path))
