"""
Trích xuất embedding CLIP cho toàn bộ ảnh CIFAR-10 -> features/features_clip.npy

Thứ tự ảnh GIỮ NGUYÊN như extract_resnet.py (train rồi test, shuffle=False),
nên chỉ số i trong features_clip.npy khớp với image_list.txt và labels.npy đã có.
=> Chạy extract_resnet.py TRƯỚC (để có image_list.txt + labels.npy), rồi chạy file này.

Mẹo: trên CPU rất chậm (~1 giờ cho 60k ảnh). Có thể trích một phần để thử nhanh:
    CLIP_MAX_IMAGES=6000 uv run python src/scripts/extract_clip.py
"""
import os
import sys
from pathlib import Path

# Script chạy trực tiếp (uv run python src/scripts/...) nên phải tự đưa src/ vào
# đường dẫn import thì mới dùng chung được cấu hình ở core/config.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torchvision
from torch.utils.data import ConcatDataset

from embedding import clip_model

DATA_ROOT = str(config.DATA_DIR)
config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
CLIP_FEATURES_PATH = config.CLIP_FEATURES_NPY

BATCH_SIZE = 128


def main():
    print(">>> TRÍCH XUẤT EMBEDDING CLIP CHO CIFAR-10")
    print(f"[*] Model: {clip_model.MODEL_NAME} / {clip_model.PRETRAINED}")

    # transform=None -> trả về ảnh PIL gốc (CLIP tự tiền xử lý bằng preprocess riêng)
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=None)
    full = ConcatDataset([trainset, testset])  # GIỮ thứ tự: train rồi test

    total = len(full)
    limit = int(os.environ.get("CLIP_MAX_IMAGES", "0")) or total
    limit = min(limit, total)
    if limit < total:
        print(f"[!] Chế độ thử nghiệm: chỉ trích {limit}/{total} ảnh "
              f"(đặt CLIP_MAX_IMAGES rỗng để trích đủ).")

    # Tải model 1 lần (sẽ tải weights nếu chưa có)
    clip_model.load()

    feats_chunks = []
    batch = []
    done = 0
    for i in range(limit):
        img, _ = full[i]            # (PIL, label) — bỏ label, đã có labels.npy
        batch.append(img.convert("RGB"))
        if len(batch) == BATCH_SIZE:
            feats_chunks.append(clip_model.encode_images(batch))
            done += len(batch)
            batch = []
            if done % (BATCH_SIZE * 10) == 0:
                print(f"   Đã xử lý: {done:>6} / {limit} ảnh")
    if batch:
        feats_chunks.append(clip_model.encode_images(batch))
        done += len(batch)

    features = np.vstack(feats_chunks).astype("float32")
    np.save(CLIP_FEATURES_PATH, features)

    print("\n" + "=" * 60)
    print("HOÀN TẤT TRÍCH XUẤT CLIP!")
    print(f"File: {CLIP_FEATURES_PATH} (Shape: {features.shape})")
    if limit < total:
        print(f"LƯU Ý: mới trích {limit}/{total} ảnh — chạy lại không giới hạn để đủ kho.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
