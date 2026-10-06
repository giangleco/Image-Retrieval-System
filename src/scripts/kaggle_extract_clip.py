# ============================================================================
#  TRÍCH XUẤT EMBEDDING CLIP CHO CIFAR-10 TRÊN KAGGLE (GPU)  ~1-2 phút
# ============================================================================
#  CÁCH DÙNG:
#   1) Tạo Notebook mới trên Kaggle.
#   2) Settings (bên phải):  Accelerator = GPU (T4 x2 hoặc P100)
#                            Internet    = On   (để tải CIFAR-10 + weights CLIP)
#   3) Dán TOÀN BỘ file này vào 1 cell và chạy.
#   4) Khi xong, vào panel "Output" -> tải file  features_clip.npy  về.
#   5) Chép features_clip.npy vào thư mục  features/  của dự án trên máy bạn.
#
#  LƯU Ý: phải dùng ĐÚNG model 'ViT-B-32' / 'openai' (giống app), và GIỮ nguyên
#  thứ tự train-rồi-test (shuffle=False) để khớp với image_list.txt & labels.npy.
# ============================================================================

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open-clip-torch==2.24.0"], check=True)

import numpy as np
import torch
import torchvision
import open_clip
from torch.utils.data import ConcatDataset, DataLoader

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 512                      # GPU mạnh -> batch lớn cho nhanh
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Thiết bị: {DEVICE} | Model: {MODEL_NAME}/{PRETRAINED}")

# --- 1. Tải CLIP ---
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model.eval().to(DEVICE)

# --- 2. Tải CIFAR-10 (train rồi test, KHÔNG shuffle) — preprocess bằng chính CLIP ---
train = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=preprocess)
test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
full  = ConcatDataset([train, test])      # GIỮ thứ tự: 50k train + 10k test = 60k
loader = DataLoader(full, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Tổng số ảnh: {len(full)}")

# --- 3. Encode toàn bộ ảnh -> embedding đã L2-normalize ---
feats = []
labels = []
with torch.no_grad():
    for bi, (images, labs) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        f = model.encode_image(images)
        f = f / f.norm(dim=-1, keepdim=True)          # L2-normalize (để cosine = inner product)
        feats.append(f.cpu().numpy().astype("float32"))
        labels.append(labs.numpy())
        if (bi + 1) % 10 == 0:
            print(f"   {min((bi+1)*BATCH_SIZE, len(full))}/{len(full)}")

features = np.vstack(feats).astype("float32")
labels = np.concatenate(labels)
np.save("features_clip.npy", features)
# (tùy chọn) lưu kèm labels để tự đối chiếu thứ tự — KHÔNG bắt buộc nếu đã có labels.npy
np.save("labels_clip_check.npy", labels)

print("XONG! features_clip.npy shape =", features.shape)   # kỳ vọng (60000, 512)
print("Tải features_clip.npy ở panel Output rồi chép vào thư mục features/ của dự án.")
