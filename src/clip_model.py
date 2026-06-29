"""
Bộ mã hoá CLIP dùng chung cho cả trích xuất (clip_extractor.py) và truy vấn (main.py).

CLIP nhúng ẢNH và VĂN BẢN vào cùng một không gian vector 512 chiều, nên có thể:
  - tìm ảnh giống một ẢNH  (encode_image)
  - tìm ảnh khớp một MÔ TẢ (encode_text)  <-- tính năng mới

Chạy hoàn toàn offline (tải weights 1 lần). Model openai/ViT-B-32 huấn luyện bằng
tiếng Anh, nên truy vấn tiếng Việt được dịch nhanh sang tiếng Anh bằng từ điển nhỏ
(đủ cho 10 lớp CIFAR-10 + màu sắc/thuộc tính phổ biến). Có thể đổi model qua biến
môi trường CLIP_MODEL / CLIP_PRETRAINED.
"""
import os
import re
import torch
import numpy as np
import open_clip

MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")
EMBED_DIM = 512  # ViT-B-32

_DEVICE = torch.device("cpu")
_model = None
_preprocess = None
_tokenizer = None


def load():
    """Tải CLIP (chỉ 1 lần, lazy)."""
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        _model.eval().to(_DEVICE)
    return _model, _preprocess, _tokenizer


def encode_images(pil_list):
    """List ảnh PIL -> ma trận embedding đã L2-normalize, shape (N, 512) float32."""
    model, preprocess, _ = load()
    batch = torch.stack([preprocess(im) for im in pil_list]).to(_DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def encode_text(text, translate=True):
    """Một chuỗi mô tả -> vector embedding đã L2-normalize, shape (512,) float32."""
    model, _, tokenizer = load()
    query = translate_vi_en(text) if translate else text
    query = f"a photo of {query}".strip()
    tokens = tokenizer([query]).to(_DEVICE)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")[0]


# ---------------------------------------------------------------------------
#  Từ điển dịch nhanh Việt -> Anh (đủ cho phạm vi CIFAR-10)
#  Khoá nhiều từ đặt trước (xử lý theo độ dài giảm dần) để khớp đúng cụm.
# ---------------------------------------------------------------------------
VI_EN = {
    # 10 lớp CIFAR-10
    "máy bay": "airplane", "phi cơ": "airplane",
    "xe tải": "truck", "xe ben": "truck",
    "xe hơi": "car", "ô tô": "car", "oto": "car", "xe ô tô": "car",
    "con chim": "bird", "chim": "bird",
    "con mèo": "cat", "mèo": "cat",
    "con nai": "deer", "hươu": "deer", "nai": "deer",
    "con chó": "dog", "chó": "dog",
    "con ếch": "frog", "ếch": "frog",
    "con ngựa": "horse", "ngựa": "horse",
    "tàu thủy": "ship", "tàu thuyền": "ship", "con tàu": "ship",
    "thuyền": "ship", "tàu": "ship",
    # màu sắc
    "màu": "color", "đỏ": "red", "xanh dương": "blue", "xanh lá": "green",
    "xanh": "blue", "vàng": "yellow", "đen": "black", "trắng": "white",
    "cam": "orange", "nâu": "brown", "hồng": "pink", "tím": "purple", "xám": "gray",
    # hành động / bối cảnh / thuộc tính
    "bầu trời": "sky", "đang bay": "flying", "đang chạy": "running",
    "đang bơi": "swimming", "đang đứng": "standing", "đang nằm": "lying",
    "bay": "flying", "chạy": "running", "bơi": "swimming",
    "nhỏ": "small", "lớn": "big", "nước": "water", "cỏ": "grass", "đường": "road",
    # từ đệm -> bỏ
    "con": "", "cái": "", "chiếc": "", "một": "", "những": "", "đang": "",
    "hình ảnh": "", "hình": "", "bức ảnh": "", "tấm ảnh": "", "ảnh": "", "bức": "", "tấm": "",
}


def translate_vi_en(text):
    """Dịch thô tiếng Việt -> tiếng Anh theo từ điển (giữ nguyên nếu đã là tiếng Anh)."""
    s = (text or "").lower().strip()
    # thay cụm dài trước, cụm ngắn sau
    for vi in sorted(VI_EN, key=len, reverse=True):
        if vi in s:
            s = s.replace(vi, f" {VI_EN[vi]} ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or (text or "").strip()
