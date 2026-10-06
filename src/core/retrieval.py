"""
Tầng LÕI của hệ thống truy xuất ảnh — hoàn toàn không phụ thuộc web framework.

Chứa toàn bộ phần "máy tìm kiếm":
  - nạp đặc trưng + nhãn + ảnh base64 vào RAM, dựng FAISS index
  - trích đặc trưng ảnh upload bằng ResNet-18
  - mã hoá mô tả văn bản bằng CLIP (lazy — chỉ tải model khi có truy vấn đầu)
  - tìm top-K (run_search) và tính Precision@K / Recall@K / AP

Tách riêng như vậy để tầng API (api.py) chỉ còn lo việc nhận request &
trả JSON, và để có thể dùng lại lõi này trong notebook, script đánh giá, CLI...
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from core import config

try:
    import faiss
except ImportError as e:  # pragma: no cover
    raise ImportError("Không tìm thấy 'faiss'. Cài bằng: uv sync (hoặc pip install faiss-cpu)") from e

import torch
import torchvision.models as models
import torchvision.transforms as transforms


# ===========================================================
# 1. NẠP DỮ LIỆU & DỰNG FAISS INDEX
# ===========================================================
if not config.FEATURES_NPY.exists() or not config.LABELS_NPY.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {config.FEATURES_NPY}. "
        "Hãy chạy trước: uv run python src/scripts/extract_resnet.py"
    )

print(f">>> Nạp đặc trưng: {config.FEATURES_NPY.name} ...")

_raw_features = np.load(config.FEATURES_NPY, mmap_mode="r")
labels: np.ndarray = np.load(config.LABELS_NPY)

with open(config.IMAGE_LIST_TXT, encoding="utf-8") as f:
    image_b64_list: list[str] = [line.strip() for line in f]

# Chuẩn hoá L2 theo hàng -> inner product trong FAISS chính là cosine similarity
features: np.ndarray = (
    _raw_features / (np.linalg.norm(_raw_features, axis=1, keepdims=True) + 1e-12)
).astype("float32")

FEATURE_DIM = int(features.shape[1])
NUM_IMAGES = int(features.shape[0])

index_resnet = faiss.IndexFlatIP(FEATURE_DIM)   # exact search, không nén vector
index_resnet.add(features)
print(f">>> FAISS (ResNet) sẵn sàng: {NUM_IMAGES} ảnh, dim {FEATURE_DIM}")

# Gom chỉ số ảnh theo lớp (để lọc nhanh) + đếm số ảnh mỗi lớp (để tính Recall)
class_to_indices = {c: np.where(labels == c)[0] for c in range(config.NUM_CLASSES)}
class_counts = {c: int(len(idxs)) for c, idxs in class_to_indices.items()}


# ===========================================================
# 2. CLIP (tuỳ chọn) — chỉ bật khi có features_clip.npy
# ===========================================================
CLIP_ENABLED = config.CLIP_FEATURES_NPY.exists()
features_clip: Optional[np.ndarray] = None
index_clip = None

if CLIP_ENABLED:
    _clip_raw = np.load(config.CLIP_FEATURES_NPY, mmap_mode="r").astype("float32")
    # chuẩn hoá lại cho chắc (phòng khi file chưa được L2-normalize)
    features_clip = (
        _clip_raw / (np.linalg.norm(_clip_raw, axis=1, keepdims=True) + 1e-12)
    ).astype("float32")
    index_clip = faiss.IndexFlatIP(features_clip.shape[1])
    index_clip.add(features_clip)
    print(f">>> FAISS (CLIP) sẵn sàng: {features_clip.shape[0]} ảnh, dim {features_clip.shape[1]}")
else:
    print(">>> CLIP TẮT (chưa có features_clip.npy) — tìm bằng văn bản không khả dụng")

_clip_module = None


def get_clip():
    """Tải module CLIP một lần duy nhất, ngay lần truy vấn văn bản đầu tiên."""
    global _clip_module
    if _clip_module is None:
        from embedding import clip_model
        clip_model.load()
        _clip_module = clip_model
    return _clip_module


# ===========================================================
# 3. TRÍCH ĐẶC TRƯNG ẢNH UPLOAD (ResNet-18 pretrained ImageNet)
# ===========================================================
_device = torch.device("cpu")

_model = models.resnet18(weights="IMAGENET1K_V1")
_model = torch.nn.Sequential(*list(_model.children())[:-1])   # bỏ lớp phân loại cuối
_model.eval().to(_device)

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(">>> ResNet-18 sẵn sàng cho ảnh upload")


def extract_feature(img: Image.Image) -> np.ndarray:
    """Ảnh PIL -> vector 512 chiều đã L2-normalize (float32)."""
    tensor = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _model(tensor).cpu().numpy().flatten()
    return (feat / (np.linalg.norm(feat) + 1e-12)).astype("float32")


# ===========================================================
# 4. TÌM KIẾM
# ===========================================================
def run_search(
    query_feat: np.ndarray,
    k: int,
    class_filter: Optional[int] = None,
    exclude_idx: Optional[int] = None,
    index=None,
    feat_matrix: Optional[np.ndarray] = None,
):
    """
    Tìm top-k ảnh giống nhất.

    query_feat   : vector truy vấn ĐÃ L2-normalize, shape (dim,)
    class_filter : None = tìm toàn kho bằng FAISS; 0-9 = chỉ tìm trong lớp đó
    exclude_idx  : loại chính ảnh truy vấn khỏi kết quả (khi query là ảnh trong kho)
    index/feat_matrix : mặc định dùng không gian ResNet; truyền index_clip/features_clip
                        để tìm trong không gian CLIP.

    Trả về (indices, scores) — hai mảng numpy cùng độ dài <= k.
    """
    if index is None:
        index = index_resnet
    if feat_matrix is None:
        feat_matrix = features

    query_feat = np.asarray(query_feat, dtype="float32").reshape(-1)
    pad = 1 if exclude_idx is not None else 0   # lấy dư 1 để còn chỗ loại ảnh truy vấn

    if class_filter is None:
        # Toàn kho -> FAISS (exact, vài ms trên 60k vector)
        scores_mat, idx_mat = index.search(query_feat.reshape(1, -1), k + pad)
        idxs, scores = idx_mat[0], scores_mat[0]
    else:
        # Trong 1 lớp (~6000 ảnh) -> dot product numpy còn nhanh hơn dựng index riêng
        cand = class_to_indices[class_filter]
        cand = cand[cand < feat_matrix.shape[0]]   # phòng khi CLIP trích thiếu ảnh
        sims = feat_matrix[cand] @ query_feat
        order = np.argsort(-sims)[: k + pad]
        idxs, scores = cand[order], sims[order]

    if exclude_idx is not None:
        keep = idxs != exclude_idx
        idxs, scores = idxs[keep], scores[keep]

    return idxs[:k], scores[:k]


def build_results(idxs: Sequence[int], scores: Optional[Sequence[float]] = None) -> list[dict]:
    """Dựng danh sách kết quả trả cho client. scores=None -> không kèm % tương đồng."""
    results = []
    for rank, i in enumerate(idxs, start=1):
        i = int(i)
        item = {
            "rank": rank,
            "index": i,
            "label": config.class_name(labels[i]),
            "image": image_b64_list[i],
        }
        if scores is not None:
            score = float(scores[rank - 1])
            item["similarity"] = round(score * 100, 1)
            item["distance"] = round(1 - score, 4)
        results.append(item)
    return results


def compute_metrics(result_labels: Sequence[int], query_label: int, k: int) -> dict:
    """
    Precision@k, Recall@k và Average Precision — chỉ tính được khi truy vấn CÓ nhãn.

    Mẫu số của Precision giữ nguyên k (đúng như định nghĩa trong BAO_CAO_DANH_GIA.md).
    Recall chia cho tổng ảnh cùng lớp trong kho, trừ chính ảnh truy vấn.
    """
    rel = [1 if int(lab) == int(query_label) else 0 for lab in result_labels]
    num_hits = sum(rel)

    precision = num_hits / k if k else 0.0
    total_relevant = max(class_counts.get(int(query_label), 0) - 1, 1)
    recall = num_hits / total_relevant

    # AP: trung bình Precision@i tại mọi vị trí i trả về ảnh ĐÚNG lớp
    hits, sum_prec = 0, 0.0
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            sum_prec += hits / i
    ap = sum_prec / hits if hits else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "ap": round(ap, 4),
        "num_hits": num_hits,
        "k": k,
    }


# ===========================================================
# 5. CÁC THAO TÁC MỨC CAO (API gọi thẳng)
# ===========================================================
def search_by_feature(query_feat, k, class_filter=None, exclude_idx=None):
    """Tìm bằng vector ResNet -> (results, result_labels)."""
    idxs, scores = run_search(query_feat, k, class_filter, exclude_idx)
    return build_results(idxs, scores), [int(labels[int(i)]) for i in idxs]


def search_by_image(img: Image.Image, k, class_filter=None):
    """Tìm bằng ảnh PIL (upload) -> (results, result_labels)."""
    return search_by_feature(extract_feature(img), k, class_filter)


def search_by_index(idx: int, k, class_filter=None):
    """Tìm bằng ảnh có sẵn trong kho -> (results, result_labels). Tự loại chính nó."""
    return search_by_feature(np.asarray(features[idx]), k, class_filter, exclude_idx=idx)


def warmup_clip() -> None:
    """Nạp sẵn model CLIP.

    Gọi hàm này TRƯỚC khi bấm giờ: lần dùng CLIP đầu tiên phải tải/khởi tạo model
    (mất vài phút nếu chưa có weights trong cache). Nếu để lẫn vào đoạn đo thời gian
    thì search_time_ms của truy vấn đầu sẽ sai lệch khổng lồ.
    """
    if CLIP_ENABLED:
        get_clip()


def search_by_text(query_text: str, k, class_filter=None):
    """Tìm bằng mô tả văn bản qua CLIP -> (results, result_labels)."""
    if not CLIP_ENABLED:
        raise RuntimeError("CLIP chưa bật: thiếu features/features_clip.npy")
    qvec = get_clip().encode_text(query_text)
    idxs, scores = run_search(qvec, k, class_filter, index=index_clip, feat_matrix=features_clip)
    return build_results(idxs, scores), [int(labels[int(i)]) for i in idxs]


def sample_class(class_index: int, n: int) -> list[dict]:
    """Bốc ngẫu nhiên n ảnh thuộc một lớp (dùng khi người dùng chỉ nêu tên lớp)."""
    cand = class_to_indices[class_index]
    n = min(n, len(cand))
    chosen = random.sample(list(cand), n)
    return build_results(chosen)   # không có điểm tương đồng vì đây là duyệt, không phải tìm


def list_images(offset: int = 0, limit: int = config.GALLERY_DEFAULT_LIMIT) -> list[dict]:
    """Lấy một trang ảnh trong kho cho gallery."""
    stop = min(offset + limit, NUM_IMAGES)
    return [
        {"index": i, "label": config.class_name(labels[i]), "image": image_b64_list[i]}
        for i in range(max(offset, 0), stop)
    ]


def class_info() -> list[dict]:
    """Danh sách 10 lớp kèm tên tiếng Việt và số ảnh mỗi lớp."""
    return [
        {
            "index": i,
            "name": config.CIFAR10_CLASSES[i],
            "name_vi": config.CIFAR10_CLASSES_VN[i],
            "count": class_counts.get(i, 0),
        }
        for i in range(config.NUM_CLASSES)
    ]
