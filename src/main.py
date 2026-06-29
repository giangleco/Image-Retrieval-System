from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import base64
import io
import time
import random
from collections import Counter
from PIL import Image

import chat_agent  # parser câu lệnh rule-based (offline)

import torch
import torchvision.transforms as transforms
import torchvision.models as models

try:
    import faiss
except ImportError as e:
    raise ImportError("Không tìm thấy 'faiss'. Hãy cài bằng: pip install faiss-cpu") from e

# ===========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & CHẾ ĐỘ HOẠT ĐỘNG
# ===========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

feature_filename = "features.npy"
FEATURES_NPY = os.path.join(PROJECT_ROOT, "features", feature_filename)
IMAGE_LIST_TXT = os.path.join(PROJECT_ROOT, "features", "image_list.txt")
LABELS_NPY = os.path.join(PROJECT_ROOT, "features", "labels.npy")

# ===========================================================
# 2. KHỞI TẠO FLASK SERVER
# ===========================================================
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static")
)

# ===========================================================
# 3. TẢI DỮ LIỆU VÀO RAM VÀ XÂY DỰNG INDEX TÌM KIẾM
# ===========================================================
print(f"\n>>> Đang khởi động hệ thống. Tải dữ liệu: {feature_filename} ...")

if not os.path.exists(FEATURES_NPY) or not os.path.exists(LABELS_NPY):
    raise FileNotFoundError(f"Lỗi: Không tìm thấy {FEATURES_NPY}. Hãy chạy file feature_extractor.py trước!")

# Tải đặc trưng và nhãn
features = np.load(FEATURES_NPY, mmap_mode='r')
all_labels = np.load(LABELS_NPY)

with open(IMAGE_LIST_TXT, "r") as f:
    image_b64_list = [line.strip() for line in f]

# Chuẩn hóa L2 theo từng hàng (để inner product trong FAISS = cosine similarity)
features_normalized = (
    features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
).astype("float32")

# FAISS Index
dim = features.shape[1]
index_faiss = faiss.IndexFlatIP(dim)
index_faiss.add(features_normalized)

print(">>> FAISS Index đã sẵn sàng!")

# ===========================================================
# 3b. CIFAR-10: TÊN LỚP + CHỈ MỤC THEO LỚP (phục vụ lọc & metric)
# ===========================================================
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
# Gom chỉ số ảnh theo từng lớp (để lọc nhanh) + đếm số ảnh mỗi lớp (cho Recall)
class_to_indices = {c: np.where(all_labels == c)[0] for c in range(len(CIFAR10_CLASSES))}
class_counts = {c: int(len(idxs)) for c, idxs in class_to_indices.items()}

MAX_K = 50  # số kết quả tối đa cho mỗi truy vấn

# ===========================================================
# 3c. CLIP: TÌM ẢNH BẰNG VĂN BẢN / ẢNH (tùy chọn — cần features_clip.npy)
# ===========================================================
CLIP_FEATURES_NPY = os.path.join(PROJECT_ROOT, "features", "features_clip.npy")
CLIP_ENABLED = os.path.exists(CLIP_FEATURES_NPY)
features_clip = None
index_clip = None

if CLIP_ENABLED:
    features_clip = np.load(CLIP_FEATURES_NPY, mmap_mode="r").astype("float32")
    # đảm bảo đã L2-normalize (an toàn nếu file chưa chuẩn hoá)
    norms = np.linalg.norm(features_clip, axis=1, keepdims=True)
    features_clip = (features_clip / (norms + 1e-12)).astype("float32")
    index_clip = faiss.IndexFlatIP(features_clip.shape[1])
    index_clip.add(features_clip)
    print(f">>> CLIP index sẵn sàng! ({features_clip.shape[0]} ảnh, dim {features_clip.shape[1]})")
else:
    print(">>> (CLIP tắt: chưa có features_clip.npy — chạy clip_extractor.py để bật tìm bằng văn bản)")

# Model CLIP được tải LAZY (chỉ khi có truy vấn đầu tiên) để khởi động nhanh
_clip = None

def get_clip():
    """Tải module clip_model 1 lần (lazy)."""
    global _clip
    if _clip is None:
        import clip_model
        clip_model.load()
        _clip = clip_model
    return _clip

# ===========================================================
# 4. MÔ HÌNH TRÍCH XUẤT ĐẶC TRƯNG ẢNH UPLOAD
# ===========================================================
device = torch.device("cpu")

model = models.resnet18(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])
print(">>> Web đang dùng mô hình PRE-TRAINED để xử lý ảnh upload mới.")
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_feature(img_pil: Image.Image) -> np.ndarray:
    """Biến ảnh PIL upload thành vector 512 chiều (đã L2-normalize)"""
    tensor = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().flatten()

    norm_val = np.linalg.norm(feat) + 1e-8
    feat = feat / norm_val
    return feat.astype("float32")

# ===========================================================
# 4b. HÀM TÌM KIẾM & TÍNH METRIC
# ===========================================================
def _label_name(c):
    """int label -> tên lớp (an toàn nếu ngoài phạm vi)."""
    c = int(c)
    return CIFAR10_CLASSES[c] if 0 <= c < len(CIFAR10_CLASSES) else str(c)

def run_search(query_feat, k, class_filter=None, exclude_idx=None,
               index=None, feat_matrix=None):
    """
    Tìm top-k ảnh giống nhất trên một index FAISS + ma trận đặc trưng.
      query_feat  : vector truy vấn đã L2-normalize, shape (dim,)
      k           : số kết quả mong muốn
      class_filter: None = tìm toàn bộ kho (FAISS); hoặc index lớp 0-9 = chỉ tìm trong lớp đó
      exclude_idx : bỏ qua ảnh này (chính ảnh truy vấn khi chọn từ kho)
      index/feat_matrix: mặc định dùng ResNet (index_faiss/features_normalized);
                         truyền vào index_clip/features_clip để tìm trên không gian CLIP.
    Trả về: (indices, scores) — 2 numpy array cùng độ dài <= k
    """
    if index is None:
        index = index_faiss
    if feat_matrix is None:
        feat_matrix = features_normalized

    query_feat = np.asarray(query_feat, dtype="float32").reshape(-1)
    pad = 1 if exclude_idx is not None else 0

    if class_filter is None:
        # Tìm toàn bộ kho bằng FAISS (exact, nhanh). Lấy dư 1 để có chỗ loại self.
        D, I = index.search(query_feat.reshape(1, -1), k + pad)
        idxs, scores = I[0], D[0]
    else:
        # Chỉ tìm trong ảnh thuộc lớp được chọn -> dot product bằng numpy (kho nhỏ ~6000 ảnh)
        cand = class_to_indices[class_filter]
        # chỉ giữ các chỉ số nằm trong phạm vi feat_matrix (CLIP có thể trích thiếu)
        cand = cand[cand < feat_matrix.shape[0]]
        sims = feat_matrix[cand] @ query_feat
        order = np.argsort(-sims)[: k + pad]
        idxs, scores = cand[order], sims[order]

    # Loại bỏ chính ảnh truy vấn nếu có
    if exclude_idx is not None:
        keep = idxs != exclude_idx
        idxs, scores = idxs[keep], scores[keep]

    return idxs[:k], scores[:k]

def compute_metrics(result_labels, query_label, k):
    """Tính Precision@k, Recall@k, AP dựa trên nhãn (chỉ khi truy vấn có nhãn)."""
    rel = [1 if int(lab) == int(query_label) else 0 for lab in result_labels]
    num_hits = sum(rel)

    precision = num_hits / k if k else 0.0
    # Tổng số ảnh cùng lớp trong kho, trừ chính nó
    total_relevant = max(class_counts.get(int(query_label), 0) - 1, 1)
    recall = num_hits / total_relevant

    # Average Precision: trung bình precision@i tại các vị trí đúng lớp
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

def build_results(idxs, scores, with_similarity=True):
    """Dựng list kết quả JSON (ảnh base64 + nhãn + % tương đồng) từ chỉ số & điểm."""
    results, result_labels = [], []
    for rank, (i, score) in enumerate(zip(idxs, scores), 1):
        i = int(i)
        lab = int(all_labels[i])
        result_labels.append(lab)
        item = {
            "rank": rank,
            "image": image_b64_list[i],
            "label": _label_name(lab),
        }
        if with_similarity:
            item["distance"] = round(1 - float(score), 4)
            item["similarity"] = round(float(score) * 100, 1)
        results.append(item)
    return results, result_labels

def clip_text_search(query_text, k, class_filter=None):
    """Mã hoá câu mô tả bằng CLIP rồi tìm trên index CLIP. Trả về (results, labels)."""
    qvec = get_clip().encode_text(query_text)            # (512,) đã L2-normalize
    idxs, scores = run_search(qvec, k, class_filter,
                              index=index_clip, feat_matrix=features_clip)
    return build_results(idxs, scores)

# ===========================================================
# 5. ROUTER (API GIAO DIỆN WEB)
# ===========================================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        images=image_b64_list[:300],
        classes=CIFAR10_CLASSES,
        clip_enabled=CLIP_ENABLED,
    )

@app.route("/text_search", methods=["GET", "POST"])
def text_search():
    """Tìm ảnh bằng MÔ TẢ văn bản (CLIP). Vd: q='a red truck', 'con mèo'."""
    if not CLIP_ENABLED:
        return jsonify({"error": "Chưa bật CLIP. Hãy tạo features_clip.npy "
                                 "(chạy clip_extractor.py hoặc trích trên Kaggle)."}), 400

    src = request.form if request.method == "POST" else request.args
    query_text = (src.get("q") or "").strip()
    if not query_text:
        return jsonify({"error": "Vui lòng nhập mô tả cần tìm!"}), 400

    try:
        k = int(src.get("k", 10))
    except (TypeError, ValueError):
        k = 10
    k = max(1, min(k, MAX_K))

    class_filter = None
    cf_raw = src.get("class_filter", "all")
    if cf_raw not in (None, "", "all"):
        try:
            cf = int(cf_raw)
            if 0 <= cf < len(CIFAR10_CLASSES):
                class_filter = cf
        except (TypeError, ValueError):
            class_filter = None

    start = time.time()
    results, _ = clip_text_search(query_text, k, class_filter)
    elapsed = time.time() - start

    print(f"\n{'='*70}\n   [TEXT-SEARCH] q={query_text!r} (k={k}) | {elapsed*1000:.1f} ms\n{'='*70}\n")
    return jsonify({
        "query_text": query_text,
        "k": k,
        "class_filter": CIFAR10_CLASSES[class_filter] if class_filter is not None else None,
        "search_time_ms": round(elapsed * 1000, 3),
        "results": results,
    })

@app.route("/search", methods=["GET", "POST"])
def search():
    # --- Đọc tham số chung: k và bộ lọc lớp (từ form nếu POST, query-string nếu GET) ---
    src = request.form if request.method == "POST" else request.args

    try:
        k = int(src.get("k", 10))
    except (TypeError, ValueError):
        k = 10
    k = max(1, min(k, MAX_K))

    class_filter = None
    cf_raw = src.get("class_filter", "all")
    if cf_raw not in (None, "", "all"):
        try:
            cf = int(cf_raw)
            if 0 <= cf < len(CIFAR10_CLASSES):
                class_filter = cf
        except (TypeError, ValueError):
            class_filter = None

    exclude_idx = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "Vui lòng chọn ảnh!"}), 400

        img = Image.open(file.stream).convert("RGB")
        query_feat = extract_feature(img)
        query_label = None

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        query_b64 = base64.b64encode(buf.getvalue()).decode()
    else:
        idx = int(request.args.get("idx", 0))
        query_b64 = image_b64_list[idx]
        query_feat = np.asarray(features_normalized[idx])
        query_label = int(all_labels[idx])
        exclude_idx = idx  # bỏ chính ảnh truy vấn khỏi kết quả

    # --- Tìm kiếm ---
    start = time.time()
    idxs, scores = run_search(query_feat, k, class_filter, exclude_idx)
    time_search = time.time() - start

    results = []
    result_labels = []
    for rank, (i, score) in enumerate(zip(idxs, scores), 1):
        i = int(i)
        lab = int(all_labels[i])
        result_labels.append(lab)
        results.append({
            "rank": rank,
            "image": image_b64_list[i],
            "distance": round(1 - float(score), 4),
            "similarity": round(float(score) * 100, 1),  # % độ tương đồng
            "label": _label_name(lab),
        })

    # --- Metric: chỉ tính khi truy vấn có nhãn VÀ không lọc theo lớp ---
    metrics = None
    if query_label is not None and class_filter is None and results:
        metrics = compute_metrics(result_labels, query_label, k)

    # In báo cáo ra Terminal
    filter_name = CIFAR10_CLASSES[class_filter] if class_filter is not None else "tất cả"
    print(f"\n{'='*70}")
    print(f"   BÁO CÁO TÌM KIẾM  (k={k}, lớp lọc={filter_name})")
    print(f"   → Thời gian tìm kiếm : {time_search*1000:.4f} ms")
    if metrics:
        print(f"   → Precision@{k}: {metrics['precision']} | "
              f"Recall@{k}: {metrics['recall']} | AP: {metrics['ap']}")
    print(f"{'='*70}\n")

    return jsonify({
        "query_image": query_b64,
        "query_label": _label_name(query_label) if query_label is not None else None,
        "k": k,
        "class_filter": CIFAR10_CLASSES[class_filter] if class_filter is not None else None,
        "search_time_ms": round(time_search * 1000, 3),
        "metrics": metrics,
        "results": results,
    })

# ===========================================================
# 6. AGENT CHAT (rule-based, offline)
# ===========================================================
def _build_reply(plan, result_labels):
    """Soạn câu trả lời thân thiện cho người dùng dựa trên kết quả tìm được."""
    n = len(result_labels)
    if plan["class_filter"] is not None:
        # Đã lọc theo lớp -> không đoán lớp nữa (sẽ lòng vòng)
        cls = CIFAR10_CLASSES[plan["class_filter"]]
        return f"Đây là {n} ảnh thuộc lớp \"{cls}\" giống ảnh của bạn nhất 👇"

    reply = f"Đây là {n} ảnh giống ảnh của bạn nhất 👇"
    # Đoán nội dung: lớp xuất hiện nhiều nhất trong kết quả
    if result_labels:
        common, cnt = Counter(result_labels).most_common(1)[0]
        reply += (f" Theo mô hình, ảnh của bạn trông giống lớp "
                  f"\"{_label_name(common)}\" nhất ({cnt}/{n} kết quả).")
    return reply


@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "")
    file = request.files.get("file")
    has_image = bool(file and file.filename)

    plan = chat_agent.parse_message(message, has_image, max_k=MAX_K)

    # --- Chỉ có CHỮ (không ảnh) + CLIP bật: tìm ảnh theo MÔ TẢ (semantic) ---
    if (not has_image and CLIP_ENABLED
            and plan["intent"] in ("browse_class", "need_image") and message.strip()):
        k = plan["k"]
        class_filter = plan["class_filter"]
        start = time.time()
        results, _ = clip_text_search(message, k, class_filter)
        elapsed = time.time() - start
        reply = f"Mình tìm theo mô tả \"{message.strip()}\" — đây là {len(results)} ảnh khớp nhất 👇"
        print(f"\n{'='*70}\n   [CHAT-CLIP] q={message!r} (k={k}) | {elapsed*1000:.1f} ms\n{'='*70}\n")
        return jsonify({
            "reply": reply,
            "query_image": None,
            "k": k,
            "class_filter": CIFAR10_CLASSES[class_filter] if class_filter is not None else None,
            "search_time_ms": round(elapsed * 1000, 3),
            "results": results,
        })

    # --- Chỉ có chữ + nhắc tên lớp: duyệt thẳng ảnh của lớp đó trong kho (không cần ảnh) ---
    if plan["intent"] == "browse_class":
        cls = plan["class_filter"]
        cand = class_to_indices[cls]
        n = min(plan["k"], len(cand))
        chosen = random.sample(list(cand), n)   # chọn ngẫu nhiên cho đa dạng
        results = [{
            "rank": rank,
            "image": image_b64_list[int(i)],
            "label": _label_name(int(all_labels[int(i)])),
        } for rank, i in enumerate(chosen, 1)]
        reply = (f"Bạn chưa gửi ảnh, nhưng mình hiểu bạn muốn xem lớp "
                 f"\"{CIFAR10_CLASSES[cls]}\". Đây là {n} ảnh thuộc lớp này trong kho 👇")
        print(f"\n{'='*70}\n   [CHAT] browse lớp '{CIFAR10_CLASSES[cls]}' (n={n})\n{'='*70}\n")
        return jsonify({
            "reply": reply,
            "query_image": None,
            "k": n,
            "class_filter": CIFAR10_CLASSES[cls],
            "results": results,
        })

    # --- Không có ảnh & không rõ lớp: chào hỏi hoặc nhắc người dùng tải ảnh ---
    if plan["intent"] != "search":
        if plan["intent"] == "greeting":
            reply = ("Xin chào! Mình là trợ lý tìm ảnh 🤖. Hãy tải lên một ảnh và "
                     "nói ví dụ: \"Tìm cho tôi 10 ảnh giống ảnh này\". "
                     "Hoặc chỉ cần gõ tên một lớp, ví dụ \"cho tôi xem 10 ảnh con mèo\", "
                     "mình sẽ lấy ảnh lớp đó trong kho.")
        else:
            reply = ("Bạn hãy đính kèm một ảnh để mình tìm ảnh giống, hoặc gõ tên một "
                     "lớp (chó, mèo, máy bay, ô tô…) để mình lấy ảnh lớp đó nhé. 📎")
        return jsonify({"reply": reply, "results": [], "query_image": None})

    # --- Có ảnh: trích đặc trưng + tìm kiếm theo K và lớp đã hiểu được ---
    img = Image.open(file.stream).convert("RGB")
    query_feat = extract_feature(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    query_b64 = base64.b64encode(buf.getvalue()).decode()

    k = plan["k"]
    class_filter = plan["class_filter"]

    start = time.time()
    idxs, scores = run_search(query_feat, k, class_filter)
    time_search = time.time() - start

    results, result_labels = [], []
    for rank, (i, score) in enumerate(zip(idxs, scores), 1):
        i = int(i)
        lab = int(all_labels[i])
        result_labels.append(lab)
        results.append({
            "rank": rank,
            "image": image_b64_list[i],
            "distance": round(1 - float(score), 4),
            "similarity": round(float(score) * 100, 1),
            "label": _label_name(lab),
        })

    reply = _build_reply(plan, result_labels)

    filter_name = CIFAR10_CLASSES[class_filter] if class_filter is not None else "tất cả"
    print(f"\n{'='*70}")
    print(f"   [CHAT] message={message!r}")
    print(f"   → hiểu: k={k}, lớp lọc={filter_name} | thời gian {time_search*1000:.2f} ms")
    print(f"{'='*70}\n")

    return jsonify({
        "reply": reply,
        "query_image": query_b64,
        "k": k,
        "class_filter": CIFAR10_CLASSES[class_filter] if class_filter is not None else None,
        "search_time_ms": round(time_search * 1000, 3),
        "results": results,
    })


if __name__ == "__main__":
    # host 0.0.0.0 để truy cập được từ ngoài container Docker; PORT đổi qua biến môi trường
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"

    print("\n" + "="*70)
    print("   HỆ THỐNG TRUY XUẤT ẢNH CIFAR-10")
    print(f"   Dữ liệu đang dùng: {feature_filename}")
    print(f"   Server đang chạy tại: http://localhost:{port}")
    print("="*70 + "\n")
    app.run(host=host, port=port, debug=debug)
