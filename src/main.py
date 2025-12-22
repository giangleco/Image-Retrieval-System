from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import base64
import io
import time  # Dùng để bấm giờ so sánh tốc độ
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import faiss # Thư viện tìm kiếm siêu tốc của Facebook

# ===========================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN (PATH CONFIGURATION)
# ===========================================================
# Mục đích: Giúp code chạy đúng trên mọi máy tính (Win/Mac/Linux)
# mà không bị lỗi "File not found".
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # Lấy đường dẫn folder src/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))   # Lấy đường dẫn folder dự án gốc

FEATURES_NPY = os.path.join(PROJECT_ROOT, "data", "features", "features.npy")
IMAGE_LIST_TXT = os.path.join(PROJECT_ROOT, "data", "features", "image_list.txt")

# ===========================================================
# 2. KHỞI TẠO FLASK SERVER
# ===========================================================
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"), # Nơi chứa file HTML
    static_folder=os.path.join(PROJECT_ROOT, "static")       # Nơi chứa CSS/JS
)

# ===========================================================
# 3. TẢI DỮ LIỆU & XÂY DỰNG INDEX (KNN + FAISS)
# ===========================================================
print(">>> Đang tải dữ liệu CIFAR-10 vào RAM...")

# Kiểm tra xem đã chạy feature_extractor.py chưa
if not os.path.exists(FEATURES_NPY):
    raise FileNotFoundError(f"Lỗi: Không tìm thấy {FEATURES_NPY}. Hãy chạy feature_extractor.py trước!")

# Load vector đặc trưng (60000, 512)
features = np.load(FEATURES_NPY).astype("float32")

# Load danh sách ảnh Base64 để hiển thị
with open(IMAGE_LIST_TXT, "r") as f:
    image_b64_list = [line.strip() for line in f]

print(f"OK! Đã tải {len(image_b64_list)} ảnh. Shape: {features.shape}")

# [QUAN TRỌNG] Chuẩn hóa L2 (L2 Normalization)
# Tại sao? Vì FAISS IndexFlatIP tính tích vô hướng (Dot Product).
# Nếu vector có độ dài = 1, thì Tích vô hướng == Cosine Similarity.
# Đây là mẹo để dùng FAISS tìm kiếm theo độ đo Cosine.
features_normalized = normalize(features, axis=1, norm='l2')

# --- CẤU HÌNH KNN (SKLEARN) ---
# Dùng thuật toán Brute-force (vét cạn) để làm chuẩn so sánh
print(">>> Đang xây dựng KNN Index...")
knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
knn.fit(features_normalized) 
print("KNN đã sẵn sàng!")

# --- CẤU HÌNH FAISS (FACEBOOK AI) ---
print(">>> Đang xây dựng FAISS Index...")
dim = features.shape[1]  # 512 chiều
index_faiss = faiss.IndexFlatIP(dim) # IP = Inner Product (Tích vô hướng)
index_faiss.add(features_normalized) # Nạp dữ liệu vào index
print("FAISS đã sẵn sàng!")

# ===========================================================
# 4. TẢI NHÃN (LABEL) ĐỂ TỰ CHẤM ĐIỂM
# ===========================================================
# Mục đích: Để biết ảnh tìm được có đúng là "con mèo" giống ảnh gốc không.
print(">>> Đang tải Labels để tính Recall...")
# Tải lại bộ dữ liệu gốc chỉ để lấy cái nhãn (targets)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)

# Gộp nhãn train và test lại theo đúng thứ tự lúc trích xuất đặc trưng
train_labels = np.array(trainset.targets)
test_labels = np.array(testset.targets)
all_labels = np.concatenate([train_labels, test_labels]) 
print("Labels đã tải xong!")

# ===========================================================
# 5. MÔ HÌNH TRÍCH ĐẶC TRƯNG (RESNET18)
# ===========================================================
# Phần này dùng để xử lý ảnh người dùng upload lên
device = torch.device("cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1]) # Bỏ lớp FC cuối
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Chuẩn hóa theo thông số ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_feature(img_pil: Image.Image) -> np.ndarray:
    """Hàm helper: Biến ảnh PIL thành vector 512 chiều"""
    tensor = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().flatten()
    
    # [QUAN TRỌNG] Chuẩn hóa vector Query
    # Vector trong kho đã chuẩn hóa thì vector tìm kiếm cũng phải chuẩn hóa
    # Cộng thêm 1e-8 để tránh lỗi chia cho 0
    norm_val = np.linalg.norm(feat) + 1e-8
    feat = feat / norm_val
    
    return feat.astype("float32")

# ===========================================================
# 6. HÀM TÍNH ĐỘ CHÍNH XÁC (METRIC)
# ===========================================================
def calculate_recall_at_10(indices, query_label):
    """
    Tính Recall@10:
    - indices: Danh sách vị trí 10 ảnh tìm được
    - query_label: Nhãn của ảnh gốc (ví dụ: số 3 - Mèo)
    """
    # Lấy nhãn của 10 ảnh kết quả
    retrieved_labels = all_labels[indices]
    
    # Đếm xem có bao nhiêu ảnh đúng (cùng nhãn với query)
    relevant_count = np.sum(retrieved_labels == query_label)
    
    # Tổng số ảnh cùng loại trong toàn bộ dữ liệu (CIFAR-10 mỗi lớp có 6000 ảnh)
    total_relevant = np.sum(all_labels == query_label) - 1 # Trừ đi chính nó
    
    # Công thức Recall = Số ảnh đúng tìm được / Tổng số ảnh đúng có trong kho
    return relevant_count / total_relevant if total_relevant > 0 else 0.0

# ===========================================================
# 7. XỬ LÝ ROUTES (GIAO DIỆN & API)
# ===========================================================
@app.route("/")
def home():
    # Hiển thị 300 ảnh đầu tiên ra màn hình chính
    return render_template("index.html", images=image_b64_list[:300])

@app.route("/search", methods=["GET", "POST"])
def search():
    # --- BƯỚC 1: LẤY ẢNH QUERY & TRÍCH XUẤT ĐẶC TRƯNG ---
    if request.method == "POST":
        # Trường hợp 1: Người dùng Upload ảnh
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "Vui lòng chọn ảnh!"}), 400
        
        img = Image.open(file.stream).convert("RGB")
        query_feat = extract_feature(img)
        query_label = None  # Ảnh ngoài nên không biết nhãn gì, không tính điểm được
        
        # Chuyển ảnh upload thành base64 để hiện lại trên web
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        query_b64 = base64.b64encode(buf.getvalue()).decode()

    else:
        # Trường hợp 2: Người dùng click vào ảnh có sẵn
        idx = int(request.args.get("idx", 0))
        query_b64 = image_b64_list[idx]
        query_feat = features_normalized[idx] # Lấy vector đã có sẵn
        query_label = all_labels[idx]         # Lấy nhãn đã biết để tính điểm

    # Reshape thành (1, 512) để thư viện hiểu
    query_feat = query_feat.reshape(1, -1)

    # --- BƯỚC 2: SO SÁNH HIỆU NĂNG (BENCHMARKING) ---
    
    # 1. Đo tốc độ KNN
    start = time.time()
    _, indices_knn = knn.kneighbors(query_feat, n_neighbors=10)
    time_knn = time.time() - start # Tính thời gian chạy

    # 2. Đo tốc độ FAISS
    start = time.time()
    # FAISS trả về D (Distances/Scores) và I (Indices)
    D, I = index_faiss.search(query_feat, 10)
    time_faiss = time.time() - start # Tính thời gian chạy

    # 3. Tính độ chính xác (Nếu có nhãn)
    recall_knn = calculate_recall_at_10(indices_knn[0], query_label) if query_label is not None else None
    recall_faiss = calculate_recall_at_10(I[0], query_label) if query_label is not None else None

    # --- BƯỚC 3: IN BÁO CÁO RA TERMINAL (CONSOLE LOG) ---
    print(f"\n{'='*70}")
    print(f"   BÁO CÁO HIỆU SUẤT TÌM KIẾM (Trên 60.000 ảnh)")
    # Nhân 1000 để đổi từ giây (s) sang mili-giây (ms) cho dễ nhìn
    print(f"   → Thời gian KNN   : {time_knn*1000:.4f} ms")
    print(f"   → Thời gian FAISS : {time_faiss*1000:.4f} ms")
    print(f"   => FAISS nhanh gấp: {time_knn/time_faiss:.1f} lần")
    
    if recall_knn is not None:
        print(f"   ------------------------------------------")
        # Lưu ý: Recall này sẽ rất nhỏ (vì chia cho 6000), chủ yếu để so sánh 2 thuật toán
        print(f"   → Recall KNN      : {recall_knn:.6f}")
        print(f"   → Recall FAISS    : {recall_faiss:.6f}")
        if recall_faiss == recall_knn:
            print(f"   => Độ chính xác: TƯƠNG ĐƯƠNG")
    print(f"{'='*70}\n")

    # --- BƯỚC 4: TRẢ KẾT QUẢ VỀ WEB ---
    # Ta chọn kết quả của FAISS để hiển thị vì nó nhanh hơn
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        # Score của FAISS là Cosine Similarity (càng lớn càng giống, max là 1)
        # Web cần hiển thị Distance (càng nhỏ càng giống). Nên lấy 1 - Score.
        distance = round(1 - float(score), 4)
        
        results.append({
            "rank": rank,
            "image": image_b64_list[idx],
            "distance": distance
        })

    return jsonify({
        "query_image": query_b64,
        "results": results
    })

# ===========================================================
# 8. CHẠY CHƯƠNG TRÌNH
# ===========================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("   CIFAR-10 IMAGE SEARCH ENGINE")
    print("   Công nghệ: ResNet-18 + FAISS + Benchmark System")
    print("   Server đang chạy tại: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=True, port=5000)