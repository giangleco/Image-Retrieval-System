from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import base64
import io
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.neighbors import NearestNeighbors

# ===========================================================
# 1. XÁC ĐỊNH ĐƯỜNG DẪN GỐC DỰ ÁN (QUAN TRỌNG NHẤT!)
# ===========================================================
# Vì main.py nằm trong thư mục src/ → phải đi lên 1 cấp để lấy đúng thư mục gốc
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))        # → .../src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) # → .../CIFAR10_IMAGE_SEARCH (thư mục gốc)

# Đường dẫn chính xác đến file đặc trưng và danh sách ảnh đã được trích xuất trước đó
FEATURES_NPY = os.path.join(PROJECT_ROOT, "data", "features", "features.npy")      # Vector đặc trưng (60000 x 512)
IMAGE_LIST_TXT = os.path.join(PROJECT_ROOT, "data", "features", "image_list.txt") # 60000 ảnh dạng base64

# ===========================================================
# 2. KHỞI TẠO ỨNG DỤNG FLASK
# ===========================================================
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),  # Thư mục chứa index.html
    static_folder=os.path.join(PROJECT_ROOT, "static")        # Thư mục chứa CSS, JS, hình ảnh
)

# ===========================================================
# 3. TẢI DỮ LIỆU ĐÃ TRÍCH XUẤT SẴN + XÂY DỰNG CHỈ MỤC KNN
# ===========================================================
print("Đang tải dữ liệu CIFAR-10 từ file đã trích xuất trước...")

# Kiểm tra file tồn tại → nếu không có thì báo lỗi rõ ràng (dễ debug)
if not os.path.exists(FEATURES_NPY):
    raise FileNotFoundError(f"Không tìm thấy file đặc trưng!\n→ {FEATURES_NPY}\nChạy lại feature_extractor.py trước!")
if not os.path.exists(IMAGE_LIST_TXT):
    raise FileNotFoundError(f"Không tìm thấy file danh sách ảnh!\n→ {IMAGE_LIST_TXT}")

# Load vector đặc trưng (60000 ảnh x 512 chiều)
features = np.load(FEATURES_NPY).astype("float32")

# Load danh sách ảnh đã mã hóa base64 (mỗi dòng 1 ảnh → hiển thị trực tiếp trên web)
with open(IMAGE_LIST_TXT, "r") as f:
    image_b64_list = [line.strip() for line in f]

print(f"Đã tải thành công {len(image_b64_list)} ảnh CIFAR-10!")
print(f"→ Kích thước vector đặc trưng: {features.shape}")  # Phải là (60000, 512)

# Xây dựng chỉ mục KNN để tìm kiếm nhanh 10 ảnh gần nhất
# Dùng metric="cosine" → tốt hơn Euclidean khi so sánh đặc trưng ảnh (đã chuẩn hóa)
print("Đang xây dựng chỉ mục KNN để tìm kiếm siêu nhanh...")
knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
knn.fit(features)  # Fit toàn bộ 60.000 vector vào bộ tìm kiếm
print("KNN đã sẵn sàng! Có thể tìm ảnh giống trong tích tắc!")

# ===========================================================
# 4. TẢI MÔ HÌNH RESNET-18 ĐỂ TRÍCH XUẤT ĐẶC TRƯNG TỪ ẢNH NGƯỜI DÙNG UPLOAD
# ===========================================================
device = torch.device("cpu")  # Dùng CPU cho ổn định (không cần GPU)
print(f"Đang tải mô hình ResNet-18 (pretrained trên ImageNet)...")

model = models.resnet18(weights="IMAGENET1K_V1")                    # Tải trọng số chính thức
model = torch.nn.Sequential(*list(model.children())[:-1])           # Bỏ lớp Fully Connected → chỉ lấy đặc trưng
model.eval()                                                        # Chuyển sang chế độ suy luận
model.to(device)

# Tiền xử lý ảnh giống hệt lúc trích xuất dữ liệu CIFAR-10
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),                                   # ResNet yêu cầu kích thước 224x224
    transforms.ToTensor(),                                           # Chuyển sang tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                 # Chuẩn hóa theo ImageNet
                         std=[0.229, 0.224, 0.225])
])

def extract_feature(img_pil: Image.Image) -> np.ndarray:
    """
    Trích xuất vector đặc trưng 512 chiều từ ảnh người dùng upload
    Input: Ảnh PIL (RGB)
    Output: Vector numpy 512 chiều (float32)
    """
    tensor = preprocess(img_pil).unsqueeze(0).to(device)  # Thêm batch dimension
    with torch.no_grad():                                 # Tắt gradient → nhanh + tiết kiệm RAM
        feat = model(tensor).cpu().numpy().flatten().astype("float32")
    return feat

# ===========================================================
# 5. CÁC ROUTE (ĐƯỜNG DẪN) CỦA WEB
# ===========================================================

@app.route("/")
def home():
    """Trang chủ: Hiển thị giao diện + 300 ảnh mẫu để người dùng click thử"""
    return render_template("index.html", images=image_b64_list[:300])

@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Xử lý tìm kiếm ảnh tương tự
    - POST: Người dùng upload ảnh
    - GET: Người dùng click vào ảnh mẫu (dùng tham số idx)
    """
    if request.method == "POST":
        # === Trường hợp người dùng upload ảnh ===
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "Không có file được upload!"}), 400

        # Đọc ảnh từ stream và chuyển sang định dạng RGB
        img = Image.open(file.stream).convert("RGB")
        query_feat = extract_feature(img)  # Trích xuất đặc trưng từ ảnh upload

        # Chuyển ảnh upload thành base64 để hiển thị trên web
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        query_b64 = base64.b64encode(buf.getvalue()).decode()

    else:
        # === Trường hợp click ảnh mẫu từ gallery ===
        idx = int(request.args.get("idx", 0))           # Lấy chỉ số ảnh từ URL (?idx=123)
        query_b64 = image_b64_list[idx]                 # Lấy ảnh base64 từ danh sách
        query_feat = features[idx]                      # Lấy vector đặc trưng đã lưu sẵn

    # === TÌM 10 ẢNH GẦN NHẤT BẰNG KNN ===
    distances, indices = knn.kneighbors(query_feat.reshape(1, -1), n_neighbors=10)
    # distances: khoảng cách cosine (càng nhỏ càng giống)
    # indices: chỉ số của 10 ảnh gần nhất trong database

    # Chuẩn bị kết quả trả về cho frontend
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        results.append({
            "rank": rank,
            "image": image_b64_list[idx],           # Ảnh base64 để hiển thị
            "distance": round(float(dist), 4)       # Khoảng cách (càng nhỏ càng giống)
        })

    # Trả về JSON để JavaScript xử lý và hiển thị
    return jsonify({
        "query_image": query_b64,
        "results": results
    })

# ===========================================================
# 6. KHỞI ĐỘNG SERVER
# ===========================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("   CIFAR-10 IMAGE SEARCH ENGINE ĐÃ KHỞI ĐỘNG THÀNH CÔNG!")
    print(f"   Thư mục dự án    : {PROJECT_ROOT}")
    print(f"   File đặc trưng   : {FEATURES_NPY}")
    print(f"   File ảnh base64  : {IMAGE_LIST_TXT}")
    print(f"   Web đang chạy tại: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=True, port=5000)