from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import base64
import io
import time  
from PIL import Image

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

# Chuẩn hóa L2 theo từng hàng (tương đương sklearn.preprocessing.normalize)
features_normalized = (
    features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
).astype("float32")

# FAISS Index
dim = features.shape[1] 
index_faiss = faiss.IndexFlatIP(dim) 
index_faiss.add(features_normalized) 

print(">>> FAISS Index đã sẵn sàng!")

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
    """Biến ảnh PIL upload thành vector 512 chiều"""
    tensor = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().flatten()
    
    norm_val = np.linalg.norm(feat) + 1e-8
    feat = feat / norm_val
    return feat.astype("float32")

# ===========================================================
# 5. ROUTER (API GIAO DIỆN WEB)
# ===========================================================
@app.route("/")
def home():
    return render_template("index.html", images=image_b64_list[:300])

@app.route("/search", methods=["GET", "POST"])
def search():
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
        query_feat = features_normalized[idx] 
        query_label = all_labels[idx]         

    query_feat = query_feat.reshape(1, -1)

    start = time.time()
    D, I = index_faiss.search(query_feat, 10)
    time_faiss = time.time() - start 

    # In báo cáo ra Terminal
    print(f"\n{'='*70}")
    print(f"   BÁO CÁO TÌM KIẾM")
    print(f"   → Thời gian FAISS : {time_faiss*1000:.4f} ms")
    print(f"{'='*70}\n")

    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
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

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   HỆ THỐNG TRUY XUẤT ẢNH CIFAR-10")
    print(f"   Dữ liệu đang dùng: {feature_filename}")
    print("   Server đang chạy tại: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=True, port=5000)