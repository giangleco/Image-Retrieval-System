
# 🔍 Hệ thống Truy Xuất Hình Ảnh Tương Tự (Image Retrieval System)

Dự án này xây dựng hệ thống **truy xuất hình ảnh tương tự** dựa trên đặc trưng sâu (deep features) được trích xuất bằng **ResNet-18** từ bộ dữ liệu **CIFAR-10**.  
Hệ thống bao gồm phần backend (Python), frontend (HTML/CSS/JS) và dữ liệu đặc trưng để tìm kiếm ảnh giống nhau.

---

## 📂 Cấu trúc thư mục dự án

```
BTL/
│── data/
│   ├── cifar-10-batches-py/        # Bộ dữ liệu CIFAR-10 gốc
│   └── features/
│       ├── features.npy            # Ma trận đặc trưng (60000 × 512)
│       └── image_list.txt          # Ảnh dạng Base64 để hiển thị giao diện
│
│── model/                          # (Tuỳ chọn) chứa các mô hình DL mở rộng
│
│── src/
│   ├── feature_extractor.py        # File dùng để trích xuất đặc trưng
│   └── main.py                     # API backend hoặc script xử lý chính
│
│── static/
│   ├── css/
│   │   └── style.css               # File CSS cho giao diện
│   └── js/
│       └── app.js                  # Xử lý logic giao diện Web
│
│── templates/
│   └── index.html                  # Giao diện chính của website
│
│── .gitignore                      # File cấu hình Git
│── README.md                       # File mô tả dự án
│── requirements.txt                # Các thư viện Python cần cài đặt
```

---

## 🧠 Mục tiêu dự án

- Trích xuất **deep features** từ toàn bộ ảnh CIFAR-10.
- Xây dựng hệ thống tìm kiếm ảnh tương tự dựa trên:
  - Khoảng cách **Euclidean / Cosine**
  - Hoặc thuật toán **KNN**
- Tạo giao diện Web để tải lên ảnh và trả về kết quả ảnh giống nhất.

---

## 📌 Các thành phần chính

### **1. Trích xuất đặc trưng (feature_extractor.py)**
- Resize ảnh → (224 × 224)
- Normalize theo chuẩn ImageNet
- Dùng ResNet-18 (pretrained ImageNet)
- Xuất ra vector 512 chiều cho mỗi ảnh

### **2. Backend (main.py)**
- Load `features.npy`
- Load danh sách ảnh Base64
- Tìm ảnh tương tự bằng KNN hoặc Cosine
- API trả kết quả cho frontend

### **3. Frontend (index.html + style.css + app.js)**
- Cho phép người dùng upload ảnh
- Gửi request đến API backend
- Hiển thị ảnh tương tự theo độ giống cao → thấp

---

## ⚙️ Cài đặt thư viện

Chạy lệnh sau để cài toàn bộ thư viện:

```
pip install -r requirements.txt
```

---

## ▶️ Chạy hệ thống

### **1. Trích xuất đặc trưng (nếu chưa có):**

```
python src/feature_extractor.py
```

### **2. Khởi chạy backend:**

```
python src/main.py
```

### **3. Mở giao diện Web**
Truy cập:

```
http://localhost:5000
```

---

## 📦 Output của hệ thống trích xuất đặc trưng

| File | Mô tả |
|------|-------|
| `features.npy` | Ma trận (60000 × 512) chứa embedding của mỗi ảnh |
| `image_list.txt` | Danh sách ảnh mã hoá Base64 phục vụ frontend |

---

## 📝 Ghi chú
- Dự án hoạt động tốt trên CPU, nhưng GPU sẽ nhanh hơn nhiều.
- Có thể mở rộng dataset khác hoặc model mạnh hơn (ResNet50, ViT…).

---

## 👨‍💻 Tác giả
Giang Lê Hoàng - 2005

