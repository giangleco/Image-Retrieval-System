
# 🔍 Hệ thống Truy Xuất Hình Ảnh Tương Tự (Image Retrieval System)

Dự án xây dựng một **hệ thống tìm kiếm ảnh tương tự** dựa trên đặc trưng sâu (deep features) từ mô hình **ResNet-18** (pretrained trên ImageNet) trên bộ dữ liệu **CIFAR-10** (60.000 ảnh).

Hệ thống hỗ trợ:
- Upload ảnh bất kỳ hoặc chọn ảnh mẫu
- Tìm và hiển thị **10 ảnh giống nhất** trong tích tắc
- So sánh hiệu suất giữa **KNN truyền thống** và **FAISS** (Facebook AI Similarity Search)

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


---

## 🧠 Mục tiêu & Điểm nổi bật

- Trích xuất **deep features** bằng ResNet-18
- So sánh hiệu suất giữa:
  - **KNN** (scikit-learn, brute-force)
  - **FAISS** (Facebook AI Similarity Search – công nghệ hiện đại nhất hiện nay)
- Đánh giá khoa học bằng:
  - **Tốc độ tìm kiếm** (ms/query)
  - **Recall@10** (độ chính xác truy xuất)
- Giao diện web **đẹp, responsive**, hỗ trợ upload + preview ảnh
- Hoạt động hoàn toàn **offline**, không cần Internet sau khi trích xuất dữ liệu

---

## ⚙️ Công nghệ sử dụng

| Công nghệ              | Mục đích sử dụng                                  |
|------------------------|---------------------------------------------------|
| PyTorch + TorchVision  | Trích xuất đặc trưng bằng ResNet-18 pretrained    |
| NumPy                  | Xử lý và lưu trữ vector đặc trưng                 |
| scikit-learn           | KNN truyền thống (để so sánh)                     |
| **FAISS**              | **Tìm kiếm vector siêu nhanh** (chính thức dùng)  |
| Flask                  | Backend web                                       |
| HTML/CSS/JS            | Giao diện người dùng đẹp, mượt mà                 |

---

## ▶️ Hướng dẫn cài đặt & chạy dự án

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## ▶️ Chạy hệ thống

### **1. Trích xuất đặc trưng (nếu chưa có):**

```
python src/feature_extractor.py
```
## 📦 Output của hệ thống trích xuất đặc trưng

| File | Mô tả |
|------|-------|
| `features.npy` | Ma trận (60000 × 512) chứa embedding của mỗi ảnh |
| `image_list.txt` | Danh sách ảnh mã hoá Base64 phục vụ frontend |

---
### **2. Khởi chạy backend:**

```
python src/main.py
```

### **3. Mở giao diện Web**
Truy cập:

```
http://localhost:5000
```
## 🚀 Nâng cấp nổi bật: Tích hợp FAISS
Hệ thống sử dụng FAISS (IndexFlatIP + cosine similarity) làm phương pháp tìm kiếm chính thức vì:

- Tốc độ: Nhanh hơn KNN 20–50 lần (thường chỉ 1–3 ms/query)
- Độ chính xác: Recall@10 tương đương hoặc tốt hơn KNN (exact search)
- Khả năng mở rộng: Dễ dàng xử lý hàng triệu đến tỷ vector
Mỗi lần tìm kiếm (khi dùng ảnh mẫu), terminal sẽ in ra so sánh:
```bash
======================================================================
   SO SÁNH HIỆU SUẤT TÌM KIẾM (60.000 ảnh)
   → KNN   : 45.23 ms
   → FAISS : 1.87 ms
   → FAISS nhanh hơn: 24.2x
   → Recall@10 KNN   : 0.0015
   → Recall@10 FAISS : 0.0015
   → Độ chính xác tương đương
======================================================================
```



## 📝 Ghi chú
- Dự án hoạt động tốt trên CPU, nhưng GPU sẽ nhanh hơn nhiều.
- Có thể mở rộng dataset khác hoặc model mạnh hơn (ResNet50, ViT…).
- Có thể mở rộng bằng:
  - Model mạnh hơn (ResNet-50, EfficientNet, ViT)
  - Dataset lớn hơn (ImageNet, LAION)
  - Chỉ mục FAISS nâng cao (IVF, PQ, HNSW)

---

## 👨‍💻 Tác giả
Giang Lê Hoàng - 2005

