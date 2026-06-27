
# 🔍 Hệ thống Truy Xuất Hình Ảnh Tương Tự (Image Retrieval System)

Dự án xây dựng một **hệ thống tìm kiếm ảnh tương tự** dựa trên đặc trưng sâu (deep features) từ mô hình **ResNet-18** (pretrained trên ImageNet) trên bộ dữ liệu **CIFAR-10** (60.000 ảnh).

Hệ thống hỗ trợ:
- Upload ảnh bất kỳ hoặc chọn ảnh mẫu
- Tìm và hiển thị **10 ảnh giống nhất** trong tích tắc
- So sánh hiệu suất giữa **KNN truyền thống** và **FAISS** (Facebook AI Similarity Search)

---

## 📂 Cấu trúc thư mục dự án

```
Image-Retrieval-System/
├── Data/                           # Dữ liệu CIFAR-10 (cùng cấp với src/)
│   └── cifar-10-batches-py/        # Bộ dữ liệu CIFAR-10 gốc
├── features/                       # Kết quả trích xuất đặc trưng (cùng cấp với src/)
│   ├── features.npy                # Ma trận đặc trưng (60000 × 512)
│   └── image_list.txt             # Ảnh dạng Base64 để hiển thị giao diện
├── src/
│   ├── Data_process.ipynb          # Notebook: download + tiền xử lý CIFAR-10
│   ├── Feature_extraction.ipynb   # Notebook: trích xuất đặc trưng ResNet-18
│   ├── feature_extractor.py      # Script trích xuất đặc trưng (chạy độc lập)
│   └── main.py                    # Backend Flask + API tìm kiếm
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── BAO_CAO_DANH_GIA.md            # Báo cáo đáp ứng tiêu chí đánh giá đồ án
├── README.md
├── pyproject.toml                 # Khai báo dependencies (thay cho requirements.txt)
├── uv.lock                        # Phiên bản thư viện đã khoá (commit lên Git)
├── Dockerfile                     # Định nghĩa image Docker
├── .dockerignore                  # File loại trừ khi build image
└── docker-compose.yml             # Cấu hình chạy bằng Docker Compose
```

---


---

## 🧠 Mục tiêu & Điểm nổi bật

- Trích xuất **deep features** bằng ResNet-18
- So sánh hiệu suất giữa:
  - **KNN** (scikit-learn, brute-force)
  - **FAISS** (Facebook AI Similarity Search – công nghệ hiện đại nhất hiện nay)
- Đánh giá khoa học bằng:
  - **Tốc độ tìm kiếm** (ms/query) – so sánh KNN vs FAISS
  - **Recall@10, Precision@10, AP** (Average Precision) – chất lượng retrieval
  - **Overlap@10** – độ trùng kết quả giữa KNN và FAISS (Jaccard)
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

Dự án dùng [**uv**](https://docs.astral.sh/uv/) để quản lý môi trường và thư viện (qua `pyproject.toml` + `uv.lock`).

### **Bước 1 — Cài uv & môi trường**

```bash
# Cài uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Tạo môi trường ảo + cài đúng phiên bản thư viện đã khoá trong uv.lock
# (uv tự tải Python 3.12, dùng wheel PyTorch CPU-only)
uv sync
```

> ⚠️ **Lưu ý về file dữ liệu:** khi mới `git clone`, repo **chưa có** thư mục `Data/`
> (bộ CIFAR-10) lẫn `features/` (vector đặc trưng) vì chúng rất nặng nên bị
> `.gitignore`. Cả hai sẽ được **tự tạo** ở Bước 2 dưới đây.

### **Bước 2 — Tải dữ liệu & trích xuất đặc trưng**

Chạy script này **một lần duy nhất**. Nó sẽ tự động:
1. **Tải bộ CIFAR-10** (~170MB) về `Data/` nếu chưa có (cần Internet ở lần đầu).
2. Trích đặc trưng bằng ResNet-18 và ghi 3 file vào `features/`.

```bash
uv run python src/feature_extractor.py
```

**Output sinh ra trong `features/`:**

| File | Mô tả |
|------|-------|
| `features.npy` | Ma trận (60000 × 512) chứa embedding của mỗi ảnh |
| `labels.npy` | Nhãn lớp của từng ảnh |
| `image_list.txt` | Danh sách ảnh mã hoá Base64 (64×64) phục vụ frontend |

> 💡 Lần đầu chạy sẽ mất vài phút (tải dữ liệu + tải trọng số ResNet-18 từ Internet).
> Những lần sau đã có sẵn `Data/` và `features/` nên **không cần chạy lại**.

### **Bước 3 — Khởi chạy web server**

```bash
uv run python src/main.py
```

Sau đó mở trình duyệt tại:

```
http://localhost:5000
```

---

## 🐳 Chạy bằng Docker

Nếu không muốn cài uv/Python trực tiếp lên máy, có thể chạy toàn bộ hệ thống bằng Docker.

### Các file Docker trong dự án

| File | Vai trò |
|------|---------|
| `Dockerfile` | Định nghĩa cách **đóng gói** ứng dụng thành image: cài thư viện theo `uv.lock`, copy mã nguồn, tải sẵn trọng số ResNet-18. |
| `.dockerignore` | Liệt kê thứ **không** đưa vào image (`.venv`, `Data/`, `features/`, `.git`…) để build nhẹ & nhanh. |
| `docker-compose.yml` | Cấu hình chạy: map cổng `5000`, **mount** `Data/` và `features/` từ máy host vào container. |

> 📌 **Vì sao mount volume?** Bộ CIFAR-10 và file đặc trưng rất nặng nên **không** nhúng vào
> image. Thay vào đó chúng được gắn (mount) từ thư mục trên máy host lúc chạy — file sinh ra
> trong container vẫn được lưu lại trên máy bạn.

### Bước 1 — Build image

```bash
docker compose build
```

### Bước 2 — Tải dữ liệu & trích xuất đặc trưng (chạy 1 lần)

Lệnh dưới chạy script trích xuất **bên trong container**; nhờ mount volume, kết quả
(`Data/` và `features/`) được ghi ra thư mục dự án trên máy host:

```bash
docker compose run --rm web uv run python src/feature_extractor.py
```

### Bước 3 — Khởi chạy web server

```bash
docker compose up
```

Mở trình duyệt tại **http://localhost:5000**. Nhấn `Ctrl+C` để dừng, hoặc chạy nền bằng
`docker compose up -d` và dừng bằng `docker compose down`.

> ⚠️ Phải chạy **Bước 2 trước**. Nếu `features/` còn trống, server sẽ báo lỗi
> `FileNotFoundError` vì chưa có dữ liệu đặc trưng để tìm kiếm.

#### (Tuỳ chọn) Không dùng compose, chạy bằng `docker` thuần

```bash
# Build
docker build -t image-retrieval-system .

# Trích xuất đặc trưng
docker run --rm -v "$PWD/Data:/app/Data" -v "$PWD/features:/app/features" \
  image-retrieval-system uv run python src/feature_extractor.py

# Chạy server
docker run --rm -p 5000:5000 -v "$PWD/features:/app/features" \
  image-retrieval-system
```

---

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

---

## 📋 Báo cáo đánh giá (tiêu chí đồ án)

Các tiêu chí đánh giá đồ án (xác định vấn đề & chiến lược, chỉ số đo lường, cải tiến thuật toán, đánh giá chất lượng mô hình, thảo luận kết quả, hướng cải thiện, tóm tắt giải pháp, điểm thú vị/khó) được trình bày chi tiết trong:

**[BAO_CAO_DANH_GIA.md](BAO_CAO_DANH_GIA.md)**

---

## 📝 Ghi chú
- Dự án hoạt động tốt trên CPU, nhưng GPU sẽ nhanh hơn nhiều.
- Có thể mở rộng dataset khác hoặc model mạnh hơn (ResNet50, ViT…).
- **Fine-tune ResNet-18:** Chạy notebook `src/Fine_tune_ResNet.ipynb` để huấn luyện ResNet-18 trên CIFAR-10; mô hình lưu tại `model/resnet18_finetuned_cifar10.pt`. Trong `feature_extractor.py` đặt `USE_FINETUNED = True` rồi chạy lại để trích đặc trưng bằng mô hình đã fine-tune (retrieval thường tốt hơn).
- Có thể mở rộng bằng:
  - Model mạnh hơn (ResNet-50, EfficientNet, ViT)
  - Dataset lớn hơn (ImageNet, LAION)
  - Chỉ mục FAISS nâng cao (IVF, PQ, HNSW)

---

## 👨‍💻 Tác giả
Giang Lê Hoàng - 2005

