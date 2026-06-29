
# 🔍 Hệ thống Truy Xuất Hình Ảnh Tương Tự (Image Retrieval System)

Dự án xây dựng một **hệ thống tìm kiếm ảnh tương tự** trên bộ dữ liệu **CIFAR-10** (60.000 ảnh), với **hai cách tìm**:
- **Tìm bằng ảnh** — đặc trưng sâu từ **ResNet-18** (pretrained ImageNet) + **FAISS**.
- **Tìm bằng mô tả văn bản** — nhúng ảnh & chữ chung một không gian bằng **CLIP** (vd gõ *"a red truck"*, *"con mèo"*).

Hệ thống hỗ trợ:
- **Trợ lý chat (một khung duy nhất)**: **đính kèm ảnh** để tìm ảnh giống, **hoặc** chỉ **gõ mô tả** — chạy **offline, không cần API key**
- **Tìm bằng mô tả tự nhiên (CLIP)**, hỗ trợ cả tiếng Việt (vd *"tìm 5 ảnh con chó"*, *"máy bay trên bầu trời"*)
- Nói kèm **số lượng K** và **lọc theo lớp** ngay trong câu chat (vd *"tìm 20 ảnh con mèo"*)
- Hiển thị **nhãn lớp + % độ tương đồng** cho từng kết quả
- Khi tìm bằng ảnh mẫu (có nhãn): hiện **Precision@K, Recall@K, Average Precision** ngay trên web
- Tìm kiếm vector siêu nhanh bằng **FAISS** (Facebook AI Similarity Search)

---

## 📂 Cấu trúc thư mục dự án

```
Image-Retrieval-System/
├── Data/                           # Dữ liệu CIFAR-10 (tự tải, đã gitignore)
│   └── cifar-10-batches-py/
├── features/                       # Kết quả trích xuất (đã gitignore)
│   ├── features.npy                # Đặc trưng ResNet-18 (60000 × 512) — tìm bằng ảnh
│   ├── features_clip.npy           # Đặc trưng CLIP (60000 × 512) — tìm bằng mô tả
│   ├── labels.npy                  # Nhãn lớp của từng ảnh
│   └── image_list.txt              # Ảnh Base64 (64×64) để hiển thị giao diện
├── src/
│   ├── feature_extractor.py        # Trích đặc trưng ResNet-18  -> features.npy
│   ├── clip_extractor.py           # Trích đặc trưng CLIP        -> features_clip.npy
│   ├── clip_model.py               # Load CLIP + encode ảnh/chữ (kèm dịch Việt→Anh)
│   ├── chat_agent.py               # Parser câu lệnh chat (rule-based, offline)
│   └── main.py                     # Backend Flask + API (FAISS, search, /chat)
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js                  # Click ảnh mẫu + hiển thị kết quả
│       └── chat.js                 # Logic khung chat
├── templates/
│   └── index.html
├── kaggle_extract_clip.py          # Trích đặc trưng CLIP trên Kaggle GPU (nhanh)
├── BAO_CAO_DANH_GIA.md             # Báo cáo đáp ứng tiêu chí đánh giá đồ án
├── README.md
├── pyproject.toml                  # Khai báo dependencies (thay cho requirements.txt)
├── uv.lock                         # Phiên bản thư viện đã khoá (commit lên Git)
├── Dockerfile                      # Định nghĩa image Docker
├── .dockerignore                   # File loại trừ khi build image
└── docker-compose.yml              # Cấu hình chạy bằng Docker Compose
```

---


---

## 🧠 Mục tiêu & Điểm nổi bật

- Trích xuất **deep features** bằng ResNet-18 (tìm bằng ảnh)
- **Tìm bằng văn bản (CLIP)**: nhúng ảnh & chữ chung không gian → gõ mô tả ra ảnh, hỗ trợ tiếng Việt
- Tìm kiếm vector siêu nhanh bằng **FAISS** (Facebook AI Similarity Search) — IndexFlatIP + cosine, exact search
- **Trợ lý chat** một khung: nhận cả ảnh lẫn mô tả, hiểu số lượng K & lớp từ ngôn ngữ tự nhiên
- Đánh giá khoa học bằng:
  - **Tốc độ tìm kiếm** (ms/query)
  - **Recall@K, Precision@K, AP** (Average Precision) – chất lượng retrieval
- Giao diện web **đẹp, responsive**, hỗ trợ upload + preview ảnh
- Hoạt động hoàn toàn **offline**, không cần Internet sau khi trích xuất dữ liệu

---

## ⚙️ Công nghệ sử dụng

| Công nghệ              | Mục đích sử dụng                                  |
|------------------------|---------------------------------------------------|
| PyTorch + TorchVision  | Trích xuất đặc trưng bằng ResNet-18 pretrained    |
| **OpenCLIP** (open-clip-torch) | **Tìm ảnh bằng mô tả văn bản** (ảnh & chữ cùng không gian) |
| NumPy                  | Xử lý, chuẩn hóa L2 & lưu trữ vector đặc trưng     |
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

### **Bước 2b (tuỳ chọn) — Trích đặc trưng CLIP để bật tìm bằng văn bản**

Muốn dùng tính năng **gõ mô tả ra ảnh**, cần thêm file `features/features_clip.npy`:

```bash
uv run python src/clip_extractor.py
```

> ⚙️ Bước này trên CPU **rất chậm** (~1 giờ cho 60k ảnh). Khuyến nghị trích nhanh trên
> **Kaggle GPU** (~1–2 phút) bằng `kaggle_extract_clip.py` rồi tải `features_clip.npy`
> về bỏ vào `features/`. Nếu **thiếu** file này, app vẫn chạy bình thường nhưng chỉ
> **tìm bằng ảnh**; phần tìm bằng mô tả sẽ tự tắt.

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

## 🚀 Công nghệ tìm kiếm: FAISS
Hệ thống dùng **FAISS** (IndexFlatIP + cosine similarity, sau khi L2-normalize) làm phương pháp tìm kiếm chính thức vì:

- **Tốc độ:** truy vấn thường chỉ 1–3 ms trên 60.000 vector 512 chiều
- **Độ chính xác:** IndexFlatIP là *exact search* (không nén vector) nên luôn trả về đúng top-K theo cosine
- **Khả năng mở rộng:** dễ dàng nâng lên index gần đúng (IVF, PQ, HNSW) khi xử lý hàng triệu–tỷ vector

Mỗi lần tìm kiếm bằng ảnh mẫu (có nhãn), terminal in ra thời gian và các chỉ số chất lượng:
```bash
======================================================================
   BÁO CÁO TÌM KIẾM  (k=10, lớp lọc=tất cả)
   → Thời gian tìm kiếm : 1.8700 ms
   → Precision@10: 0.7 | Recall@10: 0.0012 | AP: 0.83
======================================================================
```

---

## 🔤 Tìm bằng văn bản (CLIP)

Ngoài tìm bằng ảnh, hệ thống còn dùng **CLIP** để tìm ảnh từ **mô tả văn bản**:

- CLIP nhúng **ảnh và chữ vào cùng một không gian vector** → có thể so khớp câu mô tả với ảnh trong kho.
- Toàn bộ kho ảnh được mã hoá sẵn thành `features/features_clip.npy` (xem Bước 2b).
- Khi gõ mô tả, câu chữ được CLIP mã hoá rồi tìm bằng FAISS giống như tìm bằng ảnh.
- Hỗ trợ **tiếng Việt** qua bước dịch nhanh Việt→Anh trong [src/clip_model.py](src/clip_model.py) (CLIP gốc là tiếng Anh).
- Chạy **offline, không cần API key**.

Ví dụ gõ trong khung chat: *"a red truck"*, *"con mèo"*, *"máy bay trên bầu trời"*, *"tìm 5 ảnh con chó"*.

---

## 📋 Báo cáo đánh giá 

Các tiêu chí đánh giá đồ án (xác định vấn đề & chiến lược, chỉ số đo lường, cải tiến thuật toán, đánh giá chất lượng mô hình, thảo luận kết quả, hướng cải thiện, tóm tắt giải pháp, điểm thú vị/khó) được trình bày chi tiết trong:

**[BAO_CAO_DANH_GIA.md](BAO_CAO_DANH_GIA.md)**

---

## 📝 Ghi chú
- Dự án hoạt động tốt trên CPU, nhưng GPU sẽ nhanh hơn nhiều.
- Có thể mở rộng dataset khác hoặc model mạnh hơn (ResNet50, ViT…).
- Có thể mở rộng bằng:
  - Model mạnh hơn (ResNet-50, EfficientNet, ViT)
  - Dataset lớn hơn (ImageNet, LAION)
  - Chỉ mục FAISS nâng cao (IVF, PQ, HNSW)

---

## 👨‍💻 Tác giả
Giang Lê Hoàng

