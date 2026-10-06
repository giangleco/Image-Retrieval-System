
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
├── src/
│   ├── main.py                     # Điểm khởi chạy: bật uvicorn
│   ├── core/                       # Nền tảng — KHÔNG phụ thuộc web framework
│   │   ├── config.py               #   Cấu hình tập trung + nạp .env + hằng số miền
│   │   └── retrieval.py            #   LÕI tìm kiếm: FAISS, đặc trưng, metric
│   ├── api/                        # Tầng HTTP
│   │   ├── routes.py               #   Khai báo route FastAPI (mỏng)
│   │   └── schemas.py              #   Kiểu vào/ra Pydantic -> sinh Swagger
│   ├── chat/                       # Trợ lý chat
│   │   ├── service.py              #   Nghiệp vụ: hiểu ý định, chọn cách tìm
│   │   ├── prompts.py              #   Prompt & chuẩn hoá kết quả dùng chung
│   │   ├── rule_based.py           #   Parser offline (không cần API key)
│   │   └── providers/              #   Các LLM thay thế được cho nhau
│   │       ├── azure.py
│   │       └── gemini.py
│   ├── embedding/
│   │   └── clip_model.py           # Load CLIP + encode ảnh/chữ (kèm dịch Việt→Anh)
│   └── scripts/                    # Chạy MỘT LẦN, ngoài server
│       ├── extract_resnet.py       #   -> features.npy, labels.npy, image_list.txt
│       ├── extract_clip.py         #   -> features_clip.npy
│       └── kaggle_extract_clip.py  #   Bản tự chứa để dán vào Kaggle GPU
├── static/                         # Frontend tĩnh, gọi API bằng fetch
│   ├── index.html                  # Bố cục 2 cột: trái chat, phải kết quả
│   ├── css/style.css               # Hệ token màu sáng (biến CSS ở :root)
│   └── js/
│       ├── api.js                  # Lớp bọc REST API (đổi API_BASE là xong)
│       ├── app.js                  # Thanh trạng thái, gallery, khu KẾT QUẢ dùng chung
│       └── chat.js                 # Khung chat (kết quả đẩy sang app.js render)
├── data/                           # CIFAR-10 (tự tải, đã gitignore)
├── features/                       # Vector đặc trưng (đã gitignore)
│   ├── features.npy                #   ResNet-18 (60000 × 512) — tìm bằng ảnh
│   ├── features_clip.npy           #   CLIP (60000 × 512) — tìm bằng mô tả
│   ├── labels.npy
│   └── image_list.txt              #   Ảnh Base64 64×64 cho giao diện
├── docs/                           # Báo cáo, không liên quan runtime
│   ├── BAO_CAO_DANH_GIA.md
│   ├── Nhom5_Hocmay.pdf
│   └── Nhom5_Hocmay.docx
├── README.md
├── pyproject.toml                  # Khai báo dependencies
├── uv.lock                         # Phiên bản thư viện đã khoá (commit lên Git)
├── Dockerfile
├── .dockerignore
└── docker-compose.yml
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
- Giao diện web **2 cột** (chat bên trái, kết quả bên phải), nền sáng, chạy tốt trên điện thoại
- Hoạt động hoàn toàn **offline**, không cần Internet sau khi trích xuất dữ liệu

---

## ⚙️ Công nghệ sử dụng

| Công nghệ              | Mục đích sử dụng                                  |
|------------------------|---------------------------------------------------|
| PyTorch + TorchVision  | Trích xuất đặc trưng bằng ResNet-18 pretrained    |
| **OpenCLIP** (open-clip-torch) | **Tìm ảnh bằng mô tả văn bản** (ảnh & chữ cùng không gian) |
| NumPy                  | Xử lý, chuẩn hóa L2 & lưu trữ vector đặc trưng     |
| **FAISS**              | **Tìm kiếm vector siêu nhanh** (chính thức dùng)  |
| **FastAPI + Uvicorn**  | **Backend REST API** (tự sinh tài liệu Swagger)   |
| Pydantic               | Kiểm tra tham số đầu vào & khai báo schema JSON    |
| HTML/CSS/JS            | Giao diện tĩnh, gọi API bằng `fetch`              |

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

> ⚠️ **Lưu ý về file dữ liệu:** khi mới `git clone`, repo **chưa có** thư mục `data/`
> (bộ CIFAR-10) lẫn `features/` (vector đặc trưng) vì chúng rất nặng nên bị
> `.gitignore`. Cả hai sẽ được **tự tạo** ở Bước 2 dưới đây.

### **Bước 2 — Tải dữ liệu & trích xuất đặc trưng**

Chạy script này **một lần duy nhất**. Nó sẽ tự động:
1. **Tải bộ CIFAR-10** (~170MB) về `data/` nếu chưa có (cần Internet ở lần đầu).
2. Trích đặc trưng bằng ResNet-18 và ghi 3 file vào `features/`.

```bash
uv run python src/scripts/extract_resnet.py
```

**Output sinh ra trong `features/`:**

| File | Mô tả |
|------|-------|
| `features.npy` | Ma trận (60000 × 512) chứa embedding của mỗi ảnh |
| `labels.npy` | Nhãn lớp của từng ảnh |
| `image_list.txt` | Danh sách ảnh mã hoá Base64 (64×64) phục vụ frontend |

> 💡 Lần đầu chạy sẽ mất vài phút (tải dữ liệu + tải trọng số ResNet-18 từ Internet).
> Những lần sau đã có sẵn `data/` và `features/` nên **không cần chạy lại**.

### **Bước 2b (tuỳ chọn) — Trích đặc trưng CLIP để bật tìm bằng văn bản**

Muốn dùng tính năng **gõ mô tả ra ảnh**, cần thêm file `features/features_clip.npy`:

```bash
uv run python src/scripts/extract_clip.py
```

> ⚙️ Bước này trên CPU **rất chậm** (~1 giờ cho 60k ảnh). Khuyến nghị trích nhanh trên
> **Kaggle GPU** (~1–2 phút) bằng `src/scripts/kaggle_extract_clip.py` rồi tải `features_clip.npy`
> về bỏ vào `features/`. Nếu **thiếu** file này, app vẫn chạy bình thường nhưng chỉ
> **tìm bằng ảnh**; phần tìm bằng mô tả sẽ tự tắt.

### **Bước 2c (tuỳ chọn) — Bật chatbot thông minh bằng Gemini**

Mặc định chatbot dùng bộ hiểu lệnh **rule-based (offline)**. Muốn chatbot hiểu ngôn ngữ
tự nhiên linh hoạt hơn và trả lời tự nhiên, hãy thêm **API key của Google AI Studio**:

1. Lấy key miễn phí tại **https://aistudio.google.com/apikey**
2. Tạo file `.env` ở gốc dự án (copy từ `.env.example`) và điền key:
   ```bash
   cp .env.example .env
   # rồi sửa GEMINI_API_KEY=... trong .env
   ```

> 🔒 File `.env` đã được `.gitignore` bỏ qua — **không** commit key thật lên Git.
> Nếu **không** đặt key, chatbot tự động chạy chế độ rule-based (offline), app vẫn hoạt động bình thường.

### **Bước 3 — Khởi chạy web server**

```bash
uv run python src/main.py
```

Sau đó mở trình duyệt tại:

| Địa chỉ | Nội dung |
|---------|----------|
| http://localhost:5000/ | Giao diện web |
| http://localhost:5000/docs | **Swagger UI** — thử API trực tiếp trên trình duyệt |
| http://localhost:5000/redoc | Tài liệu API dạng đọc |

Khi khởi động, dòng log `>>> Chatbot: ...` cho biết chuỗi provider đang dùng,
ví dụ `gemini -> rule-based`.

---

## 🔌 REST API

Backend là **API thuần**: mọi thứ frontend hiển thị đều lấy qua HTTP + JSON, nên có thể
gọi từ ứng dụng khác (mobile, notebook, script đánh giá…) chứ không chỉ từ trang web này.
Trang tĩnh trong `static/` chỉ là **một client mẫu** — toàn bộ lời gọi mạng gom trong
[static/js/api.js](static/js/api.js), đổi `API_BASE` là trỏ được sang server khác.

Mở **http://localhost:5000/docs** để xem tài liệu tự sinh và bấm *Try it out* thử ngay
trên trình duyệt, không cần cài Postman.

### Danh sách endpoint

| Method | Đường dẫn | Chức năng |
|--------|-----------|-----------|
| `GET` | `/api/health` | Trạng thái server: số ảnh, CLIP bật/tắt, chuỗi chatbot |
| `GET` | `/api/classes` | 10 lớp CIFAR-10 kèm tên tiếng Việt & số ảnh mỗi lớp |
| `GET` | `/api/images` | Ảnh trong kho, phân trang (`offset`, `limit`) |
| `GET` | `/api/images/{idx}/similar` | Tìm ảnh giống một ảnh **có sẵn trong kho** → kèm **metrics** |
| `POST` | `/api/search/image` | Tìm ảnh giống một ảnh **tải lên** (multipart) |
| `POST` | `/api/search/text` | Tìm ảnh bằng **mô tả văn bản** qua CLIP (JSON) |
| `POST` | `/api/chat` | Trợ lý: nhận `message` và/hoặc ảnh đính kèm |

Tham số dùng chung: `k` (1…50, mặc định 10) và `class_filter` (0–9, bỏ trống = tìm toàn kho).

### Ví dụ

```bash
# Trạng thái hệ thống
curl http://localhost:5000/api/health

# Tìm 5 ảnh giống ảnh số 7 trong kho (có Precision/Recall/AP vì ảnh này có nhãn)
curl "http://localhost:5000/api/images/7/similar?k=5"

# Tìm bằng mô tả (CLIP) — hỗ trợ cả tiếng Việt
curl -X POST http://localhost:5000/api/search/text \
     -H "Content-Type: application/json" \
     -d '{"q":"a red truck","k":5}'

# Tìm bằng ảnh tải lên
curl -X POST http://localhost:5000/api/search/image \
     -F "file=@anh_cua_ban.jpg" -F "k=10"

# Trò chuyện với trợ lý (có thể kèm ảnh)
curl -X POST http://localhost:5000/api/chat \
     -F "message=tìm 8 ảnh con chó"
```

### Mã lỗi

| Mã | Khi nào |
|----|---------|
| `400` | File tải lên không phải ảnh hợp lệ; hoặc gọi `/api/chat` mà không có cả chữ lẫn ảnh |
| `404` | `idx` vượt quá số ảnh trong kho |
| `422` | Tham số sai kiểu hoặc ngoài khoảng (`k=999`, `class_filter=99`, thiếu `q`…) |
| `503` | Gọi `/api/search/text` khi chưa có `features/features_clip.npy` (CLIP tắt) |

### Kiến trúc mã nguồn

Backend tách 4 tầng, mỗi tầng một việc:

| Tầng | Thư mục | Trách nhiệm |
|------|---------|-------------|
| Khởi chạy | `src/main.py` | Bật uvicorn |
| API | `src/api/` | Nhận request, kiểm tra tham số, trả JSON |
| Nghiệp vụ | `src/chat/` | Hiểu ý định người dùng, chọn cách tìm phù hợp |
| Lõi | `src/core/` | Cấu hình, FAISS, trích đặc trưng, metric — **không dính web** |
| Mô hình | `src/embedding/` | Bộ mã hoá CLIP |
| Ngoài luồng | `src/scripts/` | Trích đặc trưng, chạy một lần |

Quy tắc phụ thuộc đi **một chiều**: `api` → `chat` → `core`. Tầng `core` không biết gì
về FastAPI lẫn chatbot, nên có thể dùng thẳng trong notebook để chạy thử nghiệm:

```python
import sys; sys.path.insert(0, "src")
from core import retrieval
results, labels = retrieval.search_by_index(7, k=10)
```

---

## 🐳 Chạy bằng Docker

Nếu không muốn cài uv/Python trực tiếp lên máy, có thể chạy toàn bộ hệ thống bằng Docker.

### Các file Docker trong dự án

| File | Vai trò |
|------|---------|
| `Dockerfile` | Định nghĩa cách **đóng gói** ứng dụng thành image: cài thư viện theo `uv.lock`, copy mã nguồn, **tải sẵn trọng số ResNet-18 và CLIP**. |
| `.dockerignore` | Liệt kê thứ **không** đưa vào image (`.venv`, `data/`, `features/`, `docs/`, `.git`…) để build nhẹ & nhanh. |
| `docker-compose.yml` | Cấu hình chạy: map cổng `5000`, **mount** `data/` và `features/` từ máy host vào container. |

> 📌 **Vì sao mount volume?** Bộ CIFAR-10 và file đặc trưng rất nặng nên **không** nhúng vào
> image. Thay vào đó chúng được gắn (mount) từ thư mục trên máy host lúc chạy — file sinh ra
> trong container vẫn được lưu lại trên máy bạn.

### Bước 1 — Build image

```bash
docker compose build
```

### Bước 2 — Tải dữ liệu & trích xuất đặc trưng (chạy 1 lần)

Lệnh dưới chạy script trích xuất **bên trong container**; nhờ mount volume, kết quả
(`data/` và `features/`) được ghi ra thư mục dự án trên máy host:

```bash
docker compose run --rm web uv run python src/scripts/extract_resnet.py
```

### Bước 3 — Khởi chạy web server

```bash
docker compose up
```

Mở trình duyệt tại **http://localhost:5000**. Nhấn `Ctrl+C` để dừng, hoặc chạy nền bằng
`docker compose up -d` và dừng bằng `docker compose down`.

> 💡 **Vì sao image nặng ~2,4GB?** Trọng số ResNet-18 và CLIP (~354MB) được nhúng sẵn
> vào image ngay lúc build. Nhờ vậy container chạy **offline hoàn toàn** và truy vấn
> tìm-bằng-chữ đầu tiên không phải chờ tải model. Nếu không nhúng, model sẽ tải lúc chạy
> vào *lớp ghi của container* — lớp này mất khi container bị xoá, nên mỗi lần
> `docker compose down && up` lại phải tải lại từ đầu (~5 phút chờ).

> ⚠️ Phải chạy **Bước 2 trước**. Nếu `features/` còn trống, server sẽ báo lỗi
> `FileNotFoundError` vì chưa có dữ liệu đặc trưng để tìm kiếm.

#### (Tuỳ chọn) Không dùng compose, chạy bằng `docker` thuần

```bash
# Build
docker build -t image-retrieval-system .

# Trích xuất đặc trưng
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/features:/app/features" \
  image-retrieval-system uv run python src/scripts/extract_resnet.py

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
- Hỗ trợ **tiếng Việt** qua bước dịch nhanh Việt→Anh trong [src/embedding/clip_model.py](src/embedding/clip_model.py) (CLIP gốc là tiếng Anh).
- Chạy **offline, không cần API key**.

Ví dụ gõ trong khung chat: *"a red truck"*, *"con mèo"*, *"máy bay trên bầu trời"*, *"tìm 5 ảnh con chó"*.

---

## 📋 Báo cáo đánh giá 

Các tiêu chí đánh giá đồ án (xác định vấn đề & chiến lược, chỉ số đo lường, cải tiến thuật toán, đánh giá chất lượng mô hình, thảo luận kết quả, hướng cải thiện, tóm tắt giải pháp, điểm thú vị/khó) được trình bày chi tiết trong:

**[docs/BAO_CAO_DANH_GIA.md](docs/BAO_CAO_DANH_GIA.md)**

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

