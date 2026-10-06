# syntax=docker/dockerfile:1

# ============================================================================
#  Image cơ sở: đã có sẵn uv + Python 3.12 (bản slim cho nhẹ)
#  -> không cần tự cài uv hay Python trong container
# ============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# ---------------------------------------------------------------------------
#  Biến môi trường tinh chỉnh uv & Python
# ---------------------------------------------------------------------------
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    TORCH_HOME=/app/.torch_cache \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5000

# Thư mục làm việc bên trong container
WORKDIR /app

# ---------------------------------------------------------------------------
#  BƯỚC 1: Cài dependencies TRƯỚC (chỉ copy 2 file khai báo)
#  -> Nếu code thay đổi nhưng dependencies không đổi, Docker dùng lại cache
#     của layer này => build lại rất nhanh.
#  --frozen : cài đúng theo uv.lock, không tự đổi lock
#  --no-dev : bỏ qua nhóm dev (ở đây không có, nhưng để cho chuẩn production)
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
#  BƯỚC 2: Tải sẵn TRỌNG SỐ MÔ HÌNH (~400MB) vào image
#  -> lúc chạy KHÔNG cần Internet và không bị treo ở truy vấn đầu tiên.
#
#  Vì sao nhúng vào image? Nếu không, CLIP sẽ tải LÚC CHẠY vào lớp ghi của
#  container — lớp này mất khi container bị xoá, nên mỗi lần
#  `docker compose down && up` lại phải chờ tải lại ~5 phút.
#
#  Vì sao đặt TRƯỚC khi copy mã nguồn? Docker huỷ cache của mọi layer phía sau
#  layer vừa đổi. Đặt sau COPY thì mỗi lần sửa code (kể cả sửa CSS) đều phải
#  tải lại 354MB. Đặt trước thì sửa code chỉ build lại vài giây.
#
#  Tên model phải KHỚP mặc định trong src/embedding/clip_model.py — có bước
#  kiểm tra ở BƯỚC 4 bên dưới để bắt lỗi nếu hai chỗ lệch nhau.
# ---------------------------------------------------------------------------
RUN uv run python -c "import torchvision.models as m; m.resnet18(weights='IMAGENET1K_V1')" \
 && uv run python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"

# ---------------------------------------------------------------------------
#  BƯỚC 3: Copy mã nguồn ứng dụng
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY static/ ./static/

# ---------------------------------------------------------------------------
#  BƯỚC 4: Xác nhận weights đã nhúng khớp với cấu hình của ứng dụng.
#  Khớp  -> nạp từ cache, vài giây.
#  Lệch  -> tải đúng model NGAY LÚC BUILD (vẫn hơn là để người dùng chờ lúc chạy).
# ---------------------------------------------------------------------------
RUN uv run python -c "import sys; sys.path.insert(0, 'src'); from embedding import clip_model; clip_model.load()"

# Cổng web mà uvicorn lắng nghe
EXPOSE 5000

# ---------------------------------------------------------------------------
#  Lệnh mặc định: chạy API server (uvicorn) — xem src/main.py.
#  (Phần trích xuất đặc trưng chạy bằng lệnh override riêng — xem README)
# ---------------------------------------------------------------------------
CMD ["uv", "run", "python", "src/main.py"]
