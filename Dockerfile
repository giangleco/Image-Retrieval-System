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
#  BƯỚC 2: Copy mã nguồn ứng dụng
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

# ---------------------------------------------------------------------------
#  BƯỚC 3: Tải sẵn trọng số ResNet-18 vào image
#  -> để lúc khởi động server KHÔNG cần Internet (offline-friendly)
# ---------------------------------------------------------------------------
RUN uv run python -c "import torchvision.models as m; m.resnet18(weights='IMAGENET1K_V1')"

# Cổng web mà Flask lắng nghe
EXPOSE 5000

# ---------------------------------------------------------------------------
#  Lệnh mặc định: chạy web server.
#  (Phần trích xuất đặc trưng chạy bằng lệnh override riêng — xem README)
# ---------------------------------------------------------------------------
CMD ["uv", "run", "python", "src/main.py"]
