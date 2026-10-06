"""
Điểm khởi chạy backend API.

    uv run python src/main.py

Server lên ở http://localhost:5000 với:
    /          giao diện web (static, gọi API bằng fetch)
    /docs      tài liệu API tương tác (Swagger UI)
    /redoc     tài liệu API dạng đọc
    /api/...   REST API

Mã nguồn chia theo tầng trách nhiệm:
    core/       cấu hình + lõi tìm kiếm (FAISS, đặc trưng, metric)
    api/        route FastAPI + schema vào/ra
    chat/       trợ lý chat và các provider LLM
    embedding/  mô hình CLIP
    scripts/    script chạy một lần để trích xuất đặc trưng
"""
import sys
from pathlib import Path

# src/ là gốc để import (from core import ..., from api import ...)
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import uvicorn

from core import config


def main() -> None:
    line = "=" * 70
    print(f"\n{line}")
    print("   HỆ THỐNG TRUY XUẤT ẢNH CIFAR-10 — REST API")
    print(f"   Web         : http://localhost:{config.PORT}/")
    print(f"   Swagger UI  : http://localhost:{config.PORT}/docs")
    print(f"   Auto-reload : {'BẬT' if config.RELOAD else 'tắt'}")
    print(f"{line}\n")

    uvicorn.run(
        "api.routes:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        # app_dir để tiến trình con của chế độ reload cũng tìm thấy các gói trong src/
        app_dir=str(SRC_DIR),
        reload_dirs=[str(SRC_DIR)] if config.RELOAD else None,
    )


if __name__ == "__main__":
    main()
