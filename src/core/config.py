"""
Cấu hình tập trung cho backend API.

Module này được import ĐẦU TIÊN (trước mọi provider LLM) nên .env chắc chắn đã
được nạp vào os.environ trước khi bất kỳ module nào đọc key — tránh trường hợp
provider bị tắt âm thầm vì đọc biến môi trường quá sớm.
"""
from __future__ import annotations

import os
from pathlib import Path

# config.py nằm ở src/core/ nên gốc dự án lùi 2 cấp
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Nạp biến môi trường từ .env (không cần thư viện ngoài).

    Biến đã tồn tại sẵn trong môi trường được GIỮ NGUYÊN (không ghi đè), để
    cấu hình truyền từ shell/Docker luôn thắng file .env.
    """
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()   # <-- chạy NGAY khi import module này

# ===========================================================
#  Đường dẫn
# ===========================================================
FEATURES_DIR = PROJECT_ROOT / "features"
FEATURES_NPY = FEATURES_DIR / "features.npy"            # đặc trưng ResNet-18
CLIP_FEATURES_NPY = FEATURES_DIR / "features_clip.npy"  # đặc trưng CLIP (tuỳ chọn)
LABELS_NPY = FEATURES_DIR / "labels.npy"
IMAGE_LIST_TXT = FEATURES_DIR / "image_list.txt"
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"

# ===========================================================
#  Miền dữ liệu CIFAR-10
# ===========================================================
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_CLASSES_VN = [
    "máy bay", "ô tô", "chim", "mèo", "hươu/nai",
    "chó", "ếch", "ngựa", "tàu thuyền", "xe tải",
]
NUM_CLASSES = len(CIFAR10_CLASSES)

# ===========================================================
#  Tham số API
# ===========================================================
DEFAULT_K = 10                 # số kết quả mặc định
MAX_K = 50                     # trần số kết quả cho mỗi truy vấn
GALLERY_DEFAULT_LIMIT = 300    # số ảnh gallery trả về mặc định
GALLERY_MAX_LIMIT = 1000

# ===========================================================
#  Chatbot
# ===========================================================
# "auto" (thử lần lượt theo PROVIDER_PRIORITY) | "azure" | "gemini" | "none"
PROVIDER_PRIORITY = ["azure", "gemini"]

# Bí danh thường bị gõ nhầm -> tên chuẩn. Nếu không có bảng này, một giá trị như
# CHAT_PROVIDER=google sẽ không khớp provider nào và chatbot LLM bị tắt âm thầm.
PROVIDER_ALIASES = {
    "google": "gemini", "google-genai": "gemini", "googleai": "gemini",
    "google_ai": "gemini", "genai": "gemini", "gemini": "gemini",
    "azure": "azure", "azure-openai": "azure", "azure_openai": "azure",
    "openai": "azure", "auto": "auto", "none": "none", "off": "none",
}

_raw_provider = (
    os.environ.get("CHAT_PROVIDER") or os.environ.get("LLM_PROVIDER") or "auto"
).strip().lower()

CHAT_PROVIDER = PROVIDER_ALIASES.get(_raw_provider, _raw_provider)
# Giá trị lạ (không phải bí danh nào) -> giữ nguyên để chat/service.py cảnh báo rõ
CHAT_PROVIDER_IS_VALID = _raw_provider in PROVIDER_ALIASES
CHAT_PROVIDER_RAW = _raw_provider

# ===========================================================
#  Server
# ===========================================================
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

# Bật auto-reload khi sửa code. Mặc định TẮT (khác Flask cũ mặc định bật) vì
# reload làm nạp lại toàn bộ features + model, rất chậm. FLASK_DEBUG giữ lại
# cho tương thích với docker-compose.yml sẵn có.
RELOAD = (
    os.environ.get("API_RELOAD") or os.environ.get("FLASK_DEBUG") or "0"
).strip().lower() in ("1", "true", "yes", "on")

# Origin được phép gọi API (CORS). "*" = mọi nơi — tiện cho dev / frontend tách rời.
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
]


def class_name(index: int) -> str:
    """index lớp -> tên tiếng Anh (an toàn nếu ngoài phạm vi)."""
    index = int(index)
    return CIFAR10_CLASSES[index] if 0 <= index < NUM_CLASSES else str(index)
