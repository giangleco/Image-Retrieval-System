"""
Provider chat: Azure OpenAI (GPT-4o / 4o-mini / ...).

Cấu hình qua biến môi trường (lấy trong Azure Portal > tài nguyên Azure OpenAI):
  - AZURE_OPENAI_API_KEY      : API key của tài nguyên
  - AZURE_OPENAI_ENDPOINT     : vd https://<ten>.openai.azure.com/
  - AZURE_OPENAI_DEPLOYMENT   : TÊN DEPLOYMENT của model (không phải tên model gốc)
  - AZURE_OPENAI_API_VERSION  : mặc định 2024-10-21

Phần prompt & chuẩn hoá kết quả nằm ở chat/prompts.py (dùng chung với Gemini).
Nếu THIẾU cấu hình hoặc GỌI LỖI, các hàm ở đây RAISE để phía gọi tự fallback.
"""
import os
import json

from chat import prompts

# Đọc cấu hình LAZY (trong hàm) chứ KHÔNG ở cấp module: nếu đọc lúc import,
# file .env có thể chưa được nạp -> Azure bị tắt âm thầm dù đã điền key.
def _deployment():
    return prompts.clean_env("AZURE_OPENAI_DEPLOYMENT")


def _api_version():
    return os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-10-21"

def _cfg_ok():
    return bool(
        prompts.clean_env("AZURE_OPENAI_API_KEY")
        and prompts.clean_env("AZURE_OPENAI_ENDPOINT")
        and _deployment()
    )


def is_enabled():
    """True nếu đủ cấu hình Azure và cài được SDK openai."""
    if not _cfg_ok():
        return False
    try:
        from openai import AzureOpenAI  # noqa: F401
        return True
    except ImportError:
        return False


_client = None


def _get_client():
    """Khởi tạo AzureOpenAI client 1 lần (lazy)."""
    global _client
    if _client is None:
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=prompts.clean_env("AZURE_OPENAI_API_KEY"),
            azure_endpoint=prompts.clean_env("AZURE_OPENAI_ENDPOINT"),
            api_version=_api_version(),
        )
    return _client


def parse_message(message, has_image, max_k=50):
    """Dùng Azure OpenAI phân tích câu lệnh -> {intent, k, class_filter, clip_query}. Raise nếu lỗi."""
    resp = _get_client().chat.completions.create(
        model=_deployment(),
        messages=[{"role": "user", "content": prompts.parse_prompt(message, has_image, max_k)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return prompts.normalize_plan(json.loads(resp.choices[0].message.content), has_image, max_k)


def smart_reply(message):
    """Sinh câu trả lời trò chuyện tự nhiên bằng Azure OpenAI. Raise nếu lỗi."""
    resp = _get_client().chat.completions.create(
        model=_deployment(),
        messages=[{"role": "user", "content": prompts.reply_prompt(message)}],
        temperature=0.7,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Azure OpenAI trả về rỗng")
    return text
