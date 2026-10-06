"""
Provider chat: Gemini (Google AI Studio).

Lấy API key tại: https://aistudio.google.com/apikey
Cấu hình qua biến môi trường:
  - GEMINI_API_KEY (hoặc GOOGLE_API_KEY)   : bắt buộc để bật Gemini
  - GEMINI_MODEL   (mặc định gemini-2.5-flash)

Phần prompt & chuẩn hoá kết quả nằm ở chat/prompts.py (dùng chung với provider khác).
Nếu THIẾU key hoặc GỌI LỖI, các hàm ở đây RAISE để phía gọi (main.py) tự fallback.
"""
import os
import json

from chat import prompts

# gemini-2.0-flash đã bị khai tử -> dùng model còn hỗ trợ. Đổi qua GEMINI_MODEL nếu cần.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def _api_key():
    return prompts.clean_env("GEMINI_API_KEY") or prompts.clean_env("GOOGLE_API_KEY")


def is_enabled():
    """True nếu có API key và cài được SDK google-genai."""
    if not _api_key():
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


_client = None


def _get_client():
    """Khởi tạo Gemini client 1 lần (lazy)."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=_api_key())
    return _client


def parse_message(message, has_image, max_k=50):
    """Dùng Gemini phân tích câu lệnh -> {intent, k, class_filter, clip_query}. Raise nếu lỗi."""
    from google.genai import types

    resp = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompts.parse_prompt(message, has_image, max_k),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return prompts.normalize_plan(json.loads(resp.text), has_image, max_k)


def smart_reply(message):
    """Sinh câu trả lời trò chuyện tự nhiên bằng Gemini. Raise nếu lỗi."""
    from google.genai import types

    resp = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompts.reply_prompt(message),
        config=types.GenerateContentConfig(temperature=0.7),
    )
    text = (resp.text or "").strip()
    if not text:
        raise ValueError("Gemini trả về rỗng")
    return text
