"""
Tầng NGHIỆP VỤ của trợ lý chat — cũng không phụ thuộc web framework.

Gồm 2 phần:
  1. Chuỗi fallback hiểu câu lệnh:  Azure OpenAI -> Gemini -> rule-based (offline)
     Provider nào thiếu cấu hình hoặc gọi lỗi thì tự động rơi xuống cái kế tiếp,
     nên hệ thống KHÔNG BAO GIỜ chết vì thiếu API key.
  2. handle_chat(): biến câu lệnh + ảnh (nếu có) thành một phản hồi hoàn chỉnh
     gồm câu trả lời và danh sách ảnh kết quả.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from PIL import Image

from core import config       # import TRƯỚC provider để .env được nạp kịp
from core import retrieval
from chat import rule_based       # parser offline — chốt chặn cuối cùng
from chat.providers import azure, gemini

_ALL_PROVIDERS = {"azure": azure, "gemini": gemini}


def _build_providers() -> list[tuple[str, object]]:
    """Dựng danh sách provider khả dụng theo cấu hình CHAT_PROVIDER."""
    if not config.CHAT_PROVIDER_IS_VALID:
        print(
            f"[CẢNH BÁO] CHAT_PROVIDER={config.CHAT_PROVIDER_RAW!r} không hợp lệ -> "
            f"chatbot LLM bị TẮT. Giá trị hợp lệ: auto | azure | gemini | none "
            f"(bí danh: {', '.join(sorted(config.PROVIDER_ALIASES))})."
        )
        return []

    if config.CHAT_PROVIDER == "none":
        return []
    order = (
        config.PROVIDER_PRIORITY
        if config.CHAT_PROVIDER == "auto"
        else [config.CHAT_PROVIDER]
    )
    return [
        (name, _ALL_PROVIDERS[name])
        for name in order
        if name in _ALL_PROVIDERS and _ALL_PROVIDERS[name].is_enabled()
    ]


PROVIDERS = _build_providers()
PROVIDER_CHAIN = [name for name, _ in PROVIDERS] + ["rule-based"]
print(f">>> Chatbot: {' -> '.join(PROVIDER_CHAIN)}")


# ===========================================================
#  Hiểu câu lệnh
# ===========================================================
def understand(message: str, has_image: bool) -> dict:
    """Câu lệnh -> {intent, k, class_filter, clip_query}, thử lần lượt từng provider."""
    plan = None
    for name, provider in PROVIDERS:
        try:
            plan = provider.parse_message(message, has_image, max_k=config.MAX_K)
            break
        except Exception as e:
            print(f"[{name}] parse lỗi -> thử provider kế tiếp: {e}")

    if plan is None:
        plan = rule_based.parse_message(message, has_image, max_k=config.MAX_K)

    # Không có ảnh mà intent='search' thì không tìm theo ảnh được -> hạ xuống need_image
    # (nhánh CLIP phía sau vẫn có thể cứu bằng cách tìm theo mô tả)
    if not has_image and plan["intent"] == "search":
        plan["intent"] = "need_image"
    return plan


def smart_reply(message: str) -> Optional[str]:
    """Trả lời trò chuyện tự nhiên bằng LLM. None nếu không provider nào dùng được."""
    for name, provider in PROVIDERS:
        try:
            return provider.smart_reply(message)
        except Exception as e:
            print(f"[{name}] smart_reply lỗi: {e}")
    return None


# ===========================================================
#  Soạn câu trả lời
# ===========================================================
GREETING_FALLBACK = (
    "Xin chào! Mình là trợ lý tìm ảnh 🤖. Hãy tải lên một ảnh và nói ví dụ "
    '"Tìm cho tôi 10 ảnh giống ảnh này". Hoặc chỉ cần gõ mô tả, ví dụ '
    '"cho tôi xem 10 ảnh con mèo".'
)
NEED_IMAGE_FALLBACK = (
    "Bạn hãy đính kèm một ảnh để mình tìm ảnh giống, hoặc gõ tên một lớp "
    "(chó, mèo, máy bay, ô tô…) để mình lấy ảnh lớp đó nhé. 📎"
)


def _image_search_reply(class_filter: Optional[int], result_labels: list[int]) -> str:
    """Câu trả lời cho trường hợp tìm bằng ảnh, kèm đoán lớp của ảnh truy vấn."""
    n = len(result_labels)
    if class_filter is not None:
        return f'Đây là {n} ảnh thuộc lớp "{config.class_name(class_filter)}" giống ảnh của bạn nhất 👇'

    reply = f"Đây là {n} ảnh giống ảnh của bạn nhất 👇"
    if result_labels:
        common, cnt = Counter(result_labels).most_common(1)[0]
        reply += (
            f' Theo mô hình, ảnh của bạn trông giống lớp "{config.class_name(common)}" '
            f"nhất ({cnt}/{n} kết quả)."
        )
    return reply


def _empty(reply: str, intent: str) -> dict:
    """Phản hồi không kèm ảnh nào (chào hỏi / nhắc tải ảnh)."""
    return {
        "reply": reply, "intent": intent, "k": 0, "results": [],
        "search_time_ms": 0.0, "class_filter": None,
    }


# ===========================================================
#  Luồng chat chính
# ===========================================================
def handle_chat(message: str, img: Optional[Image.Image] = None) -> dict:
    """
    Xử lý trọn một lượt chat -> dict khớp schema ChatResponse.

    Thứ tự ưu tiên:
      1. Có ảnh                    -> tìm ảnh tương tự (ResNet)
      2. Chỉ chữ + CLIP bật        -> tìm theo mô tả (CLIP)
      3. Chỉ chữ + nêu tên lớp     -> bốc ảnh của lớp đó trong kho (khi CLIP tắt)
      4. Chào hỏi                  -> trả lời trò chuyện
      5. Còn lại                   -> nhắc người dùng gửi ảnh / gõ rõ hơn
    """
    message = (message or "").strip()
    has_image = img is not None
    plan = understand(message, has_image)
    k, class_filter = plan["k"], plan["class_filter"]

    # --- 1. Có ảnh: tìm theo độ tương đồng ảnh ---
    if has_image:
        import base64, io

        start = time.perf_counter()
        results, result_labels = retrieval.search_by_image(img, k, class_filter)
        elapsed = (time.perf_counter() - start) * 1000

        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        return {
            "reply": _image_search_reply(class_filter, result_labels),
            "intent": "search",
            "k": k,
            "results": results,
            "search_time_ms": round(elapsed, 3),
            "class_filter": config.class_name(class_filter) if class_filter is not None else None,
            "query_image": base64.b64encode(buf.getvalue()).decode(),
        }

    # --- 2. Chỉ có chữ + CLIP bật: tìm theo mô tả ngữ nghĩa ---
    if retrieval.CLIP_ENABLED and message and plan["intent"] in ("text_search", "browse_class"):
        # LLM (nếu có) đã dịch sẵn mô tả sang tiếng Anh -> dùng bản đó cho CLIP,
        # tốt hơn từ điển Việt-Anh thủ công trong clip_model.py
        clip_q = plan.get("clip_query") or message
        retrieval.warmup_clip()   # không tính thời gian tải model vào search_time_ms
        start = time.perf_counter()
        results, _ = retrieval.search_by_text(clip_q, k, class_filter)
        elapsed = (time.perf_counter() - start) * 1000

        return {
            "reply": f'Mình tìm theo mô tả "{message}" — đây là {len(results)} ảnh khớp nhất 👇',
            "intent": "text_search",
            "k": k,
            "results": results,
            "search_time_ms": round(elapsed, 3),
            "class_filter": config.class_name(class_filter) if class_filter is not None else None,
            "query_text": message,
        }

    # --- 3. Nêu tên lớp nhưng CLIP tắt: bốc ngẫu nhiên ảnh lớp đó trong kho ---
    if plan["intent"] == "browse_class" and class_filter is not None:
        results = retrieval.sample_class(class_filter, k)
        return {
            "reply": (
                f'Bạn chưa gửi ảnh, nhưng mình hiểu bạn muốn xem lớp '
                f'"{config.class_name(class_filter)}". Đây là {len(results)} ảnh thuộc lớp này 👇'
            ),
            "intent": "browse_class",
            "k": len(results),
            "results": results,
            "search_time_ms": 0.0,
            "class_filter": config.class_name(class_filter),
        }

    # --- 4. Chào hỏi / hỏi năng lực ---
    if plan["intent"] == "greeting":
        reply = GREETING_FALLBACK
        if PROVIDERS and message:
            reply = smart_reply(message) or GREETING_FALLBACK
        return _empty(reply, "greeting")

    # --- 5. Chưa đủ thông tin ---
    return _empty(NEED_IMAGE_FALLBACK, "need_image")
