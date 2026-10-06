"""
Phần DÙNG CHUNG cho mọi nhà cung cấp LLM (Gemini, Azure OpenAI, ...).

Gồm: danh sách lớp, prompt phân tích ý định, prompt trả lời trò chuyện, và hàm
chuẩn hoá/kiểm tra kết quả JSON -> {intent, k, class_filter, clip_query}.

Nhờ tách ở đây, mỗi provider chỉ cần lo phần "gọi API" của riêng nó.
"""

import os

from core import config


def clean_env(name):
    """Đọc biến môi trường, COI placeholder ('your_...', 'https://YOUR-...') là CHƯA điền -> None.
    Tránh gọi API với giá trị mẫu (gây lỗi 'API key not valid')."""
    v = (os.environ.get(name) or "").strip()
    if not v:
        return None
    low = v.lower()
    if low.startswith("your_") or low.startswith("https://your-") or low == "your_deployment_name":
        return None
    return v


# Danh sách lớp lấy thẳng từ core.config — KHÔNG khai báo lại, để sửa một nơi
# là mọi tầng (API, chatbot, prompt) cùng thấy.
CIFAR10_CLASSES = config.CIFAR10_CLASSES
CIFAR10_VN = config.CIFAR10_CLASSES_VN

VALID_INTENTS = ("search", "text_search", "browse_class", "greeting", "need_image")


def _class_list_text():
    return "\n".join(
        f"  {i} = {en} ({vi})" for i, (en, vi) in enumerate(zip(CIFAR10_CLASSES, CIFAR10_VN))
    )


def parse_prompt(message, has_image, max_k=50):
    """Prompt yêu cầu LLM phân tích câu lệnh -> JSON {intent, k, class_filter, clip_query}."""
    return f"""Bạn là bộ phân tích ý định cho một hệ thống TÌM ẢNH (CIFAR-10).
Người dùng {"CÓ" if has_image else "KHÔNG"} đính kèm ảnh.
Câu của người dùng: "{(message or '').strip()}"

Danh sách 10 lớp ảnh (index = tên tiếng Anh (tiếng Việt)):
{_class_list_text()}

Hãy trả về DUY NHẤT một JSON với 4 khoá:
- "intent": một trong ["search", "text_search", "browse_class", "greeting", "need_image"]
    * "search": CÓ ảnh và muốn tìm ảnh GIỐNG ảnh đó.
    * "text_search": KHÔNG có ảnh, người dùng MÔ TẢ nội dung muốn tìm (vd "máy bay đang bay", "chú mèo cam đang nằm").
    * "browse_class": KHÔNG có ảnh, chỉ nêu tên đúng một lớp muốn xem (vd "cho tôi xem ảnh con mèo").
    * "greeting": chào hỏi, cảm ơn, hỏi bot làm được gì, hỏi hướng dẫn, nói chuyện phiếm. TUYỆT ĐỐI không phải yêu cầu tìm ảnh.
    * "need_image": có ý muốn tìm nhưng chưa gửi ảnh và mô tả quá mơ hồ.
- "k": số ảnh muốn lấy (số nguyên 1..{max_k}). Nếu không nói thì để 10.
- "class_filter": index lớp (0-9) nếu mô tả rõ ràng thuộc đúng một lớp trong 10 lớp trên; nếu không chắc thì null.
- "clip_query": nếu intent là "text_search", hãy dịch/viết lại mô tả thành MỘT cụm TIẾNG ANH ngắn gọn, tự nhiên để tìm ảnh (vd "an orange cat lying down", "an airplane flying in the sky"); nếu không phải text_search thì để null.

Lưu ý: câu hỏi kiểu "bạn làm được gì", "bạn là ai", "giúp tôi với" -> LUÔN là "greeting", KHÔNG tìm ảnh.
Chỉ in JSON, không giải thích. Ví dụ: {{"intent":"text_search","k":5,"class_filter":3,"clip_query":"a cat sitting"}}"""


def reply_prompt(message):
    """Prompt yêu cầu LLM trả lời trò chuyện tự nhiên (dùng cho greeting)."""
    return f"""Bạn là trợ lý thân thiện của một website TÌM ẢNH tương tự trên tập CIFAR-10
(10 lớp: máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa, tàu thuyền, xe tải).
Người dùng có thể: (1) tải ảnh lên để tìm ảnh giống, (2) gõ tên một lớp để xem ảnh lớp đó,
(3) gõ mô tả bằng lời để tìm ảnh theo nội dung.

Người dùng nói: "{(message or '').strip()}"

Hãy trả lời NGẮN GỌN (1-3 câu), bằng tiếng Việt, thân thiện, có thể dùng emoji.
Nếu phù hợp, gợi ý họ tải ảnh hoặc gõ tên một lớp. KHÔNG bịa ra kết quả tìm kiếm."""


def normalize_plan(data, has_image, max_k=50):
    """
    Chuẩn hoá & kiểm tra JSON LLM trả về -> {intent, k, class_filter, clip_query}.
    Raise ValueError nếu dữ liệu không hợp lệ (để phía gọi fallback).
    """
    if not isinstance(data, dict):
        raise ValueError(f"kết quả không phải JSON object: {type(data)}")

    intent = data.get("intent")
    if intent not in VALID_INTENTS:
        raise ValueError(f"intent không hợp lệ: {intent!r}")

    k = data.get("k", 10)
    try:
        k = max(1, min(int(k), max_k))
    except (TypeError, ValueError):
        k = 10

    cls = data.get("class_filter", None)
    if cls is not None:
        try:
            cls = int(cls)
            if not (0 <= cls <= 9):
                cls = None
        except (TypeError, ValueError):
            cls = None

    clip_query = data.get("clip_query") or None
    if clip_query is not None:
        clip_query = str(clip_query).strip() or None

    # Có ảnh => luôn là tìm theo độ tương đồng (đồng nhất với luồng cũ)
    if has_image:
        intent = "search"

    return {"intent": intent, "k": k, "class_filter": cls, "clip_query": clip_query}
