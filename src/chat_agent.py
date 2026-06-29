"""
Agent chat RULE-BASED (offline) cho hệ thống tìm ảnh.

Nhiệm vụ: đọc câu lệnh ngôn ngữ tự nhiên (tiếng Việt/Anh) kèm ảnh, rồi rút ra:
  - intent       : 'search' | 'browse_class' | 'greeting' | 'need_image'
  - k            : số lượng ảnh cần trả về
  - class_filter : index lớp CIFAR-10 (0-9) nếu người dùng nhắc tên con vật/đồ vật

KHÔNG cần API key, KHÔNG cần internet — chỉ dùng quy tắc + so khớp từ khoá.
Ví dụ: "Tìm cho tôi 10 hình ảnh giống hình trên" -> {intent: search, k: 10, class_filter: None}
"""
import re

DEFAULT_K = 10

# Bản đồ từ khoá (tiếng Việt + Anh) -> index lớp CIFAR-10.
# Đặt cụm dài/đặc thù lên trước để ưu tiên khớp đúng (vd "xe tải" trước "xe hơi").
CLASS_KEYWORDS = [
    (["máy bay", "phi cơ", "airplane", "aeroplane", "plane"], 0),   # airplane
    (["xe tải", "xe ben", "truck", "lorry"], 9),                    # truck (ưu tiên trước "xe")
    (["xe hơi", "ô tô", "oto", "xe ô tô", "automobile", "car"], 1), # automobile
    (["con chim", "chim", "bird"], 2),                              # bird
    (["con mèo", "mèo", "meo", "cat", "kitten"], 3),                # cat
    (["hươu", "nai", "deer"], 4),                                   # deer
    (["con chó", "con cho", "chó", "dog", "puppy"], 5),             # dog ("cho" trần dễ nhầm "cho tôi")
    (["con ếch", "ếch", "ech", "frog"], 6),                         # frog
    (["con ngựa", "ngựa", "ngua", "horse"], 7),                     # horse
    (["tàu thủy", "tàu thuyền", "con tàu", "tàu", "thuyền", "ship", "boat"], 8),  # ship
]

# Số đếm bằng chữ tiếng Việt -> giá trị
VN_NUMBERS = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
    "chục": 10,
}

SEARCH_KEYWORDS = [
    "tìm", "kiếm", "search", "find", "giống", "tương tự",
    "similar", "retrieve", "tra cứu", "lookup",
]
GREETING_KEYWORDS = [
    "xin chào", "chào", "hello", "hi", "giúp", "help",
    "làm được gì", "hướng dẫn", "bạn là ai",
]


def parse_k(message, default=DEFAULT_K, max_k=50):
    """Rút số lượng ảnh K từ câu lệnh (ưu tiên chữ số, sau đó số đếm tiếng Việt)."""
    m = re.search(r"\b(\d{1,3})\b", message or "")
    if m:
        return max(1, min(int(m.group(1)), max_k))
    low = (message or "").lower()
    for word, val in VN_NUMBERS.items():
        if word in low:
            return max(1, min(val, max_k))
    return default


def parse_class(message):
    """Rút lớp cần lọc (0-9) nếu người dùng có nhắc tên; None nếu không."""
    low = (message or "").lower()
    for keywords, cls in CLASS_KEYWORDS:
        for kw in keywords:
            if kw in low:
                return cls
    return None


def parse_message(message, has_image, max_k=50):
    """
    Phân tích câu lệnh -> dict {intent, k, class_filter}.
      - Có ảnh                       => search (tìm theo độ tương đồng).
      - Không ảnh + có nhắc tên lớp  => browse_class (duyệt ảnh của lớp đó trong kho).
      - Không ảnh + lời chào/hỏi     => greeting.
      - Không ảnh + ý tìm nhưng không rõ lớp => need_image (nhắc người dùng tải ảnh).
    """
    message = (message or "").strip()
    low = message.lower()
    k = parse_k(message, max_k=max_k)
    cls = parse_class(message)

    # Có ảnh => mặc định là yêu cầu tìm kiếm theo độ tương đồng
    if has_image:
        return {"intent": "search", "k": k, "class_filter": cls}

    # Không có ảnh nhưng nhắc tên một lớp => duyệt ảnh của lớp đó (không cần ảnh)
    if cls is not None:
        return {"intent": "browse_class", "k": k, "class_filter": cls}

    # Không ảnh, không rõ lớp
    if not message or any(kw in low for kw in GREETING_KEYWORDS):
        return {"intent": "greeting", "k": DEFAULT_K, "class_filter": None}

    return {"intent": "need_image", "k": k, "class_filter": None}
