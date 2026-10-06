"""
Khai báo kiểu dữ liệu vào/ra của API (Pydantic).

Nhờ các model này, FastAPI tự sinh tài liệu Swagger tại /docs và tự kiểm tra
tham số đầu vào (k, class_filter, idx...) trước khi vào tầng xử lý.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from core import config


# ===========================================================
#  Đầu vào
# ===========================================================
class TextSearchRequest(BaseModel):
    """Body cho POST /api/search/text — tìm ảnh bằng mô tả (CLIP)."""

    q: str = Field(..., min_length=1, description="Mô tả cần tìm, vd 'a red truck' / 'con mèo'")
    k: int = Field(config.DEFAULT_K, ge=1, le=config.MAX_K, description="Số ảnh muốn lấy")
    class_filter: Optional[int] = Field(
        None, ge=0, le=config.NUM_CLASSES - 1,
        description="Chỉ tìm trong 1 lớp (0-9); bỏ trống = tìm toàn kho",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"q": "an airplane flying in the sky", "k": 10, "class_filter": None}
        }
    }


# ===========================================================
#  Đầu ra
# ===========================================================
class ClassInfo(BaseModel):
    index: int = Field(..., description="Index lớp (0-9)")
    name: str = Field(..., description="Tên lớp tiếng Anh")
    name_vi: str = Field(..., description="Tên lớp tiếng Việt")
    count: int = Field(..., description="Số ảnh thuộc lớp này trong kho")


class ImageItem(BaseModel):
    """Một ảnh trong kho (dùng cho gallery)."""

    index: int = Field(..., description="Chỉ số ảnh trong kho — dùng lại để tìm ảnh tương tự")
    label: str
    image: str = Field(..., description="Ảnh JPEG 64x64 mã hoá base64 (chưa có tiền tố data:)")


class ImageListResponse(BaseModel):
    total: int = Field(..., description="Tổng số ảnh trong kho")
    offset: int
    limit: int
    items: List[ImageItem]


class ResultItem(BaseModel):
    """Một ảnh trong danh sách kết quả tìm kiếm."""

    rank: int = Field(..., description="Thứ hạng, bắt đầu từ 1")
    index: int = Field(..., description="Chỉ số ảnh trong kho")
    label: str
    image: str = Field(..., description="Ảnh JPEG base64")
    similarity: Optional[float] = Field(None, description="Độ tương đồng cosine, theo %")
    distance: Optional[float] = Field(None, description="1 - cosine similarity")


class Metrics(BaseModel):
    """Chỉ số chất lượng retrieval — chỉ có khi truy vấn là ảnh CÓ nhãn."""

    precision: float = Field(..., description="Precision@k")
    recall: float = Field(..., description="Recall@k")
    ap: float = Field(..., description="Average Precision")
    num_hits: int = Field(..., description="Số ảnh đúng lớp trong top-k")
    k: int


class SearchResponse(BaseModel):
    k: int
    results: List[ResultItem]
    search_time_ms: float = Field(..., description="Thời gian tìm kiếm (ms)")
    class_filter: Optional[str] = Field(None, description="Tên lớp đã lọc, null nếu tìm toàn kho")
    query_index: Optional[int] = Field(None, description="Chỉ số ảnh truy vấn (khi tìm bằng ảnh trong kho)")
    query_image: Optional[str] = Field(None, description="Ảnh truy vấn base64 (khi tìm bằng ảnh)")
    query_label: Optional[str] = Field(None, description="Nhãn ảnh truy vấn, null nếu ảnh upload")
    query_text: Optional[str] = Field(None, description="Mô tả truy vấn (khi tìm bằng văn bản)")
    metrics: Optional[Metrics] = None


class ChatResponse(SearchResponse):
    """Giống SearchResponse nhưng kèm câu trả lời của trợ lý."""

    reply: str = Field(..., description="Câu trả lời hiển thị trong khung chat")
    intent: str = Field(..., description="Ý định đã hiểu được: search/text_search/browse_class/greeting/need_image")


class HealthResponse(BaseModel):
    status: str
    num_images: int = Field(..., description="Số ảnh trong kho")
    feature_dim: int
    clip_enabled: bool = Field(..., description="Có bật tìm bằng văn bản (CLIP) hay không")
    chat_providers: List[str] = Field(..., description="Chuỗi provider chatbot theo thứ tự thử")
    max_k: int

