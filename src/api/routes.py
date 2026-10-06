"""
Tầng API (FastAPI) — CHỈ lo nhận request, kiểm tra tham số và trả JSON.
Toàn bộ phần tính toán nằm ở core/retrieval.py và chat/service.py.

Tài liệu tương tác tự sinh:  http://localhost:5000/docs
"""
from __future__ import annotations

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from core import config      # import đầu tiên: nạp .env trước mọi thứ khác
from core import retrieval
from api import schemas
from chat import service as chat_service

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Nạp sẵn model CLIP TRƯỚC khi server nhận request.

    Không có bước này, model chỉ được nạp ở truy vấn tìm-bằng-chữ ĐẦU TIÊN:
    người dùng gõ câu đầu rồi ngồi đợi, tưởng server treo. Nạp ở đây thì thời
    gian chờ dồn vào lúc khởi động — nhìn thấy rõ trong log.

    Lỗi ở bước này KHÔNG làm server chết: chỉ cảnh báo rồi quay về nạp lazy như cũ
    (ví dụ máy không có mạng mà cũng chưa có cache model).
    """
    if retrieval.CLIP_ENABLED:
        start = time.perf_counter()
        print(">>> Đang nạp model CLIP ...")
        try:
            retrieval.warmup_clip()
            print(f">>> CLIP sẵn sàng ({time.perf_counter() - start:.1f}s)")
        except Exception as e:
            print(f"[CẢNH BÁO] Không nạp sẵn được CLIP ({e}). "
                  f"Sẽ thử lại ở truy vấn tìm-bằng-chữ đầu tiên.")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="CIFAR-10 Image Retrieval API",
    version="2.0.0",
    description=(
        "REST API truy xuất ảnh tương tự trên CIFAR-10.\n\n"
        "- **Tìm bằng ảnh**: đặc trưng ResNet-18 + FAISS (cosine, exact search)\n"
        "- **Tìm bằng văn bản**: CLIP nhúng ảnh & chữ chung không gian\n"
        "- **Chat**: trợ lý hiểu ngôn ngữ tự nhiên (Azure OpenAI → Gemini → rule-based)"
    ),
    # Tắt 2 trang tài liệu mặc định để tự dựng lại bên dưới với CDN ĐÃ GHIM phiên bản.
    # Lý do: FastAPI mặc định trỏ ReDoc vào tag trôi nổi "redoc@next", tag này đã bị gỡ
    # khỏi CDN (404) khiến /redoc hiện ra trang TRẮNG dù route vẫn trả về 200.
    docs_url=None,
    redoc_url=None,
)

# Ghim phiên bản cụ thể => không bị vỡ khi CDN đổi/gỡ tag
SWAGGER_JS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.18.2/swagger-ui-bundle.js"
SWAGGER_CSS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.18.2/swagger-ui.css"
REDOC_JS = "https://cdn.jsdelivr.net/npm/redoc@2.5.0/bundles/redoc.standalone.js"


@app.get("/docs", include_in_schema=False)
def swagger_ui():
    """Tài liệu API tương tác (Swagger UI) — bấm 'Try it out' để gọi thử ngay."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} — Swagger UI",
        swagger_js_url=SWAGGER_JS,
        swagger_css_url=SWAGGER_CSS,
    )


@app.get("/redoc", include_in_schema=False)
def redoc():
    """Tài liệu API dạng đọc (ReDoc)."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} — ReDoc",
        redoc_js_url=REDOC_JS,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================
#  Tiện ích dùng chung
# ===========================================================
def _read_image(file: UploadFile) -> Image.Image:
    """UploadFile -> ảnh PIL RGB. Trả lỗi 400 nếu không phải ảnh hợp lệ."""
    try:
        return Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="File tải lên không phải ảnh hợp lệ.")


def _to_b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _validate_class_filter(cf: Optional[int]) -> Optional[int]:
    if cf is None:
        return None
    if not 0 <= cf < config.NUM_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"class_filter phải nằm trong 0..{config.NUM_CLASSES - 1} (hoặc bỏ trống).",
        )
    return cf


def _log(tag: str, detail: str, elapsed_ms: float, metrics: Optional[dict] = None) -> None:
    """In báo cáo tìm kiếm ra terminal (giữ nguyên thói quen của bản Flask cũ)."""
    line = "=" * 70
    print(f"\n{line}\n   [{tag}] {detail}\n   → Thời gian tìm kiếm: {elapsed_ms:.3f} ms")
    if metrics:
        print(
            f"   → Precision@{metrics['k']}: {metrics['precision']} | "
            f"Recall@{metrics['k']}: {metrics['recall']} | AP: {metrics['ap']}"
        )
    print(f"{line}\n")


# ===========================================================
#  Metadata
# ===========================================================
@app.get("/api/health", response_model=schemas.HealthResponse, tags=["metadata"],
         summary="Trạng thái hệ thống")
def health():
    """Kiểm tra server và xem những tính năng nào đang bật."""
    return {
        "status": "ok",
        "num_images": retrieval.NUM_IMAGES,
        "feature_dim": retrieval.FEATURE_DIM,
        "clip_enabled": retrieval.CLIP_ENABLED,
        "chat_providers": chat_service.PROVIDER_CHAIN,
        "max_k": config.MAX_K,
    }


@app.get("/api/classes", response_model=list[schemas.ClassInfo], tags=["metadata"],
         summary="Danh sách 10 lớp CIFAR-10")
def classes():
    return retrieval.class_info()


@app.get("/api/images", response_model=schemas.ImageListResponse, tags=["metadata"],
         summary="Lấy ảnh trong kho (phân trang)")
def images(
    offset: int = Query(0, ge=0, description="Bỏ qua bao nhiêu ảnh đầu"),
    limit: int = Query(config.GALLERY_DEFAULT_LIMIT, ge=1, le=config.GALLERY_MAX_LIMIT,
                       description="Số ảnh muốn lấy"),
):
    """Trả về một trang ảnh để dựng gallery ở phía frontend."""
    return {
        "total": retrieval.NUM_IMAGES,
        "offset": offset,
        "limit": limit,
        "items": retrieval.list_images(offset, limit),
    }


# ===========================================================
#  Tìm kiếm
# ===========================================================
@app.get("/api/images/{idx}/similar", response_model=schemas.SearchResponse, tags=["search"],
         summary="Tìm ảnh tương tự một ảnh CÓ SẴN trong kho")
def similar_to_index(
    idx: int = Path(..., ge=0, description="Chỉ số ảnh trong kho"),
    k: int = Query(config.DEFAULT_K, ge=1, le=config.MAX_K),
    class_filter: Optional[int] = Query(None, description="Chỉ tìm trong 1 lớp (0-9)"),
):
    """
    Vì ảnh truy vấn có nhãn nên response kèm **metrics** (Precision@k, Recall@k, AP).
    Ảnh truy vấn được tự loại khỏi kết quả.
    """
    if idx >= retrieval.NUM_IMAGES:
        raise HTTPException(
            status_code=404,
            detail=f"Không có ảnh idx={idx}. Kho chỉ có {retrieval.NUM_IMAGES} ảnh (0..{retrieval.NUM_IMAGES - 1}).",
        )
    class_filter = _validate_class_filter(class_filter)

    start = time.perf_counter()
    results, result_labels = retrieval.search_by_index(idx, k, class_filter)
    elapsed = (time.perf_counter() - start) * 1000

    query_label = int(retrieval.labels[idx])
    # Metric chỉ có ý nghĩa khi tìm trên TOÀN kho (lọc lớp làm precision luôn = 1)
    metrics = (
        retrieval.compute_metrics(result_labels, query_label, k)
        if class_filter is None and results else None
    )

    _log("SEARCH-BY-INDEX", f"idx={idx} (k={k})", elapsed, metrics)
    return {
        "k": k,
        "results": results,
        "search_time_ms": round(elapsed, 3),
        "class_filter": config.class_name(class_filter) if class_filter is not None else None,
        "query_index": idx,
        "query_image": retrieval.image_b64_list[idx],
        "query_label": config.class_name(query_label),
        "metrics": metrics,
    }


@app.post("/api/search/image", response_model=schemas.SearchResponse, tags=["search"],
          summary="Tìm ảnh tương tự một ảnh TẢI LÊN")
def search_image(
    file: UploadFile = File(..., description="Ảnh cần tìm (jpg/png/…)"),
    k: int = Form(config.DEFAULT_K),
    class_filter: Optional[int] = Form(None),
):
    """Ảnh upload không có nhãn nên response **không** kèm metrics."""
    k = max(1, min(k, config.MAX_K))
    class_filter = _validate_class_filter(class_filter)

    img = _read_image(file)
    start = time.perf_counter()
    results, _ = retrieval.search_by_image(img, k, class_filter)
    elapsed = (time.perf_counter() - start) * 1000

    _log("SEARCH-BY-UPLOAD", f"{file.filename!r} (k={k})", elapsed)
    return {
        "k": k,
        "results": results,
        "search_time_ms": round(elapsed, 3),
        "class_filter": config.class_name(class_filter) if class_filter is not None else None,
        "query_image": _to_b64_jpeg(img),
        "query_label": None,
    }


@app.post("/api/search/text", response_model=schemas.SearchResponse, tags=["search"],
          summary="Tìm ảnh bằng MÔ TẢ văn bản (CLIP)")
def search_text(body: schemas.TextSearchRequest):
    """
    Hỗ trợ cả tiếng Việt (được dịch thô sang tiếng Anh trước khi đưa vào CLIP).
    Cần `features/features_clip.npy`, nếu thiếu sẽ trả **503**.
    """
    if not retrieval.CLIP_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="CLIP chưa bật. Hãy tạo features/features_clip.npy "
                   "(uv run python src/scripts/extract_clip.py hoặc trích trên Kaggle).",
        )

    retrieval.warmup_clip()   # tải model TRƯỚC khi bấm giờ, để không tính vào search_time_ms
    start = time.perf_counter()
    results, _ = retrieval.search_by_text(body.q, body.k, body.class_filter)
    elapsed = (time.perf_counter() - start) * 1000

    _log("SEARCH-BY-TEXT", f"q={body.q!r} (k={body.k})", elapsed)
    return {
        "k": body.k,
        "results": results,
        "search_time_ms": round(elapsed, 3),
        "class_filter": config.class_name(body.class_filter) if body.class_filter is not None else None,
        "query_text": body.q,
    }


# ===========================================================
#  Chat
# ===========================================================
@app.post("/api/chat", response_model=schemas.ChatResponse, tags=["chat"],
          summary="Trợ lý tìm ảnh (nhận cả ảnh lẫn mô tả)")
def chat(
    message: str = Form("", description="Câu của người dùng"),
    file: Optional[UploadFile] = File(None, description="Ảnh đính kèm (tuỳ chọn)"),
):
    """
    Một endpoint duy nhất cho khung chat: tự hiểu ý định, số lượng K và lớp cần lọc,
    rồi chọn cách tìm phù hợp (theo ảnh / theo mô tả / duyệt lớp / chỉ trò chuyện).
    """
    img = _read_image(file) if file and file.filename else None
    if img is None and not message.strip():
        raise HTTPException(status_code=400, detail="Cần ít nhất một câu mô tả hoặc một ảnh.")

    data = chat_service.handle_chat(message, img)
    print(f"\n{'='*70}\n   [CHAT] {message!r} -> intent={data['intent']}, k={data['k']}\n{'='*70}\n")
    return data


# ===========================================================
#  Frontend tĩnh (gọi API bằng fetch) — đặt CUỐI để không che các route /api
# ===========================================================
class RevalidatingStaticFiles(StaticFiles):
    """StaticFiles nhưng BUỘC trình duyệt hỏi lại server trước khi dùng bản cache.

    Mặc định StaticFiles chỉ gửi ETag + Last-Modified, KHÔNG gửi Cache-Control.
    Thiếu Cache-Control, trình duyệt áp dụng "heuristic caching": tự quyết định
    dùng lại CSS/JS cũ trong một khoảng thời gian mà không hỏi server. Hậu quả:
    sửa giao diện xong, mở trang vẫn thấy bản cũ cho tới khi hard-refresh.

    "no-cache" KHÔNG phải là cấm lưu — trình duyệt vẫn lưu, chỉ là luôn hỏi lại.
    Nhờ ETag, lần hỏi đó thường nhận 304 Not Modified (không kèm nội dung) nên
    gần như không tốn băng thông.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


if config.STATIC_DIR.exists():
    app.mount("/static", RevalidatingStaticFiles(directory=config.STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(
            config.STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-cache"},
        )
