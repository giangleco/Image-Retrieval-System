// static/js/api.js — lớp bọc mỏng quanh REST API.
// Mọi lời gọi mạng của frontend đều đi qua đây, nên khi tách frontend sang
// domain khác chỉ cần đổi API_BASE (hoặc đặt window.API_BASE trước khi nạp file này).
const API_BASE = window.API_BASE || '';

const Api = {
    async _json(res) {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
        return data;
    },

    // GET /api/health — trạng thái server & tính năng đang bật
    health() {
        return fetch(`${API_BASE}/api/health`).then(this._json);
    },

    // GET /api/images — một trang ảnh trong kho
    images(limit = 300, offset = 0) {
        const qs = new URLSearchParams({ limit, offset });
        return fetch(`${API_BASE}/api/images?${qs}`).then(this._json);
    },

    // GET /api/images/{idx}/similar — tìm theo ảnh có sẵn trong kho (kèm metrics)
    similar(idx, { k = 10, classFilter = null } = {}) {
        const qs = new URLSearchParams({ k });
        if (classFilter !== null) qs.set('class_filter', classFilter);
        return fetch(`${API_BASE}/api/images/${idx}/similar?${qs}`).then(this._json);
    },

    // POST /api/search/image — tìm theo ảnh tải lên
    searchByImage(file, { k = 10, classFilter = null } = {}) {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('k', k);
        if (classFilter !== null) fd.append('class_filter', classFilter);
        return fetch(`${API_BASE}/api/search/image`, { method: 'POST', body: fd }).then(this._json);
    },

    // POST /api/search/text — tìm theo mô tả (CLIP)
    searchByText(q, { k = 10, classFilter = null } = {}) {
        return fetch(`${API_BASE}/api/search/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ q, k, class_filter: classFilter }),
        }).then(this._json);
    },

    // POST /api/chat — trợ lý (message + ảnh tuỳ chọn)
    chat(message, file) {
        const fd = new FormData();
        fd.append('message', message);
        if (file) fd.append('file', file);
        return fetch(`${API_BASE}/api/chat`, { method: 'POST', body: fd }).then(this._json);
    },
};
