// static/js/app.js
let selectedFile = null;

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const clearBtn = document.getElementById('clearBtn');
const searchBtn = document.getElementById('searchButton');
const kSelect = document.getElementById('kSelect');
const classSelect = document.getElementById('classSelect');

// Lấy giá trị tuỳ chọn hiện tại (số K + lọc lớp)
function getOptions() {
    return {
        k: kSelect ? kSelect.value : '10',
        class_filter: classSelect ? classSelect.value : 'all',
    };
}

// Chọn file
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            preview.classList.remove('hidden');
            uploadArea.style.display = 'none';
            searchBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

// Xóa ảnh
clearBtn.onclick = () => {
    fileInput.value = '';
    selectedFile = null;
    preview.classList.add('hidden');
    uploadArea.style.display = 'block';
    searchBtn.disabled = true;
};

// NÚT TÌM (ảnh upload) — CHẶN HOÀN TOÀN RELOAD
searchBtn.onclick = function (e) {
    e.preventDefault();
    e.stopPropagation();

    if (!selectedFile) return;

    showLoading(true);
    searchBtn.disabled = true;
    searchBtn.innerHTML = 'Đang tìm...';

    const opts = getOptions();
    const fd = new FormData();
    fd.append('file', selectedFile);
    fd.append('k', opts.k);
    fd.append('class_filter', opts.class_filter);

    fetch('/search', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            displayResults(data);
            searchBtn.disabled = false;
            searchBtn.innerHTML = 'Tìm lại với ảnh khác';
        })
        .catch(() => {
            showLoading(false);
            alert('Lỗi server!');
            searchBtn.disabled = false;
            searchBtn.innerHTML = 'Thử lại';
        });
};

// Tìm bằng ảnh mẫu trong kho (theo index)
function searchByIndex(idx) {
    const opts = getOptions();
    const params = new URLSearchParams({
        idx: idx,
        k: opts.k,
        class_filter: opts.class_filter,
    });
    showLoading(true);
    fetch(`/search?${params.toString()}`)
        .then(r => r.json())
        .then(data => {
            showLoading(false);
            displayResults(data);
        })
        .catch(() => {
            showLoading(false);
            alert('Lỗi server!');
        });
}

function displayResults(data) {
    document.getElementById('results').classList.remove('hidden');
    document.getElementById('queryImg').src = "data:image/jpeg;base64," + data.query_image;

    // Nhãn ảnh truy vấn (nếu có)
    const queryLabel = document.getElementById('queryLabel');
    queryLabel.innerHTML = data.query_label
        ? `Lớp truy vấn: <strong>${data.query_label}</strong>`
        : 'Ảnh upload (không có nhãn)';

    // Tiêu đề danh sách kết quả
    const title = document.getElementById('resultTitle');
    let titleText = `Top ${data.k} kết quả`;
    if (data.class_filter) titleText += ` — chỉ lớp "${data.class_filter}"`;
    if (data.search_time_ms != null) titleText += ` (${data.search_time_ms} ms)`;
    title.textContent = titleText;

    // Bảng chỉ số đánh giá
    const metricsBox = document.getElementById('metrics');
    if (data.metrics) {
        const m = data.metrics;
        metricsBox.classList.remove('hidden');
        metricsBox.innerHTML = `
            <div class="metric-card"><span class="metric-val">${m.precision}</span><span class="metric-name">Precision@${m.k}</span></div>
            <div class="metric-card"><span class="metric-val">${m.recall}</span><span class="metric-name">Recall@${m.k}</span></div>
            <div class="metric-card"><span class="metric-val">${m.ap}</span><span class="metric-name">Average Precision</span></div>
            <div class="metric-card"><span class="metric-val">${m.num_hits}/${m.k}</span><span class="metric-name">Đúng lớp</span></div>`;
    } else {
        metricsBox.classList.add('hidden');
        metricsBox.innerHTML = '';
    }

    // Lưới kết quả
    const grid = document.getElementById('resultGrid');
    grid.innerHTML = '';
    data.results.forEach(item => {
        // Tô màu nhãn trùng với lớp truy vấn để dễ nhìn
        const match = data.query_label && item.label === data.query_label ? ' match' : '';
        grid.innerHTML += `
            <div class="result-item">
                <span class="rank">${item.rank}</span>
                <img src="data:image/jpeg;base64,${item.image}">
                <div class="meta">
                    <span class="label${match}">${item.label}</span>
                    <span class="sim">${item.similarity}%</span>
                </div>
            </div>`;
    });
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}
