// static/js/app.js
// Xử lý CLICK ẢNH MẪU (gallery) + hiển thị kết quả vào khu vực #results.
// Việc upload ảnh / tìm bằng mô tả nay nằm trong khung chat (chat.js).

// Số kết quả mặc định cho click ảnh mẫu
function getOptions() {
    return { k: '10', class_filter: 'all' };
}

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

    const queryImg = document.getElementById('queryImg');
    const queryHeading = document.getElementById('queryHeading');
    const queryLabel = document.getElementById('queryLabel');

    // Tìm bằng văn bản: không có ảnh truy vấn -> hiện mô tả thay cho ảnh
    if (!data.query_image) {
        queryImg.classList.add('hidden');
        queryHeading.textContent = 'Mô tả truy vấn';
        queryLabel.textContent = data.query_text || '';
        return finishResults(data);
    }

    queryImg.classList.remove('hidden');
    queryHeading.textContent = 'Ảnh truy vấn';
    queryImg.src = "data:image/jpeg;base64," + data.query_image;

    // Nhãn ảnh truy vấn (nếu có)
    queryLabel.innerHTML = data.query_label
        ? `Lớp truy vấn: <strong>${data.query_label}</strong>`
        : 'Ảnh upload (không có nhãn)';

    finishResults(data);
}

// Render tiêu đề + bảng metric + lưới kết quả (dùng chung cho mọi kiểu truy vấn)
function finishResults(data) {
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
        const match = data.query_label && item.label === data.query_label ? ' match' : '';
        const sim = (item.similarity != null) ? `<span class="sim">${item.similarity}%</span>` : '';
        grid.innerHTML += `
            <div class="result-item">
                <span class="rank">${item.rank}</span>
                <img src="data:image/jpeg;base64,${item.image}">
                <div class="meta">
                    <span class="label${match}">${item.label}</span>
                    ${sim}
                </div>
            </div>`;
    });
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}
