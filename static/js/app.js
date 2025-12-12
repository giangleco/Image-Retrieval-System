// static/js/app.js
let selectedFile = null;

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const clearBtn = document.getElementById('clearBtn');
const searchBtn = document.getElementById('searchButton');

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

// NÚT TÌM — CHẮN CHẶN HOÀN TOÀN RELOAD
searchBtn.onclick = function(e) {
    e.preventDefault();
    e.stopPropagation();

    if (!selectedFile) return;

    showLoading(true);
    searchBtn.disabled = true;
    searchBtn.innerHTML = 'Đang tìm...';

    const fd = new FormData();
    fd.append('file', selectedFile);

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

// Các hàm khác
function searchByIndex(idx) {
    showLoading(true);
    fetch(`/search?idx=${idx}`).then(r=>r.json()).then(data => {
        showLoading(false); displayResults(data);
    });
}

function displayResults(data) {
    document.getElementById('results').classList.remove('hidden');
    document.getElementById('queryImg').src = "data:image/jpeg;base64," + data.query_image;
    const grid = document.getElementById('resultGrid');
    grid.innerHTML = '';
    data.results.forEach(item => {
        grid.innerHTML += `
            <div class="result-item">
                <span class="rank">${item.rank}</span>
                <img src="data:image/jpeg;base64,${item.image}">
                <div class="dist">Distance: ${item.distance.toFixed(4)}</div>
            </div>`;
    });
    document.getElementById('results').scrollIntoView({behavior:'smooth'});
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}