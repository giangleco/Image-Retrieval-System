// static/js/app.js — thanh trạng thái, gallery ảnh mẫu, và khu hiển thị KẾT QUẢ.
//
// Khác bản cũ: mọi kết quả (dù đến từ chat hay từ click ảnh mẫu) đều đổ vào CÙNG
// một khu ở cột phải qua Results.render(). Trước đây chat tự vẽ lưới ảnh riêng
// bên trong bong bóng chat, nên có hai đoạn code vẽ lưới gần giống hệt nhau.

const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : s;
  return d.innerHTML;
}

/* ===================== KHU KẾT QUẢ (dùng chung) ===================== */
const Results = {
  loading(on) {
    $('resultsLoading').hidden = !on;
    if (on) { $('resultsEmpty').hidden = true; $('resultsBody').hidden = true; }
  },

  error(msg) {
    this.loading(false);
    $('resultsBody').hidden = true;
    const box = $('resultsEmpty');
    box.hidden = false;
    box.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i>
                     <p>Không tìm được</p><span>${escapeHtml(msg)}</span>`;
  },

  render(data) {
    this.loading(false);
    $('resultsEmpty').hidden = true;
    $('resultsBody').hidden = false;

    // --- dòng mô tả ngắn trên tiêu đề ---
    const bits = [`${data.results.length} ảnh`];
    if (data.class_filter) bits.push(`lớp "${data.class_filter}"`);
    if (data.search_time_ms != null) bits.push(`${data.search_time_ms} ms`);
    $('resultMeta').textContent = bits.join(' · ');

    // --- khối truy vấn: ảnh hoặc câu mô tả ---
    const img = $('queryImg');
    if (data.query_image) {
      img.hidden = false;
      img.src = `data:image/jpeg;base64,${data.query_image}`;
      $('queryKind').textContent = 'Ảnh truy vấn';
      $('queryValue').innerHTML = data.query_label
        ? `Nhãn<span class="tag">${escapeHtml(data.query_label)}</span>`
        : 'Ảnh tải lên <span style="color:var(--text-3);font-weight:400">(không có nhãn)</span>';
    } else {
      img.hidden = true;
      img.removeAttribute('src');
      $('queryKind').textContent = 'Mô tả truy vấn';
      $('queryValue').textContent = data.query_text || '—';
    }

    // --- chỉ số đánh giá (chỉ có khi truy vấn mang nhãn) ---
    const mBox = $('metrics');
    if (data.metrics) {
      const m = data.metrics;
      // Số nguyên (1, 0) hiện thành "1.00" cho đồng bộ với "0.0017" bên cạnh
      const fmt = (v) => (Number.isInteger(v) ? v.toFixed(2) : String(v));
      mBox.hidden = false;
      mBox.innerHTML = [
        [fmt(m.precision), `Precision@${m.k}`],
        [fmt(m.recall), `Recall@${m.k}`],
        [fmt(m.ap), 'Average Precision'],
        [`${m.num_hits}/${m.k}`, 'Đúng lớp'],
      ].map(([v, name]) =>
        `<div class="metric"><div class="metric__val">${v}</div>
         <div class="metric__name">${name}</div></div>`).join('');
    } else {
      mBox.hidden = true;
      mBox.innerHTML = '';
    }

    // --- lưới ảnh kết quả ---
    $('resultGrid').innerHTML = data.results.map((it) => {
      const match = data.query_label && it.label === data.query_label ? ' is-match' : '';
      const sim = it.similarity != null ? `<span class="card__sim">${it.similarity}%</span>` : '';
      return `<article class="card">
          <span class="card__rank">${it.rank}</span>
          <img class="card__img" src="data:image/jpeg;base64,${it.image}"
               alt="${escapeHtml(it.label)}" title="#${it.index} · ${escapeHtml(it.label)}">
          <div class="card__meta">
            <span class="card__label${match}">${escapeHtml(it.label)}</span>${sim}
          </div>
        </article>`;
    }).join('');
  },
};

/* ===================== THANH TRẠNG THÁI ===================== */
async function initStatus() {
  const box = $('status');
  try {
    const h = await Api.health();
    const chips = [
      ['chip--ok', `${h.num_images.toLocaleString('vi-VN')} ảnh`],
      h.clip_enabled ? ['chip--ok', 'tìm bằng chữ'] : ['chip--off', 'tìm bằng chữ: tắt'],
      ['chip', `chatbot: ${h.chat_providers[0]}`],
    ];
    box.innerHTML = chips
      .map(([cls, text]) => `<span class="chip ${cls}">${escapeHtml(text)}</span>`)
      .join('');
  } catch (e) {
    box.innerHTML = `<span class="chip chip--err">mất kết nối API</span>`;
    console.error(e);
  }
}

/* ===================== GALLERY ẢNH MẪU ===================== */
async function loadGallery() {
  const gallery = $('gallery');
  try {
    const data = await Api.images(300);
    gallery.innerHTML = '';
    data.items.forEach((item) => {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${item.image}`;
      img.loading = 'lazy';
      img.alt = item.label;
      img.title = `#${item.index} · ${item.label}`;
      img.addEventListener('click', () => searchByIndex(item.index, img));
      gallery.appendChild(img);
    });
  } catch (e) {
    gallery.innerHTML = `<p class="gallery__error">Không tải được ảnh: ${escapeHtml(e.message)}</p>`;
  }
}

async function searchByIndex(idx, imgEl) {
  document.querySelectorAll('.gallery img.is-active').forEach((el) => el.classList.remove('is-active'));
  if (imgEl) imgEl.classList.add('is-active');

  Results.loading(true);
  try {
    Results.render(await Api.similar(idx, { k: 10 }));
  } catch (e) {
    Results.error(e.message);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initStatus();
  loadGallery();
});
