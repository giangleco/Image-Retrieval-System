// static/js/chat.js — Trợ lý tìm ảnh (giao tiếp với route /chat)
(function () {
    const chatLog = document.getElementById('chatLog');
    const chatText = document.getElementById('chatText');
    const chatSend = document.getElementById('chatSend');
    const chatFileInput = document.getElementById('chatFileInput');
    const chatFileName = document.getElementById('chatFileName');

    let attachedFile = null;

    // Đính kèm ảnh
    chatFileInput.addEventListener('change', () => {
        attachedFile = chatFileInput.files[0] || null;
        chatFileName.textContent = attachedFile ? attachedFile.name : '';
    });

    // Gửi bằng Enter
    chatText.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); sendMessage(); }
    });
    chatSend.addEventListener('click', sendMessage);

    function sendMessage() {
        const msg = chatText.value.trim();
        if (!msg && !attachedFile) return;

        appendUser(msg, attachedFile);

        const fd = new FormData();
        fd.append('message', msg);
        if (attachedFile) fd.append('file', attachedFile);

        const thinking = appendBot('<i class="fas fa-spinner fa-spin"></i> Đang xử lý...');

        fetch('/chat', { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => { thinking.remove(); renderBotResult(data); })
            .catch(() => { thinking.remove(); appendBot('⚠️ Lỗi server, vui lòng thử lại.'); });

        // reset input
        chatText.value = '';
        attachedFile = null;
        chatFileInput.value = '';
        chatFileName.textContent = '';
    }

    // --- Bong bóng người dùng (kèm ảnh thu nhỏ nếu có) ---
    function appendUser(text, file) {
        const row = document.createElement('div');
        row.className = 'chat-msg user';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        if (text) bubble.appendChild(document.createTextNode(text));
        if (file) {
            const img = document.createElement('img');
            img.className = 'chat-thumb';
            const reader = new FileReader();
            reader.onload = e => { img.src = e.target.result; };
            reader.readAsDataURL(file);
            bubble.appendChild(img);
        }
        row.appendChild(bubble);
        chatLog.appendChild(row);
        scrollDown();
    }

    // --- Bong bóng bot (HTML) ---
    function appendBot(html) {
        const row = document.createElement('div');
        row.className = 'chat-msg bot';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = html;
        row.appendChild(bubble);
        chatLog.appendChild(row);
        scrollDown();
        return row;
    }

    // --- Bot trả lời kèm lưới kết quả ---
    function renderBotResult(data) {
        let html = `<div class="bot-reply">${escapeHtml(data.reply || '')}</div>`;

        if (data.results && data.results.length) {
            if (data.search_time_ms != null) {
                html += `<div class="bot-meta">⏱️ ${data.search_time_ms} ms` +
                        (data.class_filter ? ` · lọc lớp "${data.class_filter}"` : '') +
                        `</div>`;
            }
            html += '<div class="chat-result-grid">';
            data.results.forEach(item => {
                const sim = (item.similarity != null)
                    ? `<span class="sim">${item.similarity}%</span>` : '';
                html += `
                    <div class="chat-result-item">
                        <span class="rank">${item.rank}</span>
                        <img src="data:image/jpeg;base64,${item.image}">
                        <div class="meta">
                            <span class="label">${item.label}</span>
                            ${sim}
                        </div>
                    </div>`;
            });
            html += '</div>';
        }
        appendBot(html);
    }

    function scrollDown() { chatLog.scrollTop = chatLog.scrollHeight; }

    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }
})();
