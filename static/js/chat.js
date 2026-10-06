// static/js/chat.js — khung chat ở cột trái.
// Chat chỉ lo hội thoại; phần ẢNH KẾT QUẢ giao cho Results.render() ở cột phải
// (xem app.js), nên bong bóng chat luôn gọn, không bị lưới ảnh chen vào.
(function () {
  const chatLog = $('chatLog');
  const chatText = $('chatText');
  const chatSend = $('chatSend');
  const fileInput = $('chatFileInput');
  const filePreview = $('filePreview');
  const filePreviewImg = $('filePreviewImg');
  const fileName = $('chatFileName');

  let attached = null;
  let busy = false;

  /* --- đính kèm ảnh --- */
  fileInput.addEventListener('change', () => setAttached(fileInput.files[0] || null));
  $('fileClear').addEventListener('click', () => setAttached(null));

  function setAttached(file) {
    attached = file;
    filePreview.hidden = !file;
    if (!file) { fileInput.value = ''; return; }
    fileName.textContent = file.name;
    const reader = new FileReader();
    reader.onload = (e) => { filePreviewImg.src = e.target.result; };
    reader.readAsDataURL(file);
  }

  /* --- câu gợi ý --- */
  document.querySelectorAll('.suggestion').forEach((btn) => {
    btn.addEventListener('click', () => { chatText.value = btn.textContent; send(); });
  });

  /* --- gửi --- */
  chatText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); send(); }
  });
  chatSend.addEventListener('click', send);

  async function send() {
    const msg = chatText.value.trim();
    if (busy || (!msg && !attached)) return;

    const intro = $('chatIntro');
    if (intro) intro.remove();

    addUser(msg, attached);
    const file = attached;
    chatText.value = '';
    setAttached(null);

    const thinking = addBot('<i class="fa-solid fa-circle-notch fa-spin"></i> Đang xử lý…');
    setBusy(true);
    if (file || msg) Results.loading(true);

    try {
      const data = await Api.chat(msg, file);
      thinking.remove();
      addBot(escapeHtml(data.reply));
      if (data.results && data.results.length) Results.render(data);
      else Results.loading(false);
    } catch (err) {
      thinking.remove();
      addBot(`⚠️ ${escapeHtml(err.message)}`, true);
      Results.error(err.message);
    } finally {
      setBusy(false);
      chatText.focus();
    }
  }

  function setBusy(on) {
    busy = on;
    chatSend.disabled = on;
  }

  /* --- bong bóng --- */
  function addUser(text, file) {
    const row = document.createElement('div');
    row.className = 'msg msg--user';
    const bubble = document.createElement('div');
    bubble.className = 'msg__bubble';
    if (text) bubble.appendChild(document.createTextNode(text));
    if (file) {
      const img = document.createElement('img');
      img.className = 'msg__thumb';
      const reader = new FileReader();
      reader.onload = (e) => { img.src = e.target.result; };
      reader.readAsDataURL(file);
      bubble.appendChild(img);
    }
    row.appendChild(bubble);
    chatLog.appendChild(row);
    scrollDown();
  }

  function addBot(html, isError) {
    const row = document.createElement('div');
    row.className = 'msg msg--bot';
    row.innerHTML = `<div class="msg__bubble${isError ? ' is-error' : ''}">${html}</div>`;
    chatLog.appendChild(row);
    scrollDown();
    return row;
  }

  function scrollDown() { chatLog.scrollTop = chatLog.scrollHeight; }
})();
