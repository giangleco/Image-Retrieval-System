
# Báo cáo đánh giá

## 2. Xác định vấn đề và chiến lược giải quyết

### Vấn đề cần giải quyết

- **Bài toán:** Truy xuất ảnh tương tự (Image Retrieval) – với một ảnh truy vấn (query), hệ thống phải tìm trong kho ảnh những ảnh **giống nhất** về mặt nội dung/nhận diện.
- **Phạm vi:** Kho ảnh cố định (CIFAR-10, 60.000 ảnh); truy vấn là ảnh upload hoặc ảnh chọn từ kho; trả về top-K ảnh gần nhất (K = 10).
- **Thách thức:** Cần so sánh nhanh giữa hàng chục nghìn ảnh; phải dùng biểu diễn ảnh (đặc trưng) có ý nghĩa để “giống” phản ánh đúng sự tương đồng nội dung.

### Chiến lược giải quyết

1. **Trích xuất đặc trưng (Feature extraction)**  
   Dùng mô hình CNN đã huấn luyện sẵn (ResNet-18, ImageNet) làm bộ trích xuất đặc trưng cố định (không fine-tune). Mỗi ảnh được biểu diễn bằng một vector 512 chiều. Cách này tận dụng tri thức từ ImageNet, phù hợp khi không có đủ dữ liệu/nhu cầu để huấn luyện lại.

2. **Lưu trữ và tìm kiếm**  
   Lưu toàn bộ vector đặc trưng (file `features.npy`). Khi có truy vấn, trích đặc trưng ảnh truy vấn rồi tìm K vector gần nhất theo độ đo **cosine similarity** (sau khi chuẩn hóa L2).

3. **Phương pháp tìm kiếm: FAISS**  
   Dùng **FAISS IndexFlatIP**: tích vô hướng trên vector đã L2-normalize tương đương cosine similarity. Đây là *exact search* (không nén vector) nên luôn trả về đúng top-K theo cosine, lại có tốc độ truy vấn rất cao.

4. **Triển khai phục vụ (serving)**  
   Backend Flask nhận ảnh (upload hoặc chỉ số ảnh trong kho), gọi pipeline trích đặc trưng + tìm kiếm, trả về top-10 ảnh và các chỉ số đánh giá (khi có nhãn).

---

## 3. Chỉ số đo lường – Định nghĩa và lý do chọn

Các chỉ số dùng để đo **chất lượng truy xuất** của hệ thống. Mỗi chỉ số được tính khi truy vấn là ảnh trong CIFAR-10 (có nhãn).

### Recall@10

- **Định nghĩa:** Trong số tất cả ảnh **cùng lớp** với ảnh truy vấn (trừ chính nó), có bao nhiêu ảnh nằm trong top-10 kết quả.  
  Recall@10 = (số ảnh đúng lớp trong top-10) / (tổng ảnh cùng lớp trong kho − 1).

- **Lý do chọn:** Bài toán retrieval quan tâm “tìm được bao nhiêu ảnh đúng trong top-K”. Recall@10 đo khả năng hệ thống “gom” được ảnh cùng lớp lên top; phù hợp khi mỗi lớp có nhiều ảnh (CIFAR-10: 6000 ảnh/lớp).

### Precision@10

- **Định nghĩa:** Trong top-10 ảnh trả về, bao nhiêu ảnh **đúng lớp** với truy vấn.  
  Precision@10 = (số ảnh đúng lớp trong top-10) / 10.

- **Lý do chọn:** Bổ sung cho Recall: đo chất lượng “độ sạch” của danh sách top-10. Recall cao nhưng precision thấp nghĩa là vẫn còn nhiều ảnh sai lớp trong top.

### Average Precision (AP)

- **Định nghĩa:** Với một truy vấn, tại mỗi vị trí k trong danh sách kết quả mà ảnh đó **đúng lớp**, tính Precision@k; AP là trung bình các giá trị Precision@k đó.

- **Lý do chọn:** AP vừa xét số ảnh đúng vừa xét **thứ tự**: ảnh đúng càng nằm trên đầu danh sách thì AP càng cao. Phù hợp chuẩn đánh giá retrieval (thứ hạng quan trọng).

### Tốc độ (thời gian/query)

- **Định nghĩa:** Thời gian (ms) từ lúc có vector truy vấn đến khi có xong top-10 bằng FAISS.

- **Lý do chọn:** Đo hiệu năng thực tế của hệ thống; FAISS cho thời gian truy vấn rất thấp (thường chỉ vài ms) trên 60.000 vector.

---

## 6. Cải tiến thuật toán và kỹ thuật

- **Chuẩn hóa L2 + Cosine:** Vector đặc trưng được chuẩn hóa L2; khi đó tích vô hướng (FAISS IndexFlatIP) tương đương cosine similarity. Cách này cho phép dùng FAISS exact search với metric cosine mà không cần metric khác.

- **FAISS IndexFlatIP:** Exact search, không nén vector → luôn trả về đúng top-K theo cosine, độ chính xác tuyệt đối. Cải tiến tiếp có thể: IVF, PQ, HNSW khi scale lên hàng triệu ảnh.

- **Tham số K = 10:** Chọn top-10 để cân bằng giữa số lượng kết quả hữu ích và độ khó (Recall/Precision không quá dễ hay quá khó). Có thể thử K = 5, 20 và ghi lại trong báo cáo nếu cần.

---

## 7. Đánh giá chất lượng mô hình / giải pháp

- **Mô hình đặc trưng:** ResNet-18 pretrained ImageNet, **không fine-tune**. “Chất lượng” ở đây được đánh giá gián tiếp qua **chất lượng retrieval** (Recall@10, Precision@10, AP), không qua loss/accuracy classification.

- **Tham số liên quan:**  
  - ResNet-18: kiến trúc cố định, weights cố định (ImageNet).  
  - FAISS: top-K = 10, metric cosine (tương đương inner product sau khi L2-normalize).  
  Không có quá trình “train” tham số; chỉ có siêu tham số K có thể thay đổi.

- **Cách đánh giá đã làm:**  
  - Với mỗi truy vấn (ảnh trong CIFAR-10), tính Recall@10, Precision@10, AP bằng FAISS; in ra terminal và hiển thị trên web.  
  - Đo thời gian (ms) mỗi truy vấn.

- **Phân tích chất lượng:**  
  - Recall/Precision/AP phản ánh chất lượng đặc trưng ResNet-18 trên CIFAR-10 (ảnh nhỏ, 10 lớp).  
  - Thời gian truy vấn FAISS rất nhỏ (vài ms) cho thấy giải pháp vừa chính xác (exact search) vừa hiệu quả.

---

## 8. Thảo luận kết quả

- **Tốc độ:** FAISS (IndexFlatIP) cho thời gian truy vấn rất thấp (thường chỉ vài ms) trên 60.000 vector 512 chiều; con số cụ thể phụ thuộc máy (CPU/GPU, thư viện).

- **Độ chính xác retrieval:** Trên CIFAR-10, Recall@10 và Precision@10 thường ở mức khiêm tốn vì (1) ảnh 32×32, chất lượng thấp; (2) ResNet-18 pretrained ImageNet chưa tối ưu cho 10 lớp CIFAR. AP phản ánh thứ hạng: ảnh đúng càng lên đầu thì AP càng cao.

- **Kết luận ngắn:** Giải pháp dùng ResNet-18 + L2-normalize + FAISS IndexFlatIP đạt độ chính xác retrieval cao (exact search) và tốc độ truy vấn tốt, phù hợp làm nền cho mở rộng (dataset lớn hơn, index gần đúng).

---

## 9. Hướng cải thiện

- **Dữ liệu:** Thử dataset lớn hơn hoặc độ phân giải cao hơn (ImageNet subset, ảnh tự thu thập) để đánh giá scalability và chất lượng đặc trưng.

- **Mô hình:** Fine-tune ResNet (hoặc dùng backbone khác) trên CIFAR-10 hoặc domain gần với ứng dụng; có thể cải thiện Recall/Precision/AP.

- **Chỉ số và tham số:** Báo cáo thêm mAP (mean AP trên nhiều query); thử nhiều K (5, 20, 50) và ghi nhận xu hướng Recall/Precision.

- **FAISS:** Khi số ảnh rất lớn, chuyển sang index gần đúng (IVF, PQ, HNSW) để giảm thời gian và bộ nhớ; đánh giá trade-off recall vs tốc độ.

- **Giao diện và trải nghiệm:** Đã hiển thị các chỉ số (Recall@10, Precision@10, AP) lên giao diện web khi truy vấn từ ảnh trong kho; có thể bổ sung mAP trung bình nhiều truy vấn.

---

## 10. Tóm tắt giải pháp end-to-end

1. **Dữ liệu:** CIFAR-10 (60.000 ảnh) – tải và tiền xử lý (resize 224×224, chuẩn hóa ImageNet) trong `feature_extractor.py`.  
2. **Đặc trưng:** Trích vector 512 chiều bằng ResNet-18 (bỏ lớp FC), lưu `features.npy`, `labels.npy` và `image_list.txt` (base64) – script `feature_extractor.py`.  
3. **Chuẩn hóa:** L2-normalize toàn bộ vector; FAISS dùng IndexFlatIP (inner product = cosine khi đã L2).  
4. **Tìm kiếm:** Với mỗi truy vấn, trích đặc trưng (nếu là ảnh mới) rồi tìm top-10 bằng FAISS; tính Recall@10, Precision@10, AP và đo thời gian khi có nhãn.  
5. **Serving:** Flask backend (`main.py`) nhận upload hoặc chỉ số ảnh, trả về top-10 ảnh và (trong terminal) các chỉ số đánh giá; giao diện web hiển thị ảnh truy vấn và kết quả.

---

## 11. Điểm thú vị và khó – Cải tiến implementation

- **Thú vị:**  
  - Dùng L2-normalize + inner product để đạt cosine trong FAISS, không cần metric riêng.  
  - FAISS IndexFlatIP là exact search nên vừa nhanh vừa cho kết quả chính xác tuyệt đối.  
  - Cùng một pipeline (ResNet → vector → search) phục vụ cả ảnh trong kho và ảnh upload.

- **Khó:**  
  - Đảm bảo thứ tự ảnh (train rồi test) khi gộp CIFAR-10 để nhãn và feature khớp từng chỉ số.  
  - Cài đặt FAISS đúng môi trường (faiss-cpu / faiss-gpu) để tránh lỗi import.

- **Cải tiến implementation:**  
  - Bắt lỗi thiếu `faiss` và hướng dẫn cài `faiss-cpu` trong thông báo lỗi.  
  - Tách hàm tính từng metric (Recall@10, Precision@10, AP) rõ ràng; in đủ chỉ số ra terminal và hiển thị trên web khi truy vấn có nhãn.  
  - Cấu trúc thư mục: `Data/` (raw CIFAR-10), `features/` (features.npy, image_list.txt) cùng cấp với `src/` để dễ triển khai và báo cáo.

---

