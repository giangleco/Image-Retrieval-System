# Báo cáo đánh giá – Đáp ứng tiêu chí đồ án

Tài liệu này bổ sung các nội dung theo yêu cầu tiêu chí đánh giá đồ án: xác định vấn đề & chiến lược, định nghĩa và lý do chọn chỉ số, cải tiến thuật toán, đánh giá chất lượng mô hình, thảo luận kết quả, hướng cải thiện, tóm tắt giải pháp và điểm thú vị/khó.

---

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

3. **Hai phương pháp tìm kiếm (để so sánh)**  
   - **KNN (scikit-learn):** brute-force, cosine – làm chuẩn (ground truth) vì luôn trả về đúng top-K theo metric.  
   - **FAISS (IndexFlatIP):** tích vô hướng trên vector đã L2-normalize tương đương cosine; dùng làm phương pháp chính vì tốc độ cao, vẫn exact search.

4. **Triển khai phục vụ (serving)**  
   Backend Flask nhận ảnh (upload hoặc chỉ số ảnh trong kho), gọi pipeline trích đặc trưng + tìm kiếm, trả về top-10 ảnh và các chỉ số đánh giá (khi có nhãn).

---

## 3. Chỉ số đo lường – Định nghĩa và lý do chọn

Các chỉ số dùng để đo **hiệu suất truy xuất** và **độ thống nhất giữa KNN và FAISS**. Mỗi chỉ số được tính khi truy vấn là ảnh trong CIFAR-10 (có nhãn).

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

### Overlap@10 (KNN vs FAISS)

- **Định nghĩa:** Độ trùng tập hợp giữa hai danh sách top-10 (chỉ số ảnh) do KNN và FAISS trả về, dùng **Jaccard similarity**:  
  Overlap@10 = |Top10_KNN ∩ Top10_FAISS| / |Top10_KNN ∪ Top10_FAISS|.  
  Giá trị từ 0 (không trùng ảnh nào) đến 1 (trùng hoàn toàn 10 ảnh).

- **Lý do chọn:** Để kiểm chứng FAISS (IndexFlatIP + L2) có cho **cùng tập kết quả** với KNN (cosine) hay không. Overlap gần 1 chứng tỏ hai phương pháp nhất quán; khác biệt có thể do thứ tự hoặc do sai số số học.

### Tốc độ (thời gian/query)

- **Định nghĩa:** Thời gian (ms) từ lúc có vector truy vấn đến lúc có xong top-10 (cho KNN và cho FAISS riêng).

- **Lý do chọn:** So sánh hiệu năng thực tế; FAISS thường nhanh hơn nhiều so với KNN brute-force trên cùng dữ liệu.

---

## 6. Cải tiến thuật toán và kỹ thuật

- **Chuẩn hóa L2 + Cosine:** Vector đặc trưng được chuẩn hóa L2; khi đó tích vô hướng (FAISS IndexFlatIP) tương đương cosine similarity. Cách này cho phép dùng FAISS exact search với metric cosine mà không cần metric khác.

- **KNN làm baseline:** KNN (scikit-learn, brute-force, metric cosine) dùng làm chuẩn so sánh vì luôn trả về đúng top-K theo cosine; mọi phương pháp tối ưu tốc độ (FAISS, IVF, …) có thể so Recall/Precision/AP/Overlap với KNN.

- **FAISS IndexFlatIP:** Exact search, không nén vector; đảm bảo kết quả trùng với KNN (trong phạm vi sai số số học). Cải tiến tiếp có thể: IVF, PQ, HNSW khi scale lên hàng triệu ảnh.

- **Tham số K = 10:** Chọn top-10 để cân bằng giữa số lượng kết quả hữu ích và độ khó (Recall/Precision không quá dễ hay quá khó). Có thể thử K = 5, 20 và ghi lại trong báo cáo nếu cần.

---

## 7. Đánh giá chất lượng mô hình / giải pháp

- **Mô hình đặc trưng:** ResNet-18 pretrained ImageNet, **không fine-tune**. “Chất lượng” ở đây được đánh giá gián tiếp qua **chất lượng retrieval** (Recall@10, Precision@10, AP), không qua loss/accuracy classification.

- **Tham số liên quan:**  
  - ResNet-18: kiến trúc cố định, weights cố định (ImageNet).  
  - KNN/FAISS: `n_neighbors = 10`, metric cosine (tương đương inner product sau L2).  
  Không có quá trình “train” tham số; chỉ có siêu tham số K có thể thay đổi và so sánh.

- **Cách đánh giá đã làm:**  
  - Với mỗi truy vấn (ảnh trong CIFAR-10), tính Recall@10, Precision@10, AP cho cả KNN và FAISS; in ra terminal.  
  - Tính Overlap@10 giữa KNN và FAISS để xác nhận hai phương pháp cho kết quả gần nhau.  
  - Đo thời gian (ms) cho từng phương pháp.

- **Phân tích chất lượng:**  
  - Recall/Precision/AP phản ánh chất lượng đặc trưng ResNet-18 trên CIFAR-10 (ảnh nhỏ, 10 lớp).  
  - Overlap@10 gần 1 và thời gian FAISS nhỏ hơn nhiều so với KNN cho thấy giải pháp FAISS vừa chính xác vừa hiệu quả.

---

## 8. Thảo luận kết quả

- **Tốc độ:** FAISS (IndexFlatIP) thường nhanh hơn KNN brute-force đáng kể (nhiều lần đến hàng chục lần) trên 60.000 vector 512 chiều; con số cụ thể phụ thuộc máy (CPU/GPU, thư viện).

- **Độ chính xác retrieval:** Trên CIFAR-10, Recall@10 và Precision@10 thường ở mức khiêm tốn vì (1) ảnh 32×32, chất lượng thấp; (2) ResNet-18 pretrained ImageNet chưa tối ưu cho 10 lớp CIFAR. AP phản ánh thứ hạng: ảnh đúng càng lên đầu thì AP càng cao.

- **Overlap@10:** Với exact search (KNN cosine và FAISS IndexFlatIP + L2), hai phương pháp trả về cùng tập top-10 (hoặc gần như vậy); Overlap@10 thường rất cao (gần 1). Khác biệt nhỏ có thể do thứ tự đồng hạng hoặc sai số số học.

- **Kết luận ngắn:** Giải pháp dùng ResNet-18 + L2-normalize + FAISS IndexFlatIP đạt độ chính xác retrieval tương đương KNN và tốc độ tốt hơn rõ rệt, phù hợp làm nền cho mở rộng (dataset lớn hơn, index gần đúng).

---

## 9. Hướng cải thiện

- **Dữ liệu:** Thử dataset lớn hơn hoặc độ phân giải cao hơn (ImageNet subset, ảnh tự thu thập) để đánh giá scalability và chất lượng đặc trưng.

- **Mô hình:** Fine-tune ResNet (hoặc dùng backbone khác) trên CIFAR-10 hoặc domain gần với ứng dụng; có thể cải thiện Recall/Precision/AP.

- **Chỉ số và tham số:** Báo cáo thêm mAP (mean AP trên nhiều query); thử nhiều K (5, 20, 50) và ghi nhận xu hướng Recall/Precision.

- **FAISS:** Khi số ảnh rất lớn, chuyển sang index gần đúng (IVF, PQ, HNSW) để giảm thời gian và bộ nhớ; đánh giá trade-off recall vs tốc độ.

- **Giao diện và trải nghiệm:** Hiển thị thêm các chỉ số (Recall@10, Precision@10, AP, Overlap@10) lên giao diện web khi truy vấn từ ảnh trong kho.

---

## 10. Tóm tắt giải pháp end-to-end

1. **Dữ liệu:** CIFAR-10 (60.000 ảnh) – download và tiền xử lý (resize 224×224, chuẩn hóa ImageNet) trong notebook `Data_process.ipynb`.  
2. **Đặc trưng:** Trích vector 512 chiều bằng ResNet-18 (bỏ lớp FC), lưu `features.npy` và `image_list.txt` (base64) – script `feature_extractor.py` hoặc notebook `Feature_extraction.ipynb`.  
3. **Chuẩn hóa:** L2-normalize toàn bộ vector; KNN dùng metric cosine, FAISS dùng IndexFlatIP (inner product = cosine khi đã L2).  
4. **Tìm kiếm:** Với mỗi truy vấn, trích đặc trưng (nếu là ảnh mới) rồi tìm top-10 bằng KNN và FAISS; so sánh thời gian và tính Recall@10, Precision@10, AP, Overlap@10 khi có nhãn.  
5. **Serving:** Flask backend (`main.py`) nhận upload hoặc chỉ số ảnh, trả về top-10 ảnh và (trong terminal) các chỉ số đánh giá; giao diện web hiển thị ảnh truy vấn và kết quả.

---

## 11. Điểm thú vị và khó – Cải tiến implementation

- **Thú vị:**  
  - Dùng L2-normalize + inner product để đạt cosine trong FAISS, không cần metric riêng.  
  - So sánh trực tiếp KNN vs FAISS (tốc độ + Overlap@10) giúp tin tưởng khi chuyển sang FAISS.  
  - Cùng một pipeline (ResNet → vector → search) phục vụ cả ảnh trong kho và ảnh upload.

- **Khó:**  
  - Đảm bảo thứ tự ảnh (train rồi test) khi gộp CIFAR-10 để nhãn và feature khớp từng chỉ số.  
  - Cài đặt FAISS đúng môi trường (faiss-cpu / faiss-gpu) để tránh lỗi import.

- **Cải tiến implementation:**  
  - Bắt lỗi thiếu `faiss` và hướng dẫn cài `faiss-cpu` trong thông báo lỗi.  
  - Tách hàm tính từng metric (Recall@10, Precision@10, AP, Overlap@10) rõ ràng; in đủ chỉ số ra terminal khi truy vấn có nhãn.  
  - Cấu trúc thư mục: `Data/` (raw CIFAR-10), `features/` (features.npy, image_list.txt) cùng cấp với `src/` để dễ triển khai và báo cáo.

---

*Tài liệu này bổ sung cho README và code trong repo, dùng để đối chiếu với các tiêu chí đánh giá đồ án.*
