import os
# Một số thư viện (Intel MKL, OpenMP) bị xung đột khi load nhiều lần → chương trình crash
# Dòng này ép Windows "bỏ qua" lỗi đó và cứ chạy tiếp → không bị tắt đột ngột
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# ===========================================================
# TỰ ĐỘNG TẠO THƯ MỤC LƯU KẾT QUẢ (data/features)
# ===========================================================
OUTPUT_DIR = os.path.join("data", "features")           # Đường dẫn thư mục lưu kết quả
os.makedirs(OUTPUT_DIR, exist_ok=True)                  # Nếu chưa có thì tạo mới, có rồi thì bỏ qua

FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.npy")     # File lưu vector đặc trưng (60000 x 512)
IMAGELIST_PATH = os.path.join(OUTPUT_DIR, "image_list.txt")  # File lưu 60000 ảnh dưới dạng base64

if __name__ == '__main__':
    print("Bắt đầu trích xuất deep features từ CIFAR-10 bằng ResNet-18...")
    print(f"Kết quả sẽ được lưu vào: {OUTPUT_DIR}/\n")

    # ===========================================================
    # 1. TIỀN XỬ LÝ ẢNH (phải giống lúc huấn luyện ImageNet)
    # ===========================================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                                  # ResNet yêu cầu ảnh 224x224
        transforms.ToTensor(),                                          # Chuyển PIL → Tensor (0-255 → 0-1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],               # Chuẩn hóa theo ImageNet
                             std=[0.229, 0.224, 0.225])                 # Giúp model nhận diện đặc trưng tốt hơn
    ])

    # ===========================================================
    # 2. TẢI TOÀN BỘ DỮ LIỆU CIFAR-10 (train + test = 60.000 ảnh)
    # ===========================================================
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  
                                            download=False, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download=False, transform=transform)

    # Gộp train + test lại thành 1 dataset duy nhất
    full_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([trainset, testset]),
        batch_size=128,         # Xử lý 128 ảnh/lần → nhanh hơn rất nhiều
        shuffle=False,          # Không xáo trộn để thứ tự ảnh cố định (rất quan trọng!)
        num_workers=0,          # BẮT BUỘC = 0 trên Windows để tránh lỗi multiprocessing
        pin_memory=False        # Tắt để tương thích Windows
    )

    # ===========================================================
    # 3. TẢI MÔ HÌNH RESNET-18 ĐÃ HUẤN LUYỆN SẴN TRÊN IMAGENET
    # ===========================================================
    model = models.resnet18(weights="IMAGENET1K_V1")        # Tải trọng số chính thức từ PyTorch
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp Fully Connected cuối → chỉ lấy đặc trưng
    model.eval()  # Chuyển sang chế độ đánh giá (không tính gradient)

    device = torch.device("cpu")        # Dùng CPU (ổn định, không lỗi CUDA)
    print(f"Đang sử dụng thiết bị: {device}")
    model.to(device)

    # ===========================================================
    # 4. HÀM DENORMALIZE: Đưa ảnh về màu sắc thật (để hiển thị đẹp trên web)
    # ===========================================================
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)   # Mean của ImageNet
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)   # Std của ImageNet

    def denormalize(tensor):
        tensor = tensor.clone()           # Tránh thay đổi tensor gốc
        tensor = tensor * std + mean      # Đảo ngược phép Normalize
        return torch.clamp(tensor, 0, 1)   # Giới hạn giá trị trong [0,1]

    # ===========================================================
    # 5. BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG & CHUYỂN ẢNH THÀNH BASE64
    # ===========================================================
    features = []       # Danh sách lưu vector đặc trưng (mỗi ảnh → 512 chiều)
    image_list = []     # Danh sách lưu ảnh dưới dạng chuỗi base64 (để hiển thị trên web)

    print("Bắt đầu xử lý 60.000 ảnh CIFAR-10... (khoảng 4-8 phút trên CPU)\n")

    with torch.no_grad():  # Tắt tính gradient → tiết kiệm RAM + chạy nhanh hơn rất nhiều
        for batch_idx, (images, _) in enumerate(full_loader):   # _ là label, không cần dùng
            images = images.to(device)

            # === Trích xuất đặc trưng ===
            feat = model(images)                    # Output shape: (batch, 512, 1, 1)
            feat = feat.cpu().numpy()               # Chuyển về numpy
            feat = feat.reshape(feat.shape[0], -1)  # → (batch, 512)
            features.append(feat)

            # === Chuyển ảnh thành base64 để hiển thị trên web (màu đúng, đẹp) ===
            for img_tensor in images.cpu():         # Duyệt từng ảnh trong batch
                img_tensor = denormalize(img_tensor)  # Đưa về màu gốc
                img_pil = transforms.ToPILImage()(img_tensor)  # Tensor → PIL Image
                buffered = BytesIO()
                img_pil.save(buffered, format="JPEG", quality=90)  # Chất lượng cao
                img_str = base64.b64encode(buffered.getvalue()).decode()  # Mã hóa base64
                image_list.append(img_str)

            # In tiến độ mỗi 50 batch (~6400 ảnh)
            if (batch_idx + 1) % 50 == 0:
                processed = (batch_idx + 1) * 128
                print(f"   Đã xử lý: {processed:>6} / 60000 ảnh")

    # ===========================================================
    # 6. GỘP DỮ LIỆU & LƯU XUỐNG Ổ CỨNG
    # ===========================================================
    features = np.vstack(features)      # Gộp tất cả batch thành 1 mảng lớn (60000, 512)
    print(f"\nFeatures shape: {features.shape}")   # Phải ra (60000, 512)

    # Lưu file đặc trưng (binary)
    np.save(FEATURES_PATH, features)
    
    # Lưu danh sách ảnh base64 (text, mỗi dòng 1 ảnh)
    with open(IMAGELIST_PATH, "w") as f:
        f.write("\n".join(image_list))

    # ===========================================================
    # HOÀN TẤT – THÔNG BÁO ĐẸP
    # ===========================================================
    print("\n" + "="*60)
    print("HOÀN TẤT 100%!")
    print("Trích xuất đặc trưng và chuyển ảnh thành công!")
    print(f"Đã lưu:")
    print(f"   → {FEATURES_PATH}")
    print(f"   → {IMAGELIST_PATH}")
    print("\nBây giờ bạn có thể chạy: python src/main.py")
    print("Mở http://127.0.0.1:5000 để dùng web tìm ảnh giống!")
    print("="*60)