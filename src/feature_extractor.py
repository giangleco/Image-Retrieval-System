import os
# Bỏ qua xung đột thư viện (Intel MKL, OpenMP) trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import base64
from io import BytesIO

# ===========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & TÊN FILE TỰ ĐỘNG
# ===========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BẬT/TẮT DÙNG MÔ HÌNH TRIPLET TẠI ĐÂY
USE_FINETUNED = True
FINETUNED_CHECKPOINT = os.path.join(PROJECT_ROOT, "model", "resnet18_triplet_cifar10.pt")

# Tự động đặt tên file dựa vào biến USE_FINETUNED
if USE_FINETUNED and os.path.isfile(FINETUNED_CHECKPOINT):
    feature_filename = "features_triplet.npy"
else:
    feature_filename = "features_pretrained.npy"

FEATURES_PATH = os.path.join(OUTPUT_DIR, feature_filename)
IMAGELIST_PATH = os.path.join(OUTPUT_DIR, "image_list.txt")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.npy")

if __name__ == '__main__':
    print(">>> BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG CIFAR-10...")
    print(f"[*] Chế độ: {'TRIPLET LEARNING' if USE_FINETUNED else 'PRE-TRAINED'}")
    print(f"[*] File lưu: {feature_filename}")

    # Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                                  
        transforms.ToTensor(),                                          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # Tải dữ liệu
    DATA_ROOT = os.path.join(PROJECT_ROOT, "Data")
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform)

    full_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([trainset, testset]),
        batch_size=128,         
        shuffle=False,          
        num_workers=0,          
        pin_memory=False        
    )

    # ===========================================================
    # 2. KHỞI TẠO MÔ HÌNH 
    # ===========================================================
    if USE_FINETUNED and os.path.isfile(FINETUNED_CHECKPOINT):
        model = models.resnet18(weights=None)
        # Bỏ lớp FC cho khớp kiến trúc Triplet
        model.fc = torch.nn.Identity() 
        model.load_state_dict(torch.load(FINETUNED_CHECKPOINT, map_location="cpu"))
        model = torch.nn.Sequential(*list(model.children())[:-1])
        print(f"--- Đang dùng mô hình TRIPLET: {FINETUNED_CHECKPOINT}")
    else:
        model = models.resnet18(weights="IMAGENET1K_V1")
        model = torch.nn.Sequential(*list(model.children())[:-1])
        print("--- Đang dùng mô hình PRE-TRAINED gốc của ImageNet")
        
    device = torch.device("cpu")        
    model.eval().to(device)

    # Hàm lấy màu thật cho ảnh base64
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def denormalize(tensor):
        tensor = tensor.clone()           
        tensor = tensor * std + mean      
        return torch.clamp(tensor, 0, 1)   

    features = []       
    image_list = []     
    labels_list = []    

    print("\n>>> Đang xử lý 60.000 ảnh (sẽ mất vài phút)...")

    with torch.no_grad():  
        for batch_idx, (images, labels) in enumerate(full_loader):
            images = images.to(device)

            # Lấy đặc trưng (vector)
            feat = model(images)                    
            feat = feat.cpu().numpy().reshape(feat.shape[0], -1) 
            features.append(feat)
            
            # Lấy nhãn
            labels_list.append(labels.numpy())

            # Xử lý Base64
            for img_tensor in images.cpu():         
                img_tensor = denormalize(img_tensor)  
                img_pil = transforms.ToPILImage()(img_tensor)  
                buffered = BytesIO()
                img_pil.save(buffered, format="JPEG", quality=90)  
                img_str = base64.b64encode(buffered.getvalue()).decode()  
                image_list.append(img_str)

            if (batch_idx + 1) % 50 == 0:
                print(f"   Đã xử lý: {(batch_idx + 1) * 128:>6} / 60000 ảnh")

    # Lưu kết quả
    features = np.vstack(features)      
    labels_arr = np.concatenate(labels_list)

    np.save(FEATURES_PATH, features)
    np.save(LABELS_PATH, labels_arr)
    
    with open(IMAGELIST_PATH, "w") as f:
        f.write("\n".join(image_list))

    print("\n" + "="*60)
    print("HOÀN TẤT TRÍCH XUẤT!")
    print(f"Dữ liệu vector  : {FEATURES_PATH} (Shape: {features.shape})")
    print(f"Dữ liệu nhãn    : {LABELS_PATH} (Shape: {labels_arr.shape})")
    print("="*60 + "\n")