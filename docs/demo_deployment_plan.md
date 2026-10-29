# Kế Hoạch Đóng Gói và Triển Khai Bản Demo (Hugging Face Spaces)

**Nền tảng:** Hugging Face Spaces (`gradio` SDK)  
**Phần cứng:** CPU Basic (2 vCPU, 16 GB RAM, Miễn phí)  
**Chính sách:** Chưa đẩy weights hoặc code lên Space khi chưa có lệnh chính thức từ người hướng dẫn.  

---

## 1. Danh Sách Kiểm Tra Trước Khi Triển Khai (Deployment Gates)

- [ ] **Cổng 1: Đóng gói Checkpoint Tối ưu:**
  - File checkpoint `checkpoints/stfpm_carpet.pth` (hoặc mô hình nén) có dung lượng $\le 50\text{ MB}$, chỉ chứa `state_dict` của mạng học sinh (không chứa biến optimizer rác).
- [ ] **Cổng 2: Kiểm thử Cục bộ Không Lỗi:**
  - Chạy `python demo/app.py` trên máy cá nhân (chế độ CPU), xác nhận server khởi động tại `http://127.0.0.1:7860`.
  - Tải thử 1 ảnh mẫu, xác nhận xuất đủ 3 ảnh (Input, Heatmap, Overlay), điểm số và thời gian đo đạc mà không bị crash.
- [ ] **Cổng 3: Tối giản Hóa Thư Viện (`requirements.txt`):**
  - Chỉ chứa các gói chạy headless trên CPU: `gradio`, `torch`, `torchvision`, `numpy`, `pillow`, `matplotlib`.

---

## 2. Cấu Trúc Thư Mục Chuẩn Cho Hugging Face Space

```text
demo/
├── README.md               # Cấu hình YAML Space + Giới thiệu tổng quan
├── app.py                  # Mã nguồn ứng dụng Gradio & Inference Engine
├── requirements.txt        # Thư viện tối giản cho môi trường CPU
├── checkpoints/            # Thư mục chứa trọng số mô hình đã huấn luyện (.pth)
│   └── .gitkeep
└── examples/               # 2-3 ảnh sản phẩm mẫu để người dùng thử nghiệm nhanh
    └── .gitkeep
```

---

## 3. Các Bước Đẩy Lên Hugging Face Spaces Khi Có Lệnh

1. Tạo một Space mới tại [huggingface.co/new-space](https://huggingface.co/new-space):
   - Đặt tên Space: `industrial-anomaly-detection-kd`
   - Chọn SDK: **Gradio**
   - Chọn Hardware: **CPU Basic (Free)**
2. Thêm Git remote cho Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<tai_khoan_cua_ban>/industrial-anomaly-detection-kd
   ```
3. Đẩy riêng thư mục `demo/` lên Space và theo dõi nhật ký build cho đến khi trạng thái chuyển sang **Running**.
