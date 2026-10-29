# Kiến Trúc Bản Demo Tương Tác: Gradio trên Hugging Face Spaces

**Nền tảng mục tiêu:** Hugging Face Spaces (CPU Basic: 2 vCPU, 16 GB RAM, Miễn phí)  
**Thư viện:** Gradio $\ge 4.20.0$, PyTorch $\ge 2.1.0$ (Bản CPU)  
**Nhóm sinh viên:** Nguyễn Văn Thành & Trần Minh Luân (CS331)  

---

## 1. Kiến Trúc Hệ Thống & Tách Rời Trách Nhiệm

Hệ thống demo tách biệt hoàn toàn giữa giao diện Web (Gradio) và cỗ máy suy luận (Inference Engine). Không chứa bất kỳ dữ liệu huấn luyện nặng nề nào trên Space.

```mermaid
flowchart TD
    subgraph Offline [Huấn luyện Ngoại tuyến trên Kaggle GPU]
        Kaggle[Huấn luyện STFPM / Mô hình Nén trên Kaggle] --> CheckpointExport[Xuất Checkpoint nhẹ: checkpoint.pth]
    end

    subgraph HuggingFaceSpace [Hạ tầng Hugging Face Spaces - CPU Basic]
        Init[Khởi động Space] --> LoadModel[Nạp Model lên CPU đúng 1 lần duy nhất]
        LoadModel --> Ready[Sẵn sàng phục vụ người dùng]

        User[Trình duyệt Người dùng] -->|Tải 1 ảnh lên & Chọn danh mục| UI[Giao diện Gradio Web]
        UI -->|Ảnh PIL + Danh mục + Ngưỡng tau| Engine[AnomalyDetectionEngine.predict()]

        Engine --> Preproc[Tiền xử lý: Resize 256x256 + Chuẩn hóa ImageNet]
        Preproc --> ModelForward[Lan truyền thuận trên CPU: torch.no_grad]
        ModelForward --> AnomalyMapGen[Tính Anomaly Map: Sai lệch chuẩn hóa L2]
        AnomalyMapGen --> PostProc[Phủ màu Jet + Trộn Alpha 50% + Đo độ trễ ms]
        PostProc --> UI

        UI -->|Hiển thị 3 ảnh + Điểm số + Kết luận PASS/DEFECT| User
    end
```

---

## 2. Các Thành Phần Chức Năng trên Giao Diện

### Đầu vào:
1. **Khung tải ảnh:** Tải 1 ảnh sản phẩm bất kỳ (PNG, JPG) hoặc chọn ảnh mẫu có sẵn trong `demo/examples/`.
2. **Chọn danh mục:** Dropdown chọn `carpet`, `bottle`, `screw`.
3. **Thanh trượt ngưỡng quyết định ($\tau$):** Cho phép điều chỉnh độ nhạy từ $0.05$ đến $1.00$ (mặc định $0.35$).

### Đầu ra:
1. **Ảnh gốc:** Hiển thị ảnh sau tiền xử lý ($256 \times 256$).
2. **Bản đồ nhiệt lỗi (Anomaly Heatmap):** Bản đồ nhiệt màu Jet thể hiện mức độ dị thường ở từng điểm ảnh.
3. **Ảnh phủ khuyết tật (Alpha Overlay):** Trộn 50% ảnh gốc và 50% heatmap để công nhân thấy rõ vị trí khuyết tật trên sản phẩm thật.
4. **Thẻ kết luận tự động:**
   - Kết luận kiểm định: **`PASS (Normal Product)`** hoặc **`DEFECT (Anomaly Detected)`**.
   - Điểm số dị thường tổng thể ($S_{\text{image}}$).
   - Độ trễ đo đạc thực tế trên CPU (ví dụ: `CPU Latency: 42.5 ms (23.5 FPS)`).
5. **Hộp giải thích khoa học:** Trình bày ngắn gọn nguyên lý chưng cất tri thức và các hạn chế vật lý (phản xạ ánh sáng, bụi thấu kính).

---

## 3. Ràng Buộc Kỹ Thuật & Giải Pháp Tối Ưu

| Ràng buộc | Giải pháp kỹ thuật |
| :--- | :--- |
| **Không có GPU trên Space miễn phí** | Mô hình học sinh STFPM chỉ có **2.78M tham số**, chạy cực nhẹ trên CPU với thời gian suy luận chỉ khoảng $35–55\text{ ms/ảnh}$ ($>18\text{ FPS}$). |
| **Không tải dataset MVTec nặng lên Space** | Không chứa dữ liệu train; chỉ kèm 3 ảnh mẫu nhỏ ($\sim 150\text{ KB/ảnh}$) trong `demo/examples/` để người dùng thử nghiệm nhanh. |
| **Không dùng điểm số giả mạo** | Thực hiện tính toán lan truyền thuận PyTorch và hàm khoảng cách chuẩn hóa thật 100%. |
