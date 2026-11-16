# Đề Xuất Đóng Băng Thiết Kế Cài Đặt (Implementation Freeze Proposal)

**Dự án:** Knowledge Distillation-Based Lightweight Industrial Anomaly Detection (`CS331`)  
**Kho mã nguồn:** `cs331-knowledge-distillation-lightweight-industrial-anomaly-detection`  
**Nhánh làm việc:** `chore/repository-reset` (HEAD: `41af7b015a69fe7480c8fd1212e6e22840ae22d0`)  
**Vai trò:** Lead Research Engineer & Independent Technical Reviewer  
**Ngày lập đề xuất:** `2026-11-14`  
**Tài liệu tham chiếu:** `papers/2402.00448v1.pdf`, `docs/research/post_audit_verification.md`, `docs/research/final_technical_audit.md`  
**Trạng thái đề xuất:** **`READY FOR USER APPROVAL (IMPLEMENTATION PROPOSAL READY)`**  
*(Dừng tại đây — chưa viết code mô hình, chưa tạo notebook, chờ người dùng phê duyệt).*

---

## 1. Tóm Tắt Điều Hành & Khung Phân Loại Nhận Thức

Tài liệu này tổng hợp toàn bộ các quyết định giải quyết 4 điểm nghẽn kỹ thuật (blockers) được chỉ ra qua báo cáo kiểm toán độc lập [post_audit_verification.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/post_audit_verification.md), đồng bộ hóa toàn bộ các tài liệu nghiên cứu trong `docs/research/`, và thiết lập bản đặc tả kỹ thuật chi tiết nhất để sẵn sàng bước vào Pha 2 (Cài đặt mã nguồn trong `src/anomaly_detection/`).

### Hệ thống Nhãn Nhận thức (Epistemic Tags)
- **`[FACT - Paper]`:** Dữ kiện, phương trình, thuật toán và bảng số liệu trích xuất trực tiếp từ bài báo gốc [2402.00448v1.pdf](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/papers/2402.00448v1.pdf).
- **`[FACT - Mathematical Proof]`:** Đẳng thức toán học, hành vi giới hạn giải tích hoặc số lượng tham số/FLOPs được kiểm chứng bằng tính toán đại số và PyTorch execution.
- **`[IMPLEMENTATION CHOICE]`:** Quyết định kỹ thuật được nhóm nghiên cứu lựa chọn để giải quyết các điểm bài báo gốc không công bố mã nguồn hoặc mô tả thiếu chi tiết.
- **`[HYPOTHESIS]`:** Giả thuyết khoa học về tính vượt trội hoặc sự đánh đổi (trade-off) của các cải tiến, cần được kiểm chứng qua thực nghiệm.
- **`[NOT VERIFIED]`:** Các đại lượng runtime (độ trễ ms/ảnh, bộ nhớ VRAM đỉnh thực tế, phụ phí FLOPs của Method A) chưa thể đo đạc do chưa có code mô hình trong repo.

---

## 2. Bảng Quyết Định Giải Quyết Bốn Điểm Nghẽn Kỹ Thuật (Four-Blocker Decision Table)

| Điểm Nghẽn (Blocker) | Câu Hỏi Kỹ Thuật Cốt Lõi | Các Phương Án Cân Nhắc | Quyết Định Đề Xuất Chính Thức | Cơ Sở Khoa Học & Phân Biệt `[FACT]` vs `[CHOICE]` |
| :--- | :--- | :--- | :--- | :--- |
| **Blocker 1: Kiến trúc `B0-Clean` Chuẩn hóa** | 1. Có giữ lớp $1\times 1$ conv trên $F_3$ trong DFE không?<br>2. Upsampling trong $S_d$ là `ConvTranspose2d` hay `Bilinear + Conv3x3`?<br>3. Đầu vào DFE là đặc trưng thô hay đã chuẩn hóa $\ell_2$?<br>4. Có ngắt đạo hàm `detach()` trước DFE không? | • $1\times 1$ conv: Giữ vs. Bỏ.<br>• $S_d$: Bilinear+3x3 vs. Deconv 2x2.<br>• DFE In: Raw $F_e$ vs. Normalized $\hat{F}_e$.<br>• Gradient: Detach vs. End-to-end. | **Cấu hình Canonical `B0-Clean`:**<br>1. **Giữ** $1\times 1$ conv trên $F_3$.<br>2. **Dùng Bilinear + Conv3x3** cho $S_d$.<br>3. **Đưa Raw $F_e$** vào DFE.<br>4. **Ngắt đạo hàm `f.detach()`** trước DFE. | • `Fig. 4` vẽ rõ khối `1x1 conv, bn, relu` trên $F_3$ `[FACT - Paper]`.<br>• Paper chỉ ghi *"reversed with transposed backbone"* `[FACT - Paper]`; Bilinear tránh sọc ca-rô (*checkerboard artifacts*), tăng chất lượng định vị pixel `[IMPLEMENTATION CHOICE]`. Deconv được giữ làm mốc so sánh FLOPs (`6.206G` MACs khớp `6.170G` Table 6).<br>• `Algo. 1` và `Fig. 4` dùng đặc trưng thô để bảo toàn biên độ năng lượng; ngắt đạo hàm trước DFE đảm bảo $S_e$ chỉ học từ $T$ theo đúng định nghĩa Dual-Student `[IMPLEMENTATION CHOICE]`. |
| **Blocker 2: Kiến trúc `Lite-DFE` Chuẩn hóa** | Nên chuẩn hóa theo thiết kế `dsmsa_db` ($192\alpha$ concat kênh) hay `dsmsa_expand` ($768\alpha$ concat kênh)? | • `dsmsa_db`: $d_b = 64\alpha \implies 192\alpha$ concat.<br>• `dsmsa_expand`: mở rộng $256\alpha \implies 768\alpha$ concat. | **Canonical Method B:** `dsmsa_db`.<br>**Ablation đối chứng:** `dsmsa_expand` trong `B1`. | • `dsmsa_db` giảm $98.4\%$ tham số DFE ở $\alpha=0.5, \rho=1.0$ ($51.9\text{k}$ params) và $98.9\%$ ở $\rho=0.5$ ($36.7\text{k}$ params), đã bao gồm đầy đủ $\text{PW}_{\text{skip}}$ `[FACT - Mathematical Proof]`.<br>• Tránh phình $768\alpha$ kênh trung gian, phù hợp tối đa cho biên tính toán nhẹ `[IMPLEMENTATION CHOICE]`. |
| **Blocker 3: Chuẩn hóa & Thí nghiệm Sơ bộ (Pilot)** | 1. Phân định công thức `Eq. (3)` vs. Tiền xử lý Mục 4.1.2 vs. Trước DFE.<br>2. Thí nghiệm 50 epochs có chứng minh được tái hiện bài báo không? | • `Eq. (3)`: $N_1$ (Unit-L2) vs. $N_2$ (Raw-L2+Cos) vs. $N_3$ (Literal Inv-Sq).<br>• Mục 4.1.2: ImageNet Z-score vs. $P_{\text{img\_l2}}$ (3 trục).<br>• 50 epochs vs. 200 epochs. | **Quy chuẩn rõ ràng:**<br>1. Tách biệt hoàn toàn 3 tầng chuẩn hóa.<br>2. `S0` 50 epochs **chỉ là kiểm tra ổn định số học / sụp đổ biểu diễn**.<br>3. Đánh giá tái hiện phải dùng 200 epochs (`G0`/`G1`). | • $N_3$ với $\epsilon > 0$ có đạo hàm bị chặn bởi $O(1/\epsilon)$ tại $r_v = \sqrt{\epsilon}$, không vô cùng nhưng gây méo thang đo nghiêm trọng `[FACT - Mathematical Proof]`.<br>• $P_{\text{img\_l2}}$ theo từng pixel triệt tiêu độ chói (*luminance*); toàn ảnh làm co tín hiệu $1/250$ lần `[FACT - Mathematical Proof]`.<br>• 50 epochs chỉ đạt $25\%$ tiến trình, không thể coi là tái hiện bài báo `[IMPLEMENTATION CHOICE]`. |
| **Blocker 4: Độ phức tạp & Ngân sách Kaggle** | 1. Thống nhất đơn vị FLOPs vs. MACs.<br>2. Sửa lỗi thiếu tham số $S_d$ và buffers.<br>3. Kiểm soát ngân sách GPU trước khi chạy. | • Báo cáo lẫn lộn FLOPs và MACs.<br>• Giả định $S_d = S_e$ (`2.783M`).<br>• Dự đoán thời gian không qua đo đạc. | **Chuẩn hóa thống kê:**<br>1. Đo bằng `FlopCounterMode` ($1\text{ MAC} = 2\text{ FLOPs}$).<br>2. Cập nhật $S_d = 3.326\text{M}$ (Bilinear) và $16.3\text{k}$ buffers cho `B0-Clean`.<br>3. Chạy benchmark sơ bộ `M0_bench` (10 phút) trước ma trận lớn. | • PyTorch `FlopCounterMode` đếm $2\text{ FLOPs}$ cho mỗi phép nhân-cộng `[FACT - Repo]`. Số liệu `6.170G` Table 6 tương ứng với MACs `[FACT - Paper]`.<br>• $S_d$ giải mã đối xứng thực tế có $3.326\text{M}$ tham số `[FACT - Mathematical Proof]`.<br>• `M0_bench` đo đạc thực tế vòng lặp, checkpoint và CPU `compute_aupro` trên node Kaggle thực tế để chốt hằng số thời gian `[IMPLEMENTATION CHOICE]`. |

---

## 3. Đặc Tả Kiến Trúc `B0-Clean` Chuẩn Hóa (Canonical Baseline Specification)

### 3.1. Phân Tích & Khóa 4 Thành Phần Kỹ Thuật
1. **Lớp tích chập $1\times 1$ trên nhánh $F_3$ của DFE:**
   - *Cơ sở trong bài báo:* Hình 4 (trang 5) của bài báo gốc in rõ một hộp xử lý `1x1 conv, bn, relu` trên nhánh $F_3$ trước khi đi vào khối ghép kênh `Concat`.
   - *Đặc tả kỹ thuật:* Nhận $F_3 \in \mathbb{R}^{B \times 256 \times 16 \times 16}$, đi qua `Conv2d(256, 256, kernel_size=1, bias=False) + BatchNorm2d(256) + ReLU()`.
   - *Chi phí:* $256 \times 256 \times 1 \times 1 = 65,536$ trọng số conv + $512$ tham số BN ($\gamma, \beta$) = **`66,048` tham số** và `513` phần tử đệm buffer. FLOPs $= 2 \times 65,536 \times 16^2 = 33,554,432$ FLOPs (**`0.0336G` FLOPs / `0.0168G` MACs**). Việc giữ lớp này đảm bảo độ trung thực cao nhất với Hình 4 và cho phép $F_3$ được chuẩn hóa thích nghi trước khi ghép với $F_1, F_2$.
2. **Cơ chế Lấy mẫu lên (Upsampling) của Student Decoder $S_d$:**
   - *Cơ sở trong bài báo:* Mục 1 (tr. 2) và Mục 3.1 (tr. 5) chỉ ghi *"transposed backbone, where downsampling operations are replaced by upsampling operations"*, không công bố mã nguồn hay loại toán tử cụ thể.
   - *Đặc tả kỹ thuật đề xuất:* Sử dụng **`nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) + Conv2d(3x3)`** trong các khối BasicBlock đảo ngược.
   - *Lý do khoa học:* Phép tích chập chuyển vị (`ConvTranspose2d`) thường gây nhiễu sọc bàn cờ (*checkerboard artifacts*) do bước nhảy stride $>1$, làm suy giảm chất lượng bản đồ dị thường pixel. Phép lấy mẫu song tuyến tính tạo ra bản đồ mượt mà, định vị chính xác hơn.
   - *Mốc đối chứng:* Về mặt tính toán, nếu dùng `ConvTranspose2d(2x2, stride=2)`, mô hình có `12.098M` tham số và **`6.206G` MACs** (khớp lệch chưa đầy $0.6\%$ so với con số `6.170G` in trong Table 6 của bài báo gốc!). Do đó, cả hai cách hiểu đều được tài liệu hóa rõ ràng: `Bilinear` là chuẩn chạy chính, `Deconv` là mốc giải thích tính toán.
3. **Đặc trưng Đầu vào Khối DFE:**
   - *Cơ sở trong bài báo:* Hình 4 dán nhãn đầu vào là $F_1, F_2, F_3$ (đặc trưng thô trước chuẩn hóa). Algorithm 2 dòng 7 ghi $\text{feature\_embedding}(\hat{F}_e)$ (đặc trưng đã chia chuẩn).
   - *Đặc tả kỹ thuật đề xuất:* Đưa **đặc trưng thô $F_e^k$** vào DFE. Phép chuẩn hóa $\ell_2$ chỉ áp dụng ở bước tính hàm mất mát `Eq. (3)` và bản đồ sai biệt. Việc đưa đặc trưng thô giúp DFE tiếp nhận đầy đủ thông tin về biên độ kích hoạt và phân phối năng lượng giữa các tầng.
4. **Cơ chế Độc lập Gradient giữa $S_e$ và $S_d$:**
   - *Cơ sở trong bài báo:* Mục 3.1 nêu rõ triết lý Dual-Student: $S_e$ học biểu diễn thuận trực tiếp từ Teacher $T$, trong khi $S_d$ học tái tạo biểu diễn ngược từ vector nhúng $F_{\text{emb}}$. Algorithm 1 dòng 16–17 ghi tách biệt: `Update parameters theta_e; Update parameters theta_d`.
   - *Đặc tả kỹ thuật đề xuất:* Áp dụng **ngắt đạo hàm `f.detach()`** trên các đặc trưng $F_e^k$ trước khi đưa vào DFE (`f_emb = self.dfe([f.detach() for f in raw_f_e])`) và sử dụng 2 optimizer riêng biệt (`optimizer_e` cho $S_e$, `optimizer_d` cho $\text{DFE} + S_d$). Điều này ngăn ngừa triệt để việc hàm mất mát tái tạo $\ell_d$ truyền gradient ngược làm biến dạng không gian biểu diễn của $S_e$.

### 3.2. Sơ Đồ Kiến Trúc Chi Tiết Từng Tầng Của `B0-Clean`

```text
Input Image: I \in R^{B x 3 x 256 x 256}
 │
 ├──> Teacher T (ResNet18, Pretrained ImageNet-1K, Frozen eval):
 │     ├── stem (conv1 + bn1 + maxpool)  --> 64 x 64 x 64
 │     ├── layer1 (conv2_x: 2 blocks)    --> F_t^1 \in R^{B x 64 x 64 x 64}
 │     ├── layer2 (conv3_x: 2 blocks)    --> F_t^2 \in R^{B x 128 x 32 x 32}
 │     └── layer3 (conv4_x: 2 blocks)    --> F_t^3 \in R^{B x 256 x 16 x 16}
 │
 ├──> Student Encoder S_e (ResNet18, Random Init, Trainable \theta_e):
 │     ├── stem (conv1 + bn1 + maxpool)  --> 64 x 64 x 64
 │     ├── layer1 (conv2_x: 2 blocks)    --> F_e^1 \in R^{B x 64 x 64 x 64}
 │     ├── layer2 (conv3_x: 2 blocks)    --> F_e^2 \in R^{B x 128 x 32 x 32}
 │     └── layer3 (conv4_x: 2 blocks)    --> F_e^3 \in R^{B x 256 x 16 x 16}
 │
 ├──> Deep Feature Embedding (DFE, Trainable \theta_dfe):
 │     Input: [F_e^1.detach(), F_e^2.detach(), F_e^3.detach()]
 │     ├── Branch 1: F_e^1 (64x64) -> Conv3x3(s=2)->BN->ReLU -> Conv3x3(s=2)->BN->ReLU --> 256 x 16 x 16
 │     ├── Branch 2: F_e^2 (32x32) -> Conv3x3(s=2)->BN->ReLU                           --> 256 x 16 x 16
 │     ├── Branch 3: F_e^3 (16x16) -> Conv1x1(s=1)->BN->ReLU                           --> 256 x 16 x 16
 │     ├── Concat(Branch 1, 2, 3)                                                      --> 768 x 16 x 16
 │     └── Residual Bottleneck Block:
 │           ├── Main: Conv3x3(768->256)->BN->ReLU -> Conv3x3(256->256)->BN
 │           ├── Shortcut: Conv1x1(768->256)->BN
 │           └── Output: F_emb = ReLU(Main + Shortcut)                                  --> 256 x 16 x 16
 │
 └──> Student Decoder S_d (Reversed ResNet18, Trainable \theta_d):
       Input: F_emb \in R^{B x 256 x 16 x 16}
       ├── rev_layer3 (2 blocks, 256->256, stride=1)                                    --> F_d^3 \in R^{B x 256 x 16 x 16}
       ├── rev_layer2 (Bilinear 2x upsample + 2 blocks, 256->128)                       --> F_d^2 \in R^{B x 128 x 32 x 32}
       └── rev_layer1 (Bilinear 2x upsample + 2 blocks, 128->64)                        --> F_d^1 \in R^{B x 64 x 64 x 64}
```

---

## 4. Đặc Tả Kiến Trúc Phương Pháp B (`Lite-DSKD`)

### 4.1. Phân Tích & Khóa Cấu Hình Khối `Lite-DFE`
Nhóm nghiên cứu phân định rõ ràng hai phương án thiết kế để giải quyết triệt để sự mâu thuẫn trong các tài liệu trước:

1. **Phương án Chuẩn Chính (Canonical Method B — `dsmsa_db`):**
   - *Cơ chế:* Trước khi ghép kênh, mỗi nhánh $k$ được chiếu điểm $\text{PW}_{1\times 1}$ về kích thước kênh co gọn $d_b = 64\alpha$ ($d_b = 64$ khi $\alpha=1.0$, $d_b = 32$ khi $\alpha=0.5$).
   - *Concat:* $U_{\text{cat}} \in \mathbb{R}^{B \times 3d_b \times 16 \times 16} = \mathbb{R}^{B \times 192\alpha \times 16 \times 16}$.
   - *Khối BDRF:* Nút cổ chai giữa $d_{\text{mid}} = 256\alpha \cdot \rho$ ($d_{\text{mid}} = 128$ khi $\alpha=0.5, \rho=1.0$; $d_{\text{mid}} = 64$ khi $\alpha=0.5, \rho=0.5$).
     - Nhánh chính: $192\alpha \xrightarrow{\text{PW}} d_{\text{mid}} \xrightarrow{\text{DW}_{3\times 3}} d_{\text{mid}} \xrightarrow{\text{PW}} 256\alpha$.
     - Nhánh tắt: $192\alpha \xrightarrow{\text{PW}_{\text{skip}}} 256\alpha$ (có kèm BatchNorm).
   - *Tham số đã kiểm chứng:*
     - Tại $\alpha=1.0, \rho=0.5$: **`141,056` params (`0.141M`)**, FLOPs $= \mathbf{0.072G}$ (giảm **$95.7\%$** tham số so với Dense-DFE `3.288M`).
     - Tại $\alpha=0.50, \rho=1.0$: **`51,904` params (`0.052M`)**, FLOPs $= \mathbf{0.026G}$ (giảm **$98.4\%$** tham số so với Dense-DFE).
     - Tại $\alpha=0.50, \rho=0.5$: **`36,736` params (`0.037M`)**, FLOPs $= \mathbf{0.019G}$ (giảm **$98.9\%$** tham số).
2. **Phương án Đối chứng Ablation (`dsmsa_expand`):**
   - *Cơ chế:* Các nhánh hạ mẫu mở rộng số kênh lên $256\alpha$ trước khi ghép kênh $\implies U_{\text{cat}} \in \mathbb{R}^{B \times 768\alpha \times 16 \times 16}$.
   - *Tham số đã kiểm chứng:*
     - Tại $\alpha=1.0, \rho=0.5$: **`408,896` params (`0.409M`)**, FLOPs $= \mathbf{0.221G}$.
     - Tại $\alpha=0.50, \rho=1.0$: **`137,696` params (`0.138M`)**, FLOPs $= \mathbf{0.073G}$.
     - Tại $\alpha=0.50, \rho=0.5$: **`104,096` params (`0.104M`)**, FLOPs $= \mathbf{0.056G}$.
3. **Quyết định Khóa:** Chọn `dsmsa_db` làm cấu hình chính cho toàn bộ ma trận Method B; giữ `dsmsa_expand` làm một thí nghiệm Ablation đơn lẻ trong Group 3 (`B1_expand`) trên Core-5 để kiểm tra xem việc mở rộng $768\alpha$ kênh có mang lại lợi ích về AUROC/PRO hay không.

### 4.2. Khối `Slim-DSKD` & Giao Thức Ablation Kích Hoạt Đầu Chiếu $\mathcal{P}_{e,d}^k$
- Mạng học sinh $S_e^{(\alpha)}$ và $S_d^{(\alpha)}$ có số kênh nội tại thu hẹp theo tỷ lệ $\alpha \in \{0.75, 0.50\}$: $(c_1^{(s)}, c_2^{(s)}, c_3^{(s)}) = (64\alpha, 128\alpha, 256\alpha)$.
- Đầu chiếu khôi phục số chiều $\alpha c_k \to c_k$ khớp với Teacher $T$:
  $$F_s^k = \mathcal{P}_s^k(G_s^k) = \phi\left(\text{BN}\left(W_{s,\text{proj}}^k *_{1\times 1} G_s^k\right)\right), \quad s \in \{e, d\}$$
- **Khóa Giao thức Ablation 3 biến thể hàm kích hoạt $\phi$ (Core-5):**
  1. `B2-BN-Only` (`proj_act="none"`): $\phi(z) = z$.
  2. `B2-BN-ReLU` (`proj_act="relu"`): $\phi(z) = \text{ReLU}(z)$ (khớp góc phần tư không âm với $F_t^k \ge 0$).
  3. `B2-BN-LeakyReLU` (`proj_act="leaky_relu"`): $\phi(z) = \text{LeakyReLU}_{0.01}(z)$ (vừa ép nhỏ miền âm, vừa chống hiện tượng chết kênh).

---

## 5. Đặc Tả Hàm Mục Tiêu & Cơ Chế Suy Luận Phương Pháp A (`FC-DSKD`)

1. **Mô-đun A1 (Spatial-Focal & Contextual Patch Discrepancy):**
   - Vector ngữ cảnh cục bộ: $\bar{F}_z^k = \text{AvgPool}_{3\times 3, s=1, p=1}(\hat{F}_z^k)$ kèm chuẩn hóa $\hat{G}_z^k = \bar{F}_z^k / (\|\bar{F}_z^k\|_2 + \epsilon)$ ($\epsilon = 10^{-8}$, padding phản chiếu).
   - Bản đồ sai biệt tổng hợp: $M_{t-s}^k = d_{\text{pt}}^{t-s, k} + \omega_{\text{ctx}} d_{\text{ctx}}^{t-s, k}$ với $\omega_{\text{ctx}} = 0.5$.
   - Trọng số tiêu cự tự chuẩn hóa: $w_s^k(I_m)_{ij} = \min(5.0, (1 + \text{sg}[\tilde{M}_{t-s}^k(I_m)_{ij}])^\gamma)$ với $\gamma = 1.0$.
2. **Mô-đun A2 (Asymmetric Stop-Gradient Cross-Student Consistency):**
   - Hàm mất mát có hướng: $\ell_{e-d}^{\text{asym}}(I_m) = \frac{1}{K}\sum_k \text{mean}(d_{\text{pt}}(\text{sg}[\hat{F}_e^k], \hat{F}_d^k) + \omega_{\text{ctx}} d_{\text{ctx}}(\text{sg}[\hat{G}_e^k], \hat{G}_d^k))$.
   - Cập nhật độc lập: $\mathcal{L}_e = \ell_e^{\text{focal}}$, $\mathcal{L}_d = \ell_d^{\text{focal}} + \beta(t) \ell_{e-d}^{\text{asym}}$ với $\beta(t) = 0.15 \min(1, t/20)$.
3. **Mô-đun A3 (Normal-Val Calibrated Synergistic Fusion & Tail-Fraction Scoring):**
   - Hiệu chuẩn Z-score chỉnh lưu trên $20\%$ tập `val/good`: $\widehat{M}_s^k = \max(0, (M_s^k - \mu_s^k)/\sigma_s^k)$ với sàn $\sigma_s^k \leftarrow \max(\sigma_s^k, 10^{-3})$.
   - Hợp nhất cộng hưởng:
     $$M_{\text{fuse}}^k = \underbrace{\alpha_k \widehat{M}_{t-d}^k + (1-\alpha_k) \widehat{M}_{t-e}^k}_{\text{Hợp tuyến tính bổ trợ (Soft-OR)}} + \underbrace{\eta \sqrt{\widehat{M}_{t-d}^k \cdot \widehat{M}_{t-e}^k + \epsilon}}_{\text{Khuếch đại đồng thuận nhân tính (Soft-AND)}}$$
     với $\alpha_k = (0.4, 0.6, 0.8)$, $\eta = 0.3$.
   - Điểm số cấp ảnh Tail-Fraction: Trung bình của Top-$0.1\%$ điểm ảnh ($\rho_{\text{tail}} = 0.001 \implies 65$ pixels trên ảnh $256\times 256$).

---

## 6. Bảng Độ Phức Tạp Đã Đối Soát Hoàn Chỉnh (Reconciled Complexity Master Table)

Bảng dưới đây thống kê chính xác số lượng tham số, tham số đệm BatchNorm, và số phép toán đo bằng PyTorch `torch.utils.flop_counter.FlopCounterMode` tại độ phân giải $256 \times 256$, `batch_size = 1`:

| Mã Cấu Hình | Mô Tả Kỹ Thuật | **Total Params** | **Trainable Params** | **Active Inf Params** | **BN Buffers** | **Train FLOPs** | **Inf FLOPs** | **Inf MACs** | **% Giảm Trainable vs B0-Clean** | **% Giảm Inf FLOPs vs B0-Clean** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B0-Paper`** | Số liệu công bố Table 6 | `39.300 M` | *Chưa công bố* | `39.300 M` | *Không rõ* | *Không rõ* | `6.170 G`* | `6.170 G`* | — | — |
| **`B0-Clean`** | **Canonical: +F3 1x1, Bilinear $S_d$** | **`12.180 M`** | **`9.397 M`** | **`12.180 M`** | `16,307` | **`13.485 G`** | **`13.485 G`** | **`6.742 G`** | **`0.0%` (Chuẩn)** | **`0.0%` (Chuẩn)** |
| `B0-Deconv` | Mốc so sánh: +F3 1x1, Deconv $S_d$ | `12.098 M` | `9.315 M` | `12.098 M` | `16,307` | `12.411 G` | `12.411 G` | `6.206 G`* | `-0.9%` | `-8.0%` |
| `B0-NoF3` | Mốc so sánh: no-F3 1x1, Bilinear $S_d$ | `12.114 M` | `9.331 M` | `12.114 M` | `15,794` | `13.451 G` | `13.451 G` | `6.726 G` | `-0.7%` | `-0.2%` |
| **`B1-db`** | `Lite-DFE` (`dsmsa_db`, $\alpha=1.0$) | `9.032 M` | `6.250 M` | `9.032 M` | `15,414` | `11.761 G` | `11.761 G` | `5.881 G` | **`-33.5%`** | **`-12.8%`** |
| `B1-expand` | `Lite-DFE` (`dsmsa_expand`, $\alpha=1.0$) | `9.300 M` | `6.518 M` | `9.300 M` | `16,438` | `11.911 G` | `11.911 G` | `5.955 G` | `-30.6%` | `-11.7%` |
| **`B2-0.75`** | `Slim-DSKD` ($\alpha=0.75$, DenseDFE) | `9.641 M` | `6.859 M` | `9.576 M` | `15,993` | `10.182 G` | `10.107 G` | `5.053 G` | **`-27.0%`** | **`-25.1%`** |
| **`B3-0.75`** | **Canonical: `dsmsa_db`, $\alpha=0.75, \rho=0.5$** | **`6.433 M`** | **`3.651 M`** | **`6.368 M`** | `14,492` | **`8.428 G`** | **`8.352 G`** | **`4.176 G`** | **`-61.2%`** | **`-38.1%`** |
| `B3-0.75-ex` | Ablation: `dsmsa_expand`, $\alpha=0.75, \rho=0.5$ | `6.584 M` | `3.802 M` | `6.519 M` | `15,260` | `8.512 G` | `8.436 G` | `4.218 G` | `-59.5%` | `-37.4%` |
| **`B3-0.50`** | **Canonical: `dsmsa_db`, $\alpha=0.50, \rho=1.0$** | **`4.454 M`** | **`1.671 M`** | **`4.410 M`** | `12,028` | **`5.874 G`** | **`5.824 G`** | **`2.912 G`** | **`-82.2%`** | **`-56.8%`** |
| `B3-0.50-ex` | Ablation: `dsmsa_expand`, $\alpha=0.50, \rho=1.0$ | `4.540 M` | `1.757 M` | `4.496 M` | `12,540` | `5.921 G` | `5.871 G` | `2.935 G` | `-81.3%` | `-56.5%` |
| `B3-0.50-r05`| Canonical: `dsmsa_db`, $\alpha=0.50, \rho=0.5$ | `4.439 M` | `1.656 M` | `4.395 M` | `11,772` | `5.867 G` | `5.816 G` | `2.908 G` | `-82.4%` | `-56.9%` |
| **`C1-0.75`** | **`FC-Lite-DSKD` (`dsmsa_db`, $\alpha=0.75$)** | **`6.433 M`** | **`3.651 M`** | **`6.433 M`** | `14,492` | **`8.428 G`** | **`8.428 G`** | **`4.214 G`** | **`-61.2%`** | **`-37.5%`** |
| **`C1-0.50`** | **`FC-Lite-DSKD` (`dsmsa_db`, $\alpha=0.50, \rho=1.0$)** | **`4.454 M`** | **`1.671 M`** | **`4.454 M`** | `12,028` | **`5.874 G`** | **`5.874 G`** | **`2.937 G`** | **`-82.2%`** | **`-56.4%`** |

*(Ghi chú: Cột `*` đối chiếu số liệu Table 6 của bài báo gốc với đơn vị MACs. Trong pha suy luận độc lập của Method B, đầu chiếu $\mathcal{P}_e^k$ được tỉa bỏ; khi chạy kết hợp với Method A trong `C1`, cả 2 đầu chiếu đều tham gia suy luận).*

---

## 7. Giao Thức Thí Nghiệm Sơ Bộ & Quản Trị Ngân Sách Kaggle GPU

### 7.1. Thí Nghiệm Đo Chuẩn Thời Gian Thực Tế (`M0_bench` — 10 Phút GPU)
Trước khi khởi chạy bất kỳ ma trận lớn nào, nhóm thực hiện phép thử kiểm chuẩn thời gian **`M0_bench`** trên 1 category (`carpet`, 20 epochs trên GPU Tesla T4):
1. Đo thời gian thực hiện 1 batch ($16$ ảnh) forward + backward trên kiến trúc `B0-Clean`.
2. Đo thời gian ghi nguyên tử checkpoint `.pth` dung lượng $\sim 105\text{ MB}$.
3. Đo thời gian chạy hàm `compute_aupro` trên CPU Kaggle với $500$ ngưỡng trên cả hai mức $\text{max\_fpr} = 0.30$ và $0.05$.
4. Hiệu chỉnh lại hệ số thời gian thực tế $T_{\text{cat}}$ trước khi chạy đại trà.

### 7.2. Phân Định Thí Nghiệm Tiền Thực Nghiệm Group 0
- **`S0_eq3_norm` (Kiểm chứng Công thức Chuẩn hóa Đặc trưng `Eq. 3`):**
  - Chạy 50 epochs trên `carpet` và `bottle` cho 3 biến thể: `N1 (Unit-L2)`, `N2 (Raw-L2 + Cosine)`, `N3 (Literal Inv-Sq L2)`.
  - *Mục tiêu:* Kiểm tra tính ổn định số học, loại trừ nguy cơ bùng nổ gradient và chọn công thức tối ưu cho `B0-Clean`.
- **`S0_input_norm` (Kiểm chứng Chuẩn hóa Ảnh Đầu vào Mục 4.1.2):**
  - Chạy 50 epochs trên `carpet` và `bottle` so sánh `P_imagenet` (ImageNet Z-score chuẩn) vs. `P_img_l2` (chuẩn hóa $\ell_2$ trên ảnh).
- **`S0_split` (Kiểm chứng Độ nhạy Cỡ mẫu Chia tập `80/20`):**
  - Chạy đủ 200 epochs trên `toothbrush` (danh mục nhỏ nhất: 48 train / 12 val) và `tile` so sánh `80/20 split` với `100% train/good`.

---

## 8. Cấu Trúc Mã Nguồn Dự Kiến Trong `src/anomaly_detection/`

Toàn bộ mã nguồn sẽ được triển khai theo cấu trúc module hóa phân tầng:
```text
src/anomaly_detection/
├── data/
│   └── mvtec.py                 # [ĐÃ CÓ] MVTecTrainDataset (80/20 split, seed=42), MVTecTestDataset
├── evaluation/
│   └── metrics.py               # [ĐÃ CÓ] compute_auroc_and_ap, compute_aupro (0.30 & 0.05)
├── utils/
│   ├── config.py                # [ĐÃ CÓ] load_yaml_config, save_yaml_config
│   ├── seed.py                  # [ĐÃ CÓ] set_seed
│   └── profiling.py             # [CẦN NÂNG CẤP] profile_dskd_model (Total/Trainable Params, FLOPs, MACs, Buffers)
├── models/                      # [SẼ CÀI ĐẶT]
│   └── dskd.py                  # ResNet18FeatureExtractor (T, Se), DeepFeatureEmbedding (Dense/Lite),
│                                # ReversedResNet18Decoder (Sd), DSKDModel, ProjectionHead
├── losses/                      # [SẼ CÀI ĐẶT]
│   └── dskd_loss.py             # l2_normalize_features, compute_layer_anomaly_map, DSKDLoss,
│                                # FCDistillationLoss (A1: Contextual+Focal, A2: Asym Stop-Grad)
├── training/                    # [SẼ CÀI ĐẶT]
│   ├── checkpoint.py            # CheckpointManager (CHECKPOINT_SCHEMA_V2, atomic save, auto-resume)
│   └── trainer.py               # DSKDTrainer (Dual optimizers, DFE input detach, val/good monitoring)
└── inference/                   # [SẼ CÀI ĐẶT]
    └── anomaly_scorer.py        # DSKDAnomalyScorer (Gaussian sigma=4, Val Z-score calibration,
                                 # Synergistic Fusion, Tail-Fraction Top-0.1% scoring)
```

---

## 9. Danh Sách Các Tài Liệu Nghiên Cứu Đã Đồng Bộ Hóa

1. **`docs/research/final_technical_audit.md`:** Đã sửa bảng độ phức tạp theo PyTorch FLOPs/MACs, làm rõ tính chất số học bounded của $N_3$ khi $\epsilon > 0$, phân định hai biến thể `Lite-DFE` (`dsmsa_db` vs `dsmsa_expand`), và chuyển trạng thái sang `READY FOR USER APPROVAL`.
2. **`docs/research/research_decision.md`:** Đã cập nhật mẫu số chuẩn nội bộ `B0-Clean` (`12.180M` total / `9.397M` trainable / `13.485G` FLOPs / `6.742G` MACs), đồng bộ tỷ lệ giảm tham số của Method B (`-82.2%` trainable, `-56.8%` FLOPs), bổ sung `M0_bench`, loại bỏ các điều khoản chốt cứng mâu thuẫn ở mục checklist, và chuyển trạng thái sang `READY FOR USER APPROVAL`.
3. **`docs/research/method_b_proposal.md`:** Đã sửa bảng độ phức tạp Mục 4.3 với các số liệu kiểm toán độc lập, làm rõ `dsmsa_db` là canonical và `dsmsa_expand` là ablation, sửa lại số tham số của $S_d$ ($3.326\text{M}$ params).
4. **`docs/research/method_comparison.md`:** Đã đồng bộ số liệu tham số và FLOPs của các ứng viên Method B và cập nhật mẫu số so sánh `B0-Clean`.
5. **`docs/research/method_a_proposal.md`:** Đã đồng bộ tỷ lệ chia tập `train/val` là `80/20` (`val_split_ratio: 0.2, seed: 42`), làm rõ ràng buộc đồng thuận bất đối xứng $A2$ ($\text{sg}[\hat{F}_e] \to \hat{F}_d$, $\beta=0.15$), và ghi chú phụ phí FLOPs `+0.16%` là `[NOT VERIFIED]`.
6. **`docs/research/kaggle_implementation_plan.md`:** Đã cập nhật dung lượng bộ nhớ tĩnh thực tế cho `B0-Clean` ($\sim 161.5\text{ MB}$), bổ sung bước benchmark `M0_bench`, đưa `S0_split` vào lịch trình, sửa lỗi ghi chú nhầm Mô-đun A3 thành B2.
7. **`docs/research/project_timeline.md`:** Đã đưa `M0_bench` vào Ngày 9, cập nhật số tham số và FLOPs của Method B trong tiêu chí nghiệm thu.
8. **`docs/research/experiment_design.md`:** Đã đưa `M0_bench` vào bảng ma trận tổng, chuẩn hóa quy ước báo cáo độ phức tạp (bổ sung BN buffers).

---

## 10. Phán Quyết Sẵn Sàng (Implementation Readiness Verdict)

> **PHÁN QUYẾT CHÍNH THỨC: `READY FOR USER APPROVAL`**
>
> Toàn bộ 4 điểm nghẽn kỹ thuật đã được phân tích và giải quyết triệt để trên cơ sở toán học và thực nghiệm. Mọi tài liệu nghiên cứu đã được đồng bộ hóa nhất quán 100%. Nhóm nghiên cứu **dừng tại đây** theo đúng giới hạn nghiêm ngặt, bảo toàn toàn vẹn trạng thái Git, và sẵn sàng chuyển sang giai đoạn viết mã nguồn (`src/anomaly_detection/`) ngay khi nhận được sự phê duyệt chính thức từ người dùng.
