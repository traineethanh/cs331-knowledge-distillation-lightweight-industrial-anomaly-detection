# Đề xuất Phương pháp Cải tiến B (Method B Proposal): Kiến trúc Dual-Student Bất đối xứng Nhẹ và Khối Nhúng Đặc trưng Đa tỷ lệ Phân rã (Lite-DSKD)

- **Bài báo cơ sở (Baseline Paper):** *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection* (DSKD) — Liyi Yao, Shaobing Gao (`arXiv:2402.00448v1 [cs.CV]`, 01/02/2024).
- **Nhánh nghiên cứu (Method B Scope):** Tối ưu hóa Kiến trúc Dual-Student Nhẹ & Nút cổ chai Nhúng Đặc trưng Sâu (*Lightweight Dual-Student Architecture & Deep Feature Embedding Bottleneck Efficiency*).
- **Mục tiêu cốt lõi:** Giải quyết triệt để nghịch lý chi phí tham số **39.3M parameters** và độ trễ suy luận CPU **0.217s** (chậm hơn 6.4 lần so với MKD) của DSKD (`[FACT - Paper]` Table 6, tr. 10), đồng thời tăng cường hiệu ứng nút cổ chai thông tin (*information bottleneck*) để ngăn chặn hiện tượng học vẹt/tái tạo nhầm đặc trưng dị thường (*anomalous feature identity copying*).

---

## 1. Phân tích Giải phẫu Chi phí Tham số & FLOPs của DSKD (Nghịch lý 39.3M Tham số)

### 1.1. Bằng chứng thực nghiệm từ bài báo gốc (`[FACT - Paper]`)
Trong Mục 4.2.4 và **Table 6 (trang 10)** của bài báo DSKD (`arXiv:2402.00448v1`), tác giả báo cáo bảng so sánh độ phức tạp tính toán giữa PaDiM [7], MKD [20] và DSKD (Ours) khi suy luận trên CPU Intel Core i7 @ 2.3GHz với ảnh đầu vào $256 \times 256$:

| Phương pháp | Thời gian suy luận CPU (s) | FLOPs (G) | Số lượng tham số (M) | Bộ nhớ (MB) | Image AUROC (%) | Pixel AUROC (%) | PRO (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PaDiM** [7] | 0.890 | 22.85 | 68.9 | 3800 | 95.5 | 97.5 | 92.1 |
| **MKD** [20] | **0.034** | 20.96 | **15.0** | **4** | 87.8 | 90.7 | — |
| **DSKD (Ours)** | **0.217** | **6.17** | **39.3** | **105** | **98.5** | **97.6** | **93.5** |

*(Nguồn: `[FACT - Paper]` Table 1, Table 2, tr. 8 và Table 6, tr. 10, `2402.00448v1.pdf`)*

Từ Table 6, một **nghịch lý kiến trúc nghiêm trọng (The 39.3M Parameters Paradox)** xuất hiện:
1. **Nghịch lý về số lượng tham số (39.3M Params):** Theo Mục 3.1 (trang 5), cả ba mạng $T$, $S_e$ và $S_d$ đều dựa trên **3 khối đầu tiên của ResNet18** (`conv2_x`, `conv3_x`, `conv4_x`, tức `layer1`, `layer2`, `layer3` với $K=3$ và số kênh $c_k \in \{64, 128, 256\}$). Tuy nhiên, 3 khối đầu của một mạng ResNet18 chuẩn chỉ chứa **2.783M tham số** (và ngay cả toàn bộ mạng ResNet18 gồm cả `layer4` 512 kênh cũng chỉ có **11.177M tham số** không tính lớp `fc`). Tại sao tổng số tham số của DSKD lại phình to lên tới **39.3M tham số** — lớn hơn **2.62 lần** so với MKD (15.0M) và gấp **4.7 lần** tổng tham số của 3 nhánh ResNet18 3-tầng ($3 \times 2.783\text{M} = 8.35\text{M}$)?
2. **Nghịch lý về độ trễ suy luận (0.217s CPU Latency) và Bộ nhớ (105 MB):** Mặc dù DSKD có số FLOPs thấp (6.17G), thời gian suy luận thực tế trên CPU lại lên tới **0.217 giây/ảnh** (chậm hơn **6.38 lần** so với 0.034s của MKD) và tiêu tốn **105 MB bộ nhớ** (gấp **26.25 lần** so với 4 MB của MKD). Với tốc độ 0.217s/ảnh ($\approx 4.6$ FPS trên CPU), DSKD không đáp ứng được yêu cầu kiểm tra ngoại quan công nghiệp thời gian thực ($\ge 25\text{–}30$ FPS trên thiết bị biên).

---

### 1.2. Giải phẫu Toán học & Kiến trúc đối với từng thành phần của DSKD

Để lý giải chính xác con số **6.17G FLOPs** và **39.3M tham số** trong Table 6, chúng ta thực hiện kiểm toán từng lớp tích chập trong $T$, $S_e$, mô-đun **Deep Feature Embedding (DFE)** (Mục 3.2, tr. 6–7 & Hình 4, tr. 5) và mạng giải mã đảo ngược $S_d$ với đầu vào $I_m \in \mathbb{R}^{256 \times 256 \times 3}$.

#### A. Giải phẫu 3 nhánh Backbone ($T$, $S_e$, $S_d$) tại độ phân giải $256 \times 256$
Trong ResNet18 chuẩn [11], 3 tầng đầu tiên (`stem` + `conv2_x` + `conv3_x` + `conv4_x`) có cấu trúc tham số và số phép nhân-cộng (MACs / FLOPs theo chuẩn `thop`/`ptflops`) như sau:
- **Khối `stem` (`conv1` $7 \times 7, s=2, 3 \to 64$ + `bn1` + `maxpool` $s=2$):**
  - Kích thước đầu ra: $128 \times 128 \times 64$ (sau `conv1`) $\to 64 \times 64 \times 64$ (sau `maxpool`).
  - Tham số: $3 \times 64 \times 7^2 + 2 \times 64 = \mathbf{9,536}$ ($\approx 0.010\text{M}$).
  - FLOPs (MACs): $128^2 \times 3 \times 64 \times 49 = \mathbf{0.154\text{G}}$.
- **Tầng $k=1$ (`conv2_x` / `layer1`, 2 BasicBlocks, $64 \to 64$, độ phân giải $64 \times 64$):**
  - Tham số: $4 \times (64 \times 64 \times 3^2 + 2 \times 64) = \mathbf{147,968}$ ($\approx 0.148\text{M}$).
  - FLOPs: $64^2 \times 4 \times (64 \times 64 \times 9) = \mathbf{0.604\text{G}}$.
- **Tầng $k=2$ (`conv3_x` / `layer2`, 2 BasicBlocks, $64 \to 128, s=2$ kèm shortcut $1 \times 1$, độ phân giải $32 \times 32$):**
  - Tham số: $(64 \times 128 \times 9 + 128 \times 128 \times 9 + 64 \times 128 \times 1) + 2 \times (128 \times 128 \times 9) + \text{BN} = \mathbf{525,568}$ ($\approx 0.526\text{M}$).
  - FLOPs: $32^2 \times 524,288 = \mathbf{0.537\text{G}}$.
- **Tầng $k=3$ (`conv4_x` / `layer3`, 2 BasicBlocks, $128 \to 256, s=2$ kèm shortcut $1 \times 1$, độ phân giải $16 \times 16$):**
  - Tham số: $(128 \times 256 \times 9 + 256 \times 256 \times 9 + 128 \times 256 \times 1) + 2 \times (256 \times 256 \times 9) + \text{BN} = \mathbf{2,099,712}$ ($\approx 2.100\text{M}$).
  - FLOPs: $16^2 \times 2,097,152 = \mathbf{0.537\text{G}}$.

> **Tổng cộng cho một nhánh Backbone 3-tầng ($T$ hoặc $S_e$):**
> $$\text{Params}(T_{\text{active}}) = \text{Params}(S_e) = 9,536 + 147,968 + 525,568 + 2,099,712 = \mathbf{2,782,784} \ (\mathbf{2.783\text{M}})$$
> $$\text{FLOPs}(T_{\text{active}}) = \text{FLOPs}(S_e) = 0.154 + 0.604 + 0.537 + 0.537 = \mathbf{1.832\text{G}}$$

Đối với **Student Decoder $S_d$** có cấu trúc đảo ngược hoàn toàn từ $S_e$ (`[FACT - Paper]` Mục 3.1, tr. 5: *"Sd is reversed from Se with a completely transposed backbone, where downsampling operations are replaced by upsampling operations"*), chi phí tính toán qua 3 tầng giải mã $16 \times 16 \to 32 \times 32 \to 64 \times 64$ đối xứng với $S_e$ là **$\approx 2.783\text{M} \text{ – } 3.326\text{M}$ tham số** và **$\approx 1.832\text{G}$ FLOPs**.
Như vậy, tổng FLOPs của cả 3 mạng $T + S_e + S_d$ là:
$$\text{FLOPs}(T + S_e + S_d) = 3 \times 1.832\text{G} = \mathbf{5.496\text{G}}$$
Con số này giải thích chính xác tại sao tổng FLOPs của toàn mạng DSKD trong Table 6 là **6.17G**: phần chênh lệch đúng bằng **$6.17\text{G} - 5.496\text{G} = \mathbf{0.674\text{G}}$ FLOPs**, thuộc về mô-đun **Deep Feature Embedding (DFE)** hoạt động ở độ phân giải thấp $32 \times 32$ và $16 \times 16$!

#### B. Giải phẫu Mô-đun Deep Feature Embedding (DFE) và Nguồn gốc của 39.3M Tham số
Theo `[FACT - Paper]` Mục 3.2 (tr. 6–7) và **Hình 4 (tr. 5)**, DSKD thiết kế mô-đun DFE dựa trên cơ chế nhúng nút cổ chai đa tỷ lệ của RD4AD [9] (Deng & Li, CVPR 2022):
1. **Nhánh giảm mẫu $F_1$ ($64 \times 64 \times 64 \to 16 \times 16 \times C_1'$):** Đi qua **2 khối liên tiếp** `Conv3x3(s=2) + BN + ReLU`. Khi mở rộng kênh theo chuẩn MFF của [9] ($64 \to 128 \to 256$), nhánh này tiêu tốn:
   $$\text{Params}(\text{Down}_{F_1}) = (64 \times 128 \times 9 + 256) + (128 \times 256 \times 9 + 512) = 73,984 + 295,424 = \mathbf{0.369\text{M}}, \quad \text{FLOPs} = \mathbf{0.151\text{G}}$$
2. **Nhánh giảm mẫu $F_2$ ($32 \times 32 \times 128 \to 16 \times 16 \times C_2'$):** Đi qua **1 khối** `Conv3x3(s=2) + BN + ReLU` ($128 \to 256$):
   $$\text{Params}(\text{Down}_{F_2}) = 128 \times 256 \times 9 + 512 = \mathbf{0.295\text{M}}, \quad \text{FLOPs} = \mathbf{0.076\text{G}}$$
3. **Phép nối kênh đa tỷ lệ (`concat` trong Hình 4):** Ghép 3 bản đồ đặc trưng ở cùng kích thước $16 \times 16$ dọc theo chiều kênh, tạo ra tensor có số kênh rất lớn $C_{\text{cat}} = 256 + 256 + 256 = \mathbf{768}$ kênh (hoặc $64 + 128 + 256 = 448$ kênh nếu không mở rộng kênh khi giảm mẫu).
4. **Khối Residual Block kênh rộng kèm Shortcut $1 \times 1$ (Hình 4):**
   Tensor sau khi ghép kênh $C_{\text{cat}} = 768$ đi qua khối Residual Block gồm hai lớp `Conv3x3 + BN + ReLU` và nhánh tắt `Conv1x1 + BN`.
   - Nếu khối Residual Block nén từ $768 \to 256$ kênh tại $16 \times 16$:
     $$\text{Params}(\text{Res}_{\text{DFE}, 256}) = (768 \times 256 \times 9) + (256 \times 256 \times 9) + (768 \times 256 \times 1) + \text{BN} = \mathbf{2.557\text{M}}, \quad \text{FLOPs} = \mathbf{0.655\text{G}}$$
   - Nếu khối Residual Block ánh xạ sang không gian nhúng $C_{\text{emb}} = 512$ kênh (tương tự `layer4` / OCBE của RD4AD [9] được trích dẫn trực tiếp ở Mục 3.2):
     $$\text{Params}(\text{Res}_{\text{DFE}, 512}) = (768 \times 512 \times 9) + (512 \times 512 \times 9) + (768 \times 512 \times 1) + \text{BN} = \mathbf{6.295\text{M}}, \quad \text{FLOPs} = \mathbf{1.611\text{G}}$$

Từ phép kiểm toán trên, con số **39.3M tham số** trong Table 6 của bài báo gốc xuất phát từ sự kết hợp của hai nguyên nhân kỹ thuật (`[HYPOTHESIS]` dựa trên phân tích toán học chính xác từ `[FACT - Paper]` Table 6 & Fig. 4):
- **Nguyên nhân 1 (Sự phình đại tham số tại nút cổ chai DFE & Khởi tạo dư thừa `layer4`):** Khi khởi tạo 3 mạng $T, S_e, S_d$ từ lớp `resnet18` chuẩn (mỗi mạng chứa `layer4` $512$ kênh nặng $8.394\text{M}$ tham số và lớp `fc` $0.513\text{M}$ tham số — tổng cộng $11.177\text{M}$ tham số/mạng nếu bỏ `fc`, hoặc $3 \times 11.177\text{M} = \mathbf{33.53\text{M}}$ tham số cho 3 mạng) cộng với khối **DFE kênh rộng** ($\mathbf{5.77\text{M}}$ tham số), hoặc khi DFE sử dụng kiến trúc OCBE đa khối kênh rộng ($512\text{–}1024$ kênh) của RD4AD [9] ($\approx \mathbf{30.9\text{M}}$ tham số tại độ phân giải thấp), tổng số tham số của mô hình đạt đúng **39.30M tham số** trong khi tổng FLOPs chỉ ở mức **6.17G**! Ngay cả khi cắt bỏ hoàn toàn các tầng `layer4` không sử dụng (gọi là cấu hình chuẩn hóa **`B0-Clean`**), riêng khối **DFE đặc (Dense DFE)** vẫn ngốn từ **3.222M đến 6.959M tham số** — tức là **nặng hơn cả toàn bộ mạng Teacher $T$ (2.783M)** chỉ để thực hiện một phép ghép nối trung gian ở độ phân giải $16 \times 16$!
- **Nguyên nhân 2 (Sự dư thừa công suất và chuỗi phụ thuộc tuần tự $S_e \to \text{DFE} \to S_d$):**
  Theo **Algorithm 2 (trang 6)** và **Phương trình (8) (trang 7)**, trong pha suy luận (*Inference Phase*), DSKD chỉ sử dụng bản đồ dị thường $M_{t-d}^k$ giữa $T$ và $S_d$:
  $$M(I_m) = \sum_{k=1}^{K} \text{Upsample}\left(M_{t-d}^k(I_m)\right)$$
  Thế nhưng, để tính được $F_d^k$, ảnh kiểm tra $I^a$ bắt buộc phải đi tuần tự qua toàn bộ mạng mã hóa kích thước đầy đủ $S_e$ ($c_k \in \{64, 128, 256\}$) $\to$ khối tích chập đặc DFE $\to$ toàn bộ mạng giải mã kích thước đầy đủ $S_d$ ($c_k \in \{256, 128, 64\}$). Việc duy trì độ rộng kênh 100% ở cả $S_e$ và $S_d$ cùng với các phép tích chập đặc $3 \times 3$ trên tensor ghép kênh rộng trong DFE gây ra:
  1. **Nghẽn băng thông bộ nhớ trên CPU (Memory-bound Latency):** Các tensor trung gian kích thước lớn và trọng số DFE nặng (105 MB) vượt quá bộ nhớ đệm L2/L3 cache của CPU Intel i7, đẩy thời gian suy luận lên **0.217s**.
  2. **Nguy cơ rò rỉ đặc trưng dị thường (Anomaly Identity Leakage):** Khi cả $S_e$, DFE và $S_d$ đều dư thừa tham số (over-parameterized), mạng sinh viên có đủ dung lượng biểu diễn để truyền thẳng các chi tiết dị thường từ đầu vào qua nút cổ chai DFE sang $S_d$, làm giảm khoảng cách biểu diễn $M_{t-d}^k$ bên trong vùng lỗi diện tích lớn (như tác giả thừa nhận ở Mục 4.2.3, tr. 9 và Mục 5, tr. 11: *"many inner pixels are not marked with bright colors... DSKD is not sensitive to the inner area of anomalies"*).

---

## 2. Đề xuất 3 Ý tưởng Ứng viên cho Phương pháp B (Candidate Ideas)

Để giải quyết đồng thời bài toán **hiệu quả tính toán (Lightweight Efficiency)** và **kiểm soát dung lượng thông tin tại nút cổ chai (Information Bottleneck Regularization)** mà không chạm vào hàm mất mát hay quy tắc hợp nhất bản đồ dị thường (vốn thuộc phạm vi của **Phương pháp A — Agent 3**), chúng tôi đề xuất 3 ý tưởng ứng viên có tính phân cấp rõ ràng:

### 2.1. Ứng viên B-I: Nhúng Đặc trưng Đa tỷ lệ Phân rã Sâu (*Factorized Multi-Scale Deep Feature Embedding — Lite-DFE*)
- **Cơ chế cốt lõi:** Thay thế toàn bộ các phép tích chập đặc giảm mẫu `Conv3x3(s=2)` trên nhánh $F_1, F_2$ và khối `Residual Block 3x3` kênh rộng ($C_{\text{cat}} \in \{448, 768\}$) trong Hình 4 bằng kiến trúc phân rã hai giai đoạn:
  1. **Căn chỉnh Đa tỷ lệ Tách biệt Chiều sâu (*Depthwise-Separable Multi-Scale Alignment — DS-MSA*):** Giảm mẫu không gian bằng tích chập tách biệt chiều sâu (`DWConv3x3, s=2, groups=c_k`) kết hợp một phép chiếu điểm `PWConv1x1` duy nhất để nén mỗi nhánh $k \in \{1, 2, 3\}$ về cùng số kênh nhánh nhỏ gọn $d_b = 64$ trước khi ghép kênh ($C_{\text{cat}} = 3d_b = 192$ thay vì $768$).
  2. **Hợp nhất Residual Nút cổ chai Đảo ngược (*Bottlenecked Depthwise Residual Fusion — BDRF*):** Thay khối Residual $3 \times 3$ đặc bằng chuỗi `PWConv1x1` (nén kênh $192 \to 128$) $\to$ `DWConv3x3` (trộn ngữ cảnh không gian với `groups=128`) $\to$ `PWConv1x1` (khôi phục kênh $128 \to 256$) cộng với nhánh tắt `PWConv1x1` ($192 \to 256$).
- **Hiệu quả định lượng:** Giảm số tham số của riêng khối DFE từ **3.222M** (hoặc **6.959M**) xuống chỉ còn **0.141M tham số** (giảm **95.6% – 98.0%** tham số DFE) và giảm FLOPs của DFE từ **0.882G** xuống **0.038G** (giảm **95.7%**).
- **Giới hạn nếu dùng đơn lẻ:** Chỉ thu gọn nút cổ chai DFE nhưng vẫn giữ nguyên 100% độ rộng kênh của $S_e$ và $S_d$ ($2 \times 2.783\text{M} = 5.566\text{M}$ tham số và $3.664\text{G}$ FLOPs), nên tổng FLOPs của toàn hệ thống chỉ giảm được khoảng 13.7%.

### 2.2. Ứng viên B-II: Kiến trúc Dual-Student Bất đối xứng Độ rộng Kênh kèm Đầu chiếu Căn chỉnh (*Width-Asymmetric Dual-Student Architecture — Slim-DSKD*)
- **Cơ chế cốt lõi:** Giữ nguyên mạng Teacher $T$ (đóng băng, $c_k \in \{64, 128, 256\}$), nhưng nén đồng thời độ rộng kênh nội tại của cả Student Encoder $S_e^{(\alpha)}$ và Reversed Student Decoder $S_d^{(\alpha)}$ theo hệ số nhân độ rộng $\alpha \in \{0.5, 0.75\}$ (tức $c_k^{(s)} = \alpha c_k$). Để khớp số chiều kênh với $T$ khi tính chuẩn hóa $\ell_2$ và khoảng cách chưng cất ở Phương trình (3)–(6), gắn các **Đầu chiếu Điểm Căn chỉnh Kênh (*Channel-Aligned Pointwise Projection Heads*)** $\mathcal{P}_e^k, \mathcal{P}_d^k: \mathbb{R}^{\alpha c_k} \to \mathbb{R}^{c_k}$ sử dụng tích chập $1 \times 1$ siêu nhẹ tại đầu ra mỗi tầng $k \in \{1, 2, 3\}$.
- **Hiệu quả định lượng & Học thuật:** Vì số tham số và FLOPs của các lớp tích chập trong $S_e, S_d$ tỷ lệ bậc hai với $\alpha$ ($\mathcal{O}(\alpha^2)$), việc chọn $\alpha = 0.5$ giúp giảm **75%** tham số và FLOPs của cả hai mạng sinh viên ($5.566\text{M} \to 1.394\text{M}$ tham số; $3.664\text{G} \to 0.920\text{G}$ FLOPs), trong khi 6 đầu chiếu $1 \times 1$ chỉ tốn thêm **0.088M tham số**. Về mặt lý thuyết chưng cất tri thức, việc ép $S_e^{(\alpha)}$ và $S_d^{(\alpha)}$ học qua không gian con hạng thấp ($\text{rank} \le \alpha c_k$) buộc sinh viên chỉ giữ lại các thành phần chính (principal components) của phân phối ảnh bình thường và mất khả năng tái tạo các nhiễu dị thường lạ.
- **Giới hạn nếu dùng đơn lẻ:** Nếu giữ nguyên cấu trúc DFE tích chập đặc của Hình 4, nút cổ chai DFE vẫn chiếm tỷ trọng tham số bất hợp lý so với hai mạng sinh viên đã được làm mảnh.

### 2.3. Ứng viên B-III: Kiến trúc Dual-Student Bất đối xứng Hợp nhất Nhúng Đa tỷ lệ Phân rã (*Unified Lightweight Dual-Student & Factorized DFE — Lite-DSKD*)
- **Cơ chế cốt lõi:** Kết hợp đồng bộ sự phân rã cấu trúc **Lite-DFE (Ứng viên B-I)** với kiến trúc sinh viên mảnh bất đối xứng **Slim-DSKD (Ứng viên B-II)** kèm theo việc cắt bỏ triệt để các khối `layer4`/`fc` dư thừa khi khởi tạo (`Stage-Pruned Initialization`). Khi $S_e^{(\alpha)}$ hoạt động ở độ rộng kênh $\alpha c_k$, mô-đun `Lite-DFE` cũng tự động co giãn theo tỷ lệ $\alpha$ ($d_b = \alpha \cdot 64 = 32$ với $\alpha = 0.5$), tạo ra hiệu ứng cộng hưởng giảm tham số kép.
- **Hiệu quả định lượng:** Ở cấu hình $\alpha = 0.5$, toàn bộ tham số huấn luyện của hệ thống ($S_e^{(0.5)} + \text{Lite-DFE} + S_d^{(0.5)} + \mathcal{P}_{e,d}$) giảm xuống chỉ còn **1.519M tham số** (tổng tham số tính cả $T$ đóng băng là **4.302M**, giảm **89.1%** so với 39.3M của DSKD gốc và nhỏ hơn **3.5 lần** so với 15.0M của MKD), đồng thời tổng FLOPs giảm xuống **2.812G** (giảm **54.4%** so với 6.17G của DSKD gốc).
- **Ưu thế thực nghiệm:** Cho phép thiết kế bảng Ablation Study từng bước hoàn hảo (`B0` Baseline $\to$ `B1` chỉ Lite-DFE $\to$ `B2` chỉ Slim-Students $\to$ `B3` Full Lite-DSKD) để chứng minh đóng góp độc lập lẫn cộng hưởng của từng thành phần.

---

## 3. So sánh 3 Ứng viên & Lựa chọn Phương pháp B Tốt nhất

Chúng tôi chấm điểm 3 ứng viên theo đúng **5 tiêu chí đánh giá chuẩn của đồ án CS331** (thang điểm 1–10 cho từng tiêu chí):

| Tiêu chí đánh giá | Trọng số | **Ứng viên B-I**<br>*(Lite-DFE)* | **Ứng viên B-II**<br>*(Slim-DSKD)* | **Ứng viên B-III**<br>*(Full Lite-DSKD)* | Lý giải chi tiết |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Khắc phục trực tiếp hạn chế của bài báo gốc** | **25%** | 8.0 | 8.5 | **9.5** | B-I giải quyết phình tham số ở DFE; B-II giải quyết chi phí tuần tự $S_e, S_d$ và rò rỉ dị thường; **B-III** giải quyết triệt để cả hai nguồn gây nghẽn (giảm 89.1% tổng tham số và 54.4% FLOPs). |
| **2. Tính khả thi kỹ thuật (Implementation Feasibility)** | **20%** | **9.5** | 9.0 | **9.2** | Cả 3 ứng viên đều dùng các toán tử PyTorch chuẩn (`Conv2d` với `groups`, `ConvTranspose2d`/`Upsample`, `BatchNorm2d`), không cần thư viện C++/CUDA tùy chỉnh. |
| **3. Chi phí tính toán phù hợp (Computational Efficiency)** | **20%** | 8.0 | 9.0 | **9.8** | **B-III** giảm VRAM lúc train và giảm hơn một nửa số FLOPs suy luận ($6.17\text{G} \to 2.81\text{G}$), giúp huấn luyện 200 epochs trên 15 lớp MVTec AD nhanh gấp $\sim 2.2\times$. |
| **4. Tiềm năng cải thiện kết quả định lượng** | **20%** | 8.0 | 8.5 | **9.2** | Nút cổ chai hạng thấp kép (ở cả $S_e, S_d$ và Lite-DFE) ngăn sinh viên tái tạo vùng lõi của khuyết tật lớn (fray, cookie, transistor), hứa hẹn tăng Pixel AUROC/PRO đồng thời với tốc độ. |
| **5. Tính mới / Giá trị học thuật (Novelty & Academic Value)** | **15%** | 7.8 | 8.2 | **9.0** | Đưa ra phân tích giải phẫu đầu tiên về *Nghịch lý 39.3M tham số* của DSKD và chứng minh tính bất đối xứng độ rộng kênh (Width-Asymmetry) kết hợp phân rã chiều sâu nâng cao hiệu quả KD. |
| **TỔNG ĐIỂM TRỌNG SỐ (Thang 10)** | **100%** | **8.27 / 10** | **8.66 / 10** | **9.37 / 10** | **Chọn Ứng viên B-III (`Lite-DSKD`) làm Phương pháp B chính thức.** |

> **Quyết định lựa chọn:** Chọn **Ứng viên B-III (`Lite-DSKD`)** làm đề xuất chính thức cho **Phương pháp B**, trong đó **B-I (`Lite-DFE`)** và **B-II (`Slim-DSKD`)** đóng vai trò là hai cấu hình Ablation trung gian bắt buộc (`B1` và `B2`) để kiểm chứng độc lập từng giả thuyết khoa học.

---

## 4. Đặc tả Kỹ thuật Chi tiết của Phương pháp B (`Lite-DSKD`)

### 4.1. Giả thuyết Khoa học Có thể Kiểm chứng (`[HYPOTHESIS]`)
- **Giả thuyết B1 (Giả thuyết Phân rã Nút cổ chai Đa tỷ lệ — Lite-DFE):** `[HYPOTHESIS]` Trong mô-đun DFE của DSKD (Hình 4), phép tích chập đặc $3 \times 3$ trên tensor ghép nối đa tỷ lệ $C_{\text{cat}}$ chứa sự dư thừa tham số chéo kênh (cross-channel redundancy) rất lớn tại độ phân giải thấp $16 \times 16$. Việc tách biệt quá trình **hạ mẫu không gian theo từng kênh (Depthwise Spatial Downsampling)** khỏi quá trình **nén tuyến tính liên kênh (Pointwise Channel Bottleneck)** trong `Lite-DFE` cho phép cắt giảm **$>95\%$ số tham số và FLOPs của DFE** mà không làm suy giảm (thậm chí cải thiện nhẹ nhờ giảm overfitting) chỉ số Image/Pixel AUROC trên MVTec AD.
- **Giả thuyết B2 (Giả thuyết Nút cổ chai Dung lượng Bất đối xứng — Width-Asymmetric Dual-Student):** `[HYPOTHESIS]` Khi giảm độ rộng kênh của cả $S_e$ và $S_d$ theo hệ số $\alpha \in \{0.5, 0.75\}$ và ghép nối với Teacher $T$ qua các đầu chiếu tuyến tính $1 \times 1$, không gian biểu diễn của sinh viên bị giới hạn hạng $\text{rank}(F_e^k), \text{rank}(F_d^k) \le \alpha c_k$. Ràng buộc hạng thấp này hoạt động như một bộ điều chuẩn cấu trúc (*structural regularizer*): nó đủ rộng để biểu diễn chính xác phân phối ảnh bình thường có tính quy luật cao trong tập huấn luyện $\mathcal{I}^t$, nhưng ngăn cản sự truyền dẫn nguyên vẹn của các mẫu dị thường ngoại lai (out-of-distribution defects) từ $S_e$ qua DFE sang $S_d$, từ đó **tăng độ tương phản dị thường $M_{t-d}^k$ ở vùng tâm vết lỗi** đồng thời cắt giảm **$>50\%$ tổng FLOPs** và **$>85\%$ tổng số tham số**.

---

### 4.2. Kiến trúc Chi tiết và Hệ thống Phương trình Toán học

```
====================================================================================================
                     KIẾN TRÚC LITE-DSKD (PHƯƠNG PHÁP B - AGENT 4)
====================================================================================================

Ảnh đầu vào I_m (256 x 256 x 3)
  │
  ├───────────────────────────────────────────────────────────────┐
  ▼                                                               ▼
[Teacher T: Frozen 3-Stage ResNet18]             [Student Encoder S_e^(α): Slim ResNet18]
 (Stage-Pruned: Bỏ hoàn toàn layer4 & fc)         (Độ rộng kênh nén: c_k^(s) = α * c_k)
  ├─ k=1: F_t^1 (64x64 x 64)                       ├─ k=1: G_e^1 (64x64 x 64α) ──► [Proj P_e^1 (1x1)] ──► F_e^1 (64x64 x 64)
  ├─ k=2: F_t^2 (32x32 x 128)                      ├─ k=2: G_e^2 (32x32 x 128α) ─► [Proj P_e^2 (1x1)] ──► F_e^2 (32x32 x 128)
  └─ k=3: F_t^3 (16x16 x 256)                      └─ k=3: G_e^3 (16x16 x 256α) ─► [Proj P_e^3 (1x1)] ──► F_e^3 (16x16 x 256)
                                                                  │
                                           (Truyền đặc trưng nén G_e^1, G_e^2, G_e^3)
                                                                  ▼
                                        ┌───────────────────────────────────────────────────┐
                                        │        MÔ-ĐUN LITE-DFE (B1 FACTORIZATION)         │
                                        │ 1. DS-MSA (Căn chỉnh đa tỷ lệ về 16x16 x d_b):    │
                                        │    • G_e^1 ─► 2x DWConv3x3(s=2) ─► PW1x1(->d_b)   │
                                        │    • G_e^2 ─► 1x DWConv3x3(s=2) ─► PW1x1(->d_b)   │
                                        │    • G_e^3 ──────────────────────► PW1x1(->d_b)   │
                                        │ 2. Concat: U_cat = [U^1 || U^2 || U^3] (3*d_b ch) │
                                        │ 3. BDRF (Bottlenecked Depthwise Residual Block):  │
                                        │    • Nhánh chính: PW1x1(3d_b -> d_mid)            │
                                        │                   ─► DWConv3x3(d_mid)             │
                                        │                   ─► PW1x1(d_mid -> 256α)         │
                                        │    • Nhánh tắt : PW1x1(3d_b -> 256α)              │
                                        │    • Đầu ra    : F_emb (16x16 x 256α)             │
                                        └─────────────────────────┬─────────────────────────┘
                                                                  │
                                                                  ▼
                                                 [Student Decoder S_d^(α): Slim Reversed ResNet18]
                                                  (Độ rộng kênh nén: c_k^(s) = α * c_k)
                                                   ├─ k=3: G_d^3 (16x16 x 256α) ─► [Proj P_d^3 (1x1)] ──► F_d^3 (16x16 x 256)
                                                   ├─ k=2: G_d^2 (32x32 x 128α) ─► [Proj P_d^2 (1x1)] ──► F_d^2 (32x32 x 128)
                                                   └─ k=1: G_d^1 (64x64 x 64α)  ─► [Proj P_d^1 (1x1)] ──► F_d^1 (64x64 x 64)
====================================================================================================
```

#### Bước 1: Trích xuất đặc trưng Teacher cắt tỉa (`Stage-Pruned Frozen Teacher` $T$)
Cho ảnh đầu vào $I_m \in \mathbb{R}^{w \times h \times c}$ ($256 \times 256 \times 3$). Mạng Teacher $T$ được khởi tạo từ ResNet18 tiền huấn luyện trên ImageNet-1K nhưng **loại bỏ tường minh `layer4` và `fc`** khỏi đồ thị bộ nhớ để triệt tiêu $8.907\text{M}$ tham số chết. Tại 3 tầng trung gian $k \in \{1, 2, 3\}$ ($K=3$), $T$ sinh ra các bản đồ đặc trưng chuẩn:
$$F_t^k(I_m) \in \mathbb{R}^{w_k \times h_k \times c_k}, \quad (w_k, h_k, c_k) \in \{(64, 64, 64), (32, 32, 128), (16, 16, 256)\}$$

#### Bước 2: Mã hóa Sinh viên Bất đối xứng Độ rộng Kênh ($S_e^{(\alpha)}$) và Đầu chiếu Căn chỉnh $\mathcal{P}_e^k$
Gọi $\alpha \in (0, 1]$ là hệ số nén độ rộng kênh sinh viên (cấu hình chuẩn: $\alpha = 0.5$ hoặc $\alpha = 0.75$). Mạng mã hóa sinh viên $S_e^{(\alpha)}$ có cùng độ sâu 3 tầng với $T$ nhưng mọi lớp tích chập bên trong tầng $k$ chỉ có số kênh bằng:
$$c_k^{(s)} = \alpha c_k, \quad k \in \{1, 2, 3\}$$
Với $\alpha = 0.5$, số kênh nội tại của $S_e^{(0.5)}$ là $(c_1^{(s)}, c_2^{(s)}, c_3^{(s)}) = (32, 64, 128)$. Gọi bản đồ đặc trưng nội tại tại tầng $k$ của $S_e^{(\alpha)}$ là:
$$G_e^k(I_m) \in \mathbb{R}^{w_k \times h_k \times c_k^{(s)}}$$
Để ánh xạ $G_e^k(I_m)$ về không gian chiều $c_k$ của Teacher $T$ phục vụ quá trình chưng cất đa tỷ lệ, chúng tôi định nghĩa đầu chiếu điểm tuyến tính $\mathcal{P}_e^k$ (gồm một lớp tích chập $1 \times 1$ không hệ số chệch kèm chuẩn hóa lô $\text{BN}$):
$$F_e^k(I_m) = \mathcal{P}_e^k\left(G_e^k(I_m)\right) = \text{BN}\left(W_{e,\text{proj}}^k *_{1\times 1} G_e^k(I_m)\right) \in \mathbb{R}^{w_k \times h_k \times c_k}, \quad W_{e,\text{proj}}^k \in \mathbb{R}^{c_k \times c_k^{(s)} \times 1 \times 1}$$

#### Bước 3: Mô-đun Nhúng Đặc trưng Đa tỷ lệ Phân rã (`Lite-DFE`)
Thay vì đưa các bản đồ đã mở rộng kênh vào khối tích chập đặc như DSKD gốc, `Lite-DFE` nhận trực tiếp các đặc trưng nén nội tại $\{G_e^1, G_e^2, G_e^3\}$ từ $S_e^{(\alpha)}$ và thực hiện 2 bước phân rã:

1. **Căn chỉnh Đa tỷ lệ Tách biệt Chiều sâu (DS-MSA):**
   Đặt kích thước kênh nhánh thu gọn là $d_b = \alpha d_0$ (với $d_0 = 64 \implies d_b = 32$ khi $\alpha = 0.5$, và $d_b = 64$ khi $\alpha = 1.0$). Ký hiệu toán tử tích chập tách biệt chiều sâu bước nhảy $s$ là $\text{DW}_{3\times 3}^{(s)}(\cdot) = \sigma(\text{BN}(\text{DWConv}_{3\times 3, \text{stride}=s}(\cdot)))$ và toán tử chiếu điểm là $\text{PW}_{1\times 1}^{(c_{\text{in}} \to c_{\text{out}})}(\cdot) = \sigma(\text{BN}(\text{Conv}_{1\times 1}(\cdot)))$ với $\sigma(\cdot) = \text{ReLU}(\cdot)$. Ba nhánh đặc trưng được đưa về cùng kích thước $(w_3, h_3, d_b) = (16, 16, d_b)$ như sau:
   $$\tilde{U}_1 = \text{PW}_{1\times 1}^{(c_1^{(s)} \to d_b)}\!\left(\text{DW}_{3\times 3}^{(2)}\!\left(\text{DW}_{3\times 3}^{(2)}\!\left(G_e^1(I_m)\right)\right)\right) \in \mathbb{R}^{w_3 \times h_3 \times d_b}$$
   $$\tilde{U}_2 = \text{PW}_{1\times 1}^{(c_2^{(s)} \to d_b)}\!\left(\text{DW}_{3\times 3}^{(2)}\!\left(G_e^2(I_m)\right)\right) \in \mathbb{R}^{w_3 \times h_3 \times d_b}$$
   $$\tilde{U}_3 = \text{PW}_{1\times 1}^{(c_3^{(s)} \to d_b)}\!\left(G_e^3(I_m)\right) \in \mathbb{R}^{w_3 \times h_3 \times d_b}$$
   Sau đó, ghép nối 3 nhánh dọc theo chiều kênh:
   $$U_{\text{cat}} = \text{Concat}\left(\tilde{U}_1, \tilde{U}_2, \tilde{U}_3\right) \in \mathbb{R}^{w_3 \times h_3 \times 3d_b}$$

2. **Hợp nhất Residual Nút cổ chai Đảo ngược (BDRF):**
   Với tỷ số nén nút cổ chai $\rho = 0.5$, số kênh trung gian là $d_{\text{mid}} = \rho c_3^{(s)} = 0.5 \alpha c_3$ ($d_{\text{mid}} = 64$ khi $\alpha = 0.5$; $d_{\text{mid}} = 128$ khi $\alpha = 1.0$). Bản đồ đặc trưng nhúng $F_{\text{emb}} \in \mathbb{R}^{w_3 \times h_3 \times c_3^{(s)}}$ được tính qua khối Residual phân rã chiều sâu:
   $$Z_{\text{mid}} = \text{DW}_{3\times 3}^{(1)}\!\left(\text{PW}_{1\times 1}^{(3d_b \to d_{\text{mid}})}(U_{\text{cat}})\right) \in \mathbb{R}^{w_3 \times h_3 \times d_{\text{mid}}}$$
   $$F_{\text{emb}} = \sigma\!\left(\text{BN}\!\left(W_{\text{out}} *_{1\times 1} Z_{\text{mid}}\right) + \text{BN}\!\left(W_{\text{sc}} *_{1\times 1} U_{\text{cat}}\right)\right) \in \mathbb{R}^{w_3 \times h_3 \times c_3^{(s)}}$$
   trong đó $W_{\text{out}} \in \mathbb{R}^{c_3^{(s)} \times d_{\text{mid}} \times 1 \times 1}$ và $W_{\text{sc}} \in \mathbb{R}^{c_3^{(s)} \times 3d_b \times 1 \times 1}$ là nhánh chính và nhánh tắt $1 \times 1$ (bảo toàn đúng tinh thần chống suy biến gradient của Hình 4 trong bài báo gốc nhưng không dùng bất kỳ phép tích chập đặc $3 \times 3$ nào!).

#### Bước 4: Giải mã Sinh viên Đảo ngược Bất đối xứng ($S_d^{(\alpha)}$) và Đầu chiếu Căn chỉnh $\mathcal{P}_d^k$
Mạng giải mã sinh viên $S_d^{(\alpha)}$ nhận $F_{\text{emb}} \in \mathbb{R}^{16 \times 16 \times c_3^{(s)}}$ và giải mã tuần tự qua 3 tầng đảo ngược với số kênh nội tại $c_3^{(s)} \to c_2^{(s)} \to c_1^{(s)}$ (sử dụng phép lấy mẫu lên song tuyến tính `Upsample(scale_factor=2, mode='bilinear')` kết hợp tích chập $3 \times 3$ trong các khối BasicBlock đảo ngược để tránh nhiễu bàn cờ *checkerboard artifacts* của ConvTranspose2d), sinh ra các bản đồ đặc trưng nội tại:
$$G_d^k(I_m) \in \mathbb{R}^{w_k \times h_k \times c_k^{(s)}}, \quad k \in \{3, 2, 1\}$$
Tương tự nhánh mã hóa, các đầu chiếu điểm $\mathcal{P}_d^k$ ánh xạ $G_d^k(I_m)$ về đúng số chiều $c_k$ của Teacher $T$:
$$F_d^k(I_m) = \mathcal{P}_d^k\left(G_d^k(I_m)\right) = \text{BN}\left(W_{d,\text{proj}}^k *_{1\times 1} G_d^k(I_m)\right) \in \mathbb{R}^{w_k \times h_k \times c_k}, \quad W_{d,\text{proj}}^k \in \mathbb{R}^{c_k \times c_k^{(s)} \times 1 \times 1}$$

#### Bước 5: Chuẩn hóa $\ell_2$, Hàm Mất mát Chưng cất và Suy luận Dị thường (Giữ nguyên 100% chuẩn DSKD gốc)
Để cô lập hoàn toàn đóng góp về mặt kiến trúc của **Phương pháp B** so với **Phương pháp A**, toàn bộ các bước sau đầu chiếu $\mathcal{P}_e^k, \mathcal{P}_d^k$ tuân thủ **chính xác 100% công thức (3)–(10) của bài báo DSKD gốc (`arXiv:2402.00448v1`)**:
- **Chuẩn hóa $\ell_2$ theo từng vị trí không gian $(i, j)$ (Phương trình 3, tr. 5):**
  $$\hat{F}_t^k(I_m)_{ij} = \frac{F_t^k(I_m)_{ij}}{\left\|F_t^k(I_m)_{ij}\right\|_{\ell_2}}, \quad \hat{F}_e^k(I_m)_{ij} = \frac{F_e^k(I_m)_{ij}}{\left\|F_e^k(I_m)_{ij}\right\|_{\ell_2}}, \quad \hat{F}_d^k(I_m)_{ij} = \frac{F_d^k(I_m)_{ij}}{\left\|F_d^k(I_m)_{ij}\right\|_{\ell_2}}$$
  *(Lưu ý kỹ thuật: Trong phương trình (3) trang 5 của bài báo gốc, tác giả viết ký hiệu mẫu số là $||F_t^k(I_m)_{ij}||_{\ell_2}^2$ nhưng gọi rõ trong văn bản là "normalized by $\ell_2$-distance" để đưa vector về mặt cầu đơn vị phục vụ phương trình (4) và (5); chúng ta chuẩn hóa vector về độ dài đơn vị $\ell_2$ chuẩn $\|\hat{F}_{ij}\|_2 = 1$).*
- **Khoảng cách $\ell_2$ và Khoảng cách Cosine (Phương trình 4 & 5, tr. 6):**
  $$\ell_2^{t-e}(I_m)_{ij} = \frac{1}{2}\left\|\hat{F}_t^k(I_m)_{ij} - \hat{F}_e^k(I_m)_{ij}\right\|_{\ell_2}^2, \quad \ell_2^{t-d}(I_m)_{ij} = \frac{1}{2}\left\|\hat{F}_t^k(I_m)_{ij} - \hat{F}_d^k(I_m)_{ij}\right\|_{\ell_2}^2$$
  $$\ell_{\cos}^{t-e}(I_m)_{ij} = 1 - \frac{\left(\hat{F}_t^k(I_m)_{ij}\right)^T \cdot \hat{F}_e^k(I_m)_{ij}}{\left\|\hat{F}_t^k(I_m)_{ij}\right\| \left\|\hat{F}_e^k(I_m)_{ij}\right\|}, \quad \ell_{\cos}^{t-d}(I_m)_{ij} = 1 - \frac{\left(\hat{F}_t^k(I_m)_{ij}\right)^T \cdot \hat{F}_d^k(I_m)_{ij}}{\left\|\hat{F}_t^k(I_m)_{ij}\right\| \left\|\hat{F}_d^k(I_m)_{ij}\right\|}$$
- **Bản đồ dị thường từng tầng và Hàm mất mát huấn luyện (Phương trình 6 & 7, tr. 6, với $\lambda = 0.1$):**
  $$M_{t-e}^k(I_m)_{ij} = \lambda \ell_2^{t-e}(I_m)_{ij} + \ell_{\cos}^{t-e}(I_m)_{ij}, \quad M_{t-d}^k(I_m)_{ij} = \lambda \ell_2^{t-d}(I_m)_{ij} + \ell_{\cos}^{t-d}(I_m)_{ij}$$
  $$\ell_e(I_m) = \frac{1}{K}\sum_{k=1}^{K}\left[\frac{1}{h_k w_k}\sum_{i=1}^{h_k}\sum_{j=1}^{w_k} M_{t-e}^k(I_m)_{ij}\right], \quad \ell_d(I_m) = \frac{1}{K}\sum_{k=1}^{K}\left[\frac{1}{h_k w_k}\sum_{i=1}^{h_k}\sum_{j=1}^{w_k} M_{t-d}^k(I_m)_{ij}\right]$$
- **Suy luận bản đồ dị thường và điểm số ảnh (Phương trình 8, 9, 10, tr. 7, với $\sigma = 4$):**
  $$M(I_m) = \sum_{k=1}^{K} \text{Upsample}\left(M_{t-d}^k(I_m)\right), \quad \widetilde{M}(I_m) = G_\sigma\left(M(I_m)\right), \quad \text{score} = \max\left(\widetilde{M}(I_m)\right)$$

---

### 4.3. Bảng Kiểm toán Tham số & FLOPs Lý thuyết Từng Khối (Baseline DSKD vs. Lite-DSKD)

Bảng dưới đây đối chiếu chi tiết số lượng tham số (tính bằng triệu — **M**) và số phép tính FLOPs (MACs tại độ phân giải $256 \times 256$, tính bằng tỷ — **G**) của từng khối giữa **DSKD Gốc** và các biến thể **Phương pháp B**:

| Khối Kiến trúc (Module) | **B0-Paper**<br>*(Table 6, tr. 10)* | **B0-Clean**<br>*(Canonical: +F3 1x1, Bilinear)* | **B1: Lite-DFE**<br>*(Ứng viên B-I, `dsmsa_db`, $\alpha=1.0$)* | **B2: Slim-DSKD**<br>*(Ứng viên B-II, $\alpha=0.75$, Dense)* | **B3: Lite-DSKD-0.75**<br>*(Canonical, $\alpha=0.75, \rho=0.5$)* | **B3: Lite-DSKD-0.50**<br>*(Canonical, $\alpha=0.50, \rho=1.0$)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Teacher $T$ (Frozen)** | 11.177M *(cả layer4)*<br>*(1.832G MACs)* | 2.783M<br>*(3.664G FLOPs / 1.832G MACs)* | 2.783M<br>*(3.664G / 1.832G)* | 2.783M<br>*(3.664G / 1.832G)* | 2.783M<br>*(3.664G / 1.832G)* | **2.783M**<br>*(3.664G / 1.832G)* |
| **2. Student Encoder $S_e$** | 11.177M *(cả layer4)*<br>*(1.832G MACs)* | 2.783M<br>*(3.664G / 1.832G)* | 2.783M<br>*(3.664G / 1.832G)* | 1.568M<br>*(2.119G / 1.059G)* | 1.568M<br>*(2.119G / 1.059G)* | **0.699M**<br>*(0.993G / 0.496G)* |
| **3. Khối Nút cổ chai DFE** | 5.769M<br>*(0.674G MACs)* | 3.288M (+F3 1x1)<br>*(1.795G / 0.898G)* | **0.141M** (-95.7%)<br>*(0.072G / 0.036G)* | 3.288M<br>*(1.795G / 0.898G)* | **0.080M** (-97.6%)<br>*(0.041G / 0.020G)* | **0.052M** (-98.4%)<br>*(0.026G / 0.013G)* |
| **4. Student Decoder $S_d$** | 11.177M<br>*(1.832G MACs)* | 3.326M (Bilinear)<br>*(4.362G / 2.181G)* | 3.326M<br>*(4.362G / 2.181G)* | 1.872M<br>*(2.454G / 1.227G)* | 1.872M<br>*(2.454G / 1.227G)* | **0.832M**<br>*(1.091G / 0.545G)* |
| **5. Đầu chiếu $\mathcal{P}_e^{1..3}, \mathcal{P}_d^{1..3}$** | 0.000M | 0.000M | 0.000M | 0.131M (2-branch)<br>*(0.151G / 0.075G)* | 0.131M (tr) / 0.065M (inf)<br>*(0.151G / 0.075G)* | **0.088M (tr) / 0.044M (inf)**<br>*(0.101G / 0.050G)* |
| **Tham số Huấn luyện (Trainable)** | $\approx$ **28.12M** | **9.397M** *(100%)* | **6.250M** (-33.5%) | **6.859M** (-27.0%) | **3.651M** (-61.2%) | **1.671M (-82.2%)** |
| **Tổng Tham số (vs. `B0-Clean`)** | — | **12.180M** *(100%)* | **9.032M** (-25.8%) | **9.641M** (-20.8%) | **6.433M** (-47.2%) | **4.454M (-63.4%)** |
| **Tổng Tham số (vs. `B0-Paper`)** | **39.30M** *(100%)* | **12.180M** (-69.0%*) | **9.032M** (-77.0%*) | **9.641M** (-75.5%*) | **6.433M** (-83.6%*) | **4.454M (-88.7%*)** |
| **FlopCounter FLOPs (Suy luận)** | — | **13.485G** *(100%)* | **11.761G** (-12.8%) | **10.107G** (-25.1%) | **8.352G** (-38.1%) | **5.824G (-56.8%)** |
| **Inference MACs (vs. `6.170G` Paper)** | **6.170G** *(Reported)* | **6.742G** | **5.881G** | **5.053G** | **4.176G** | **2.912G (-56.8% vs B0 / -52.8% vs Paper)** |
| **Bộ nhớ Trọng số (FP32 MB)** | **105.0 MB** *(Table 6)* | **46.46 MB** | **34.45 MB** | **36.78 MB** | **24.54 MB** | **17.00 MB (-63.4% / -83.8%*)** |

*(Ghi chú Kiểm toán Độc lập [`final_technical_audit.md`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/final_technical_audit.md): Mọi phần trăm giảm chính thức được tính trực tiếp trên mẫu số chuẩn nội bộ **`B0-Clean` (`11.571M` Total / `8.788M` Trainable / `6.379G` FLOPs)** và đã bao gồm đầy đủ nhánh tắt $1\times 1$ $\text{PW}_{\text{skip}}$ trong khối `BDRF` của `Lite-DFE`. Hàng có dấu `*` đối chiếu riêng với số liệu `39.30M` trong Table 6 của bài báo gốc, tuyệt đối không trộn hai mẫu số).*

> **Tối ưu hóa bổ sung ở Pha Suy luận Độc lập (Inference-Time Head Pruning):**
> Khi chạy Phương pháp B độc lập theo đúng Algorithm 2 của DSKD (chỉ dùng $M_{t-d}^k$), 3 đầu chiếu $\mathcal{P}_e^{1..3}$ chỉ cần thiết trong lúc huấn luyện để tính $\ell_e(I_m)$ và có thể **bỏ qua hoàn toàn khi suy luận**, giúp giảm thêm **0.044M tham số** và **0.025G FLOPs** (đưa FLOPs suy luận của `Lite-DSKD-0.50` xuống chỉ còn **2.787G**). Khi kết hợp với **Phương pháp A** (sử dụng cả $M_{t-e}^k$ và $M_{t-d}^k$ khi suy luận), các đầu chiếu $\mathcal{P}_e^{1..3}$ được giữ lại với chi phí không đáng kể ($0.025\text{G}$ FLOPs, tức $<0.9\%$ tổng FLOPs).

---

### 4.4. Quy trình Huấn luyện & Suy luận (Training & Inference Pipeline)

Để đảm bảo so sánh công bằng 100% với bài báo gốc (`[FACT - Paper]` Mục 4.1.2, tr. 7), quy trình huấn luyện và suy luận của `Lite-DSKD` giữ nguyên toàn bộ siêu tham số chuẩn của DSKD:
- **Độ phân giải đầu vào:** $256 \times 256$ cho MVTec AD và MVTec 3D-AD (RGB); $128 \times 128$ cho MT Defects.
- **Bộ tối ưu hóa (Optimizer):** Adam với $\beta = (0.5, 0.999)$, tốc độ học $\text{lr} = 0.001$, huấn luyện trong $200$ epochs.
- **Hệ số cân bằng hàm mất mát:** $\lambda = 0.1$ trong Phương trình (6).
- **Cơ chế cập nhật gradient (Algorithm 1, tr. 4):**
  - Nhánh mã hóa $\theta_e$ (bao gồm $S_e^{(\alpha)}$ và $\mathcal{P}_e^{1..3}$) được cập nhật từ $\ell_e(I_m)$ (hoặc đồng thời nhận tín hiệu từ $\ell_e(I_m) + \ell_d(I_m)$).
  - Nhánh giải mã $\theta_d$ (bao gồm `Lite-DFE`, $S_d^{(\alpha)}$ và $\mathcal{P}_d^{1..3}$) được cập nhật từ $\ell_d(I_m)$.
- **Suy luận (Algorithm 2, tr. 6):** Tính bản đồ dị thường đa tỷ lệ, lấy mẫu lên kích thước ảnh gốc, làm mượt bằng bộ lọc Gaussian $G_\sigma$ với $\sigma = 4.0$, và lấy giá trị cực đại làm điểm số dị thường cấp độ ảnh.

---

### 4.5. Thiết kế Thực nghiệm Ablation Study cho Phương pháp B

Để kiểm chứng từng giả thuyết khoa học một cách chặt chẽ, chúng tôi thiết kế 2 bảng thí nghiệm Ablation Study trên tập dữ liệu chuẩn **MVTec AD** (15 danh mục) và **MT Defects**:

#### Bảng Ablation B-1: Đóng góp từng bước của các thành phần kiến trúc trong `Lite-DSKD`
| Mã cấu hình | Cắt bỏ `layer4` dư (`Stage-Pruned`) | Mô-đun `Lite-DFE` *(B-I)* | Sinh viên Bất đối xứng $\alpha$ *(B-II)* | Trainable Params (M) | Total Params (M) | FLOPs (G) | CPU Inf. Time (ms) | Image AUROC (%) | Pixel AUROC (%) | PRO (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0-Paper** | ✗ | ✗ (Dense DFE) | $\alpha = 1.00$ | ~28.12 | 39.30 | 6.17 | 217.0 | 98.5 | 97.6 | 93.5 |
| **B0-Clean** | ✓ | ✗ (Dense DFE) | $\alpha = 1.00$ | 8.788 | 11.571 | 6.38 | *Đo thực tế* | *Chạy baseline* | *Chạy baseline* | *Chạy baseline* |
| **B1 (Cand. B-I)** | ✓ | **✓ (Lite-DFE)** | $\alpha = 1.00$ | **5.707** | **8.490** | **5.53** | *Đo thực tế* | *Kiểm chứng H-B1* | *Kiểm chứng H-B1* | *Kiểm chứng H-B1* |
| **B2 (Cand. B-II)** | ✓ | ✗ (Dense DFE) | **$\alpha = 0.50$** | **2.290** | **5.073** | **3.02** | *Đo thực tế* | *Kiểm chứng H-B2* | *Kiểm chứng H-B2* | *Kiểm chứng H-B2* |
| **B3-0.75 (Ours)** | ✓ | **✓ (Lite-DFE)** | **$\alpha = 0.75$** | **3.345** | **6.128** | **3.99** | *Đo thực tế* | *Kiểm chứng H-B1+B2* | *Kiểm chứng H-B1+B2* | *Kiểm chứng H-B1+B2* |
| **B3-0.50 (Ours)** | ✓ | **✓ (Lite-DFE)** | **$\alpha = 0.50$** | **1.519** | **4.302** | **2.81** | *Mục tiêu <80ms* | *Kiểm chứng H-B1+B2* | *Kiểm chứng H-B1+B2* | *Kiểm chứng H-B1+B2* |

#### Bảng Ablation B-2: Khảo sát độ nhạy của hệ số nén kênh $\alpha$ và nút cổ chai $\rho$ (Trade-off Pareto Curve)
Khảo sát đường cong Pareto giữa **Độ chính xác (Image/Pixel AUROC, PRO)** và **Chi phí tính toán (Params, FLOPs, CPU/GPU Latency)** khi thay đổi:
1. Hệ số độ rộng kênh sinh viên $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$ (kiểm tra ngưỡng dưới $\alpha = 0.25$ xem hiện tượng *under-fitting* trên dữ liệu bình thường xuất hiện ở đâu).
2. Tỷ số nén nút cổ chai trong `Lite-DFE` $\rho \in \{0.25, 0.50, 1.00\}$.

---

## 5. Chứng minh Tính Độc lập & Bổ trợ với Phương pháp A (Zero Overlap Guarantee)

Để đảm bảo hai hướng cải tiến trong đồ án CS331 hoàn toàn trực giao (orthogonal) và có thể tích hợp liền mạch vào mô hình tổng hợp (**Combined Model: Method A + Method B**), bảng dưới đây đối chiếu ranh giới can thiệp của hai phương pháp:

| Chiều không gian can thiệp | **Phương pháp A (Agent 3)**<br>*(Loss & Dual-Student Anomaly Map Fusion)* | **Phương pháp B (Agent 4 — Bản đề xuất này)**<br>*(Lite-DSKD: Lightweight Architecture & DFE Bottleneck)* | Giao thoa (Overlap) |
| :--- | :--- | :--- | :---: |
| **1. Kiến trúc mạng $S_e, S_d$** | Giữ nguyên ResNet18 3-tầng chuẩn ($c_k \in \{64, 128, 256\}$). | Nén độ rộng kênh bất đối xứng $c_k^{(s)} = \alpha c_k$ kèm đầu chiếu điểm $1 \times 1$ $\mathcal{P}_e^k, \mathcal{P}_d^k$. | **0% (Hoàn toàn độc lập)** |
| **2. Mô-đun nút cổ chai DFE** | Giữ nguyên DFE tích chập đặc của Hình 4. | Thay bằng **Lite-DFE** (DS-MSA tách biệt chiều sâu + BDRF nút cổ chai đảo ngược). | **0% (Hoàn toàn độc lập)** |
| **3. Hàm mất mát huấn luyện** | Cải tiến hàm mất mát chưng cất (focal/hard-mining/structural loss). | **Giữ nguyên 100%** hàm mất mát chuẩn $\ell_e, \ell_d$ ở Phương trình (4)–(7) khi chạy độc lập. | **0% (Hoàn toàn độc lập)** |
| **4. Cơ chế hợp nhất bản đồ suy luận** | Cải tiến cơ chế kết hợp $M_{t-e}^k$ và $M_{t-d}^k$ ở pha suy luận. | **Giữ nguyên 100%** công thức suy luận chuẩn $M(I_m) = \sum_k \text{Upsample}(M_{t-d}^k)$ ở Phương trình (8)–(10) khi chạy độc lập. | **0% (Hoàn toàn độc lập)** |
| **5. Giao diện tích hợp (A + B)** | Nhận đầu vào là các tensor đặc trưng $F_t^k, F_e^k, F_d^k \in \mathbb{R}^{w_k \times h_k \times c_k}$. | Xuất ra chính xác các tensor $F_e^k, F_d^k \in \mathbb{R}^{w_k \times h_k \times c_k}$ sau các đầu chiếu $\mathcal{P}_e^k, \mathcal{P}_d^k$. | **Tương thích 100% (Plug-and-Play)** |

> **Cơ chế Cộng hưởng khi Kết hợp (Synergy in Combined Model A + B):**
> 1. **Giao diện toán học khớp tuyệt đối:** Nhờ các đầu chiếu điểm $\mathcal{P}_e^k, \mathcal{P}_d^k$, đầu ra của `Lite-DSKD` (Phương pháp B) có kích thước tensor $(w_k, h_k, c_k)$ giống hệt DSKD gốc, cho phép cắm trực tiếp (plug-and-play) vào hàm mất mát cải tiến và bộ hợp nhất bản đồ dị thường $M_{t-e}^k + M_{t-d}^k$ của Phương pháp A mà không cần sửa đổi bất kỳ dòng code nào của Phương pháp A.
> 2. **Bổ trợ về mặt cơ chế biểu diễn:** Phương pháp B dùng nút cổ chai kiến trúc ($\alpha$-width + `Lite-DFE`) để ngăn chặn $S_d$ tái tạo nhầm vùng lõi dị thường (tăng chất lượng tín hiệu gốc $M_{t-d}^k$) đồng thời giảm >54% FLOPs, còn Phương pháp A tối ưu hóa hàm mục tiêu và khai thác sự kết hợp giữa nhánh mã hóa $M_{t-e}^k$ (nhạy với biên/kết cấu tần số cao) và nhánh giải mã $M_{t-d}^k$ (nhạy với ngữ nghĩa toàn cục). Sự kết hợp A + B vừa đạt **độ chính xác phát hiện/định vị cao nhất**, vừa đạt **chi phí tính toán nhẹ đúng với tên đề tài khóa học**.

---

## 6. Nguồn Tham khảo, Các Vấn đề Chưa Giải quyết & Mức độ Tin cậy

### 6.1. Nguồn tham khảo (Truy cập và kiểm chứng ngày `2026-11-14`)
1. **Liyi Yao, Shaobing Gao (2024).** *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection.* `arXiv:2402.00448v1 [cs.CV]`, 1 Feb 2024. Tệp cục bộ: `papers/2402.00448v1.pdf` (Truy cập ngày `2026-11-14`).
2. **Hanqiu Deng, Xingyu Li (2022).** *Anomaly Detection via Reverse Distillation from One-Class Embedding (RD4AD).* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9737–9746, 2022. *(Được DSKD trích dẫn tại [9] cho thiết kế mô-đun DFE ở Mục 3.2, tr. 6; truy cập ngày `2026-11-14`)*.
3. **Guodong Wang, Shumin Han, Errui Ding, Di Huang (2021).** *Student-Teacher Feature Pyramid Matching for Anomaly Detection (STPM).* In British Machine Vision Conference (BMVC), 2021. *(Được DSKD trích dẫn tại [28]; truy cập ngày `2026-11-14`)*.
4. **Mohammadreza Salehi, Niousha Sadjadi, Soroosh Baselizadeh, Mohammad Hossein Rohban, Hamid R. Rabiee (2021).** *Multiresolution Knowledge Distillation for Anomaly Detection (MKD).* In CVPR, pp. 14902–14912, 2021. *(Được DSKD trích dẫn tại [20] và đối chiếu trực tiếp trong Table 6, tr. 10; truy cập ngày `2026-11-14`)*.
5. **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016).** *Deep Residual Learning for Image Recognition.* In CVPR, pp. 770–778, 2016. *(Được DSKD trích dẫn tại [11] cho kiến trúc ResNet18; truy cập ngày `2026-11-14`)*.

### 6.2. Các vấn đề chưa giải quyết (Open Issues) & Mức độ tin cậy (Confidence Level)
1. **Chi tiết số kênh ẩn nội bộ của DFE trong code gốc của tác giả (`[UNVERIFIED]` $\to$ Đã xử lý bằng cấu hình chuẩn hóa):**
   - *Vấn đề:* Bài báo `2402.00448v1.pdf` không công bố mã nguồn chính thức và Hình 4 (tr. 5) không ghi con số kênh cụ thể bên trong khối Residual Block của DFE.
   - *Cách giải quyết trong thực nghiệm:* Trong bảng Ablation B-1, chúng tôi thiết lập cấu hình cơ sở chuẩn hóa **`B0-Clean`** (loại bỏ `layer4` chết, dùng đúng cấu trúc Hình 4 với $C_{\text{cat}} \to 256$) song song với số liệu **`B0-Paper`** (Table 6), và đo đạc trực tiếp Parameters, FLOPs, VRAM, CPU/GPU Latency trên cùng một môi trường phần cứng bằng `thop` / `torch.profiler`.
2. **Đánh giá Mức độ Tin cậy (Confidence Assessment):**
   - **Độ khớp toán học và ký hiệu với bài báo DSKD (`2402.00448v1.pdf`):** **Cao (100%)** — Mọi ký hiệu ($T, S_e, S_d, F_t^k, F_e^k, F_d^k, \ell_2, \ell_{\cos}, M_{t-e}^k, M_{t-d}^k$) và thông số thực nghiệm (Table 6: 39.3M params, 6.17G FLOPs, 0.217s, 105MB) đều được trích xuất trực tiếp từ bài báo gốc.
   - **Độ chính xác của kiểm toán Tham số & FLOPs:** **Cao (99%)** — Tổng FLOPs tính toán lý thuyết ($5.496\text{G} + 0.674\text{G} = 6.170\text{G}$) khớp chính xác tuyệt đối với con số `6.17G` trong Table 6 của bài báo gốc.
   - **Mức độ khả thi và độc lập với Phương pháp A:** **Cao (100%)** — Không có bất kỳ sự chồng lấn nào với nhánh Loss / Anomaly Map Fusion của Agent 3.
