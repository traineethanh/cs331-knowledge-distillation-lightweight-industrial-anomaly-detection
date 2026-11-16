# Báo Cáo Kiểm Toán Kỹ Thuật Độc Lập & Đóng Băng Thiết Kế Nghiên Cứu (Final Technical Audit & Research Freeze)

**Dự án:** Knowledge Distillation-Based Lightweight Industrial Anomaly Detection (CS331)  
**Vai trò Kiểm toán:** Research Lead & Independent Technical Auditor  
**Nhánh làm việc:** `chore/repository-reset` (HEAD: `41af7b015a69fe7480c8fd1212e6e22840ae22d0`)  
**Bài báo nền tảng (Baseline):** *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection* (`arXiv:2402.00448v1`, Liyi Yao & Shaobing Gao, 01/02/2024 — `papers/2402.00448v1.pdf`)  
**Ngày kiểm toán độc lập:** `2026-11-14` (Đã đối soát & đồng bộ theo `post_audit_verification.md`)  
**Trạng thái sau kiểm toán & đối soát:** **`READY FOR USER APPROVAL (IMPLEMENTATION PROPOSAL READY)`**

---

## 0. Nguyên Tắc Kiểm Toán Độc Lập & Phân Loại Nhận Thức

Báo cáo kiểm toán này được thực hiện hoàn toàn độc lập, **không bảo vệ các quyết định trước đó chỉ vì chúng được 5 agents chấm điểm cao (`4.79–4.81/5.0`)**. Mọi phương trình, kích thước tensor, số lượng tham số, trích dẫn học thuật và giao thức thực nghiệm trong 10 tài liệu thuộc `docs/research/` đều được đối chiếu trực tiếp với tệp PDF gốc trong `papers/` và mã nguồn trong `src/anomaly_detection/`.

- **`[FACT - Paper]`:** Nội dung, ký hiệu và con số in trực tiếp trong bài báo gốc (`papers/2402.00448v1.pdf`) hoặc các PDF đối chứng trong `papers/`.
- **`[FACT - Mathematical Proof]`:** Hệ quả toán học hoặc số lượng tham số/FLOPs được chứng minh bằng biến đổi đại số tường minh.
- **`[AUDIT FINDING - ERROR/MISMATCH]`:** Sai sót toán học, mâu thuẫn chéo giữa các tài liệu, hoặc lỗi trích dẫn của các agents ở Vòng 1–2 đã bị phát hiện qua kiểm toán.
- **`[MANDATORY FIX]`:** Quy định sửa đổi bắt buộc đã được áp dụng để đóng băng thiết kế trước khi cài đặt code.

---

## 1. Kiểm Toán Nghiêm Ngặt Công Thức `Eq. (3)`–`Eq. (6)` Của Bài Báo Gốc DSKD (Ưu Tiên P0)

### 1.1. Kiểm tra trực tiếp hình ảnh trang 4, 5, 6 của `papers/2402.00448v1.pdf`

1. **`[FACT - Paper]` Tại Trang 5, Mục 3.1 (*Dual-Student Knowledge Distillation*), `Eq. (3)` được in nguyên văn là:**
   $$\hat{F}_t^k(I_m)_{ij} = \frac{F_t^k(I_m)_{ij}}{\left\|F_t^k(I_m)_{ij}\right\|_{\ell_2}^2}, \quad \hat{F}_e^k(I_m)_{ij} = \frac{F_e^k(I_m)_{ij}}{\left\|F_e^k(I_m)_{ij}\right\|_{\ell_2}^2}, \quad \hat{F}_d^k(I_m)_{ij} = \frac{F_d^k(I_m)_{ij}}{\left\|F_d^k(I_m)_{ij}\right\|_{\ell_2}^2} \tag{Eq. 3 - Literal PDF}$$
   trong đó mẫu số có số mũ bình phương `$2$` và chỉ số dưới `$\ell_2$`.
2. **`[FACT - Paper]` Tại Trang 6, `Eq. (4)`, `Eq. (5)` và `Eq. (6)` được in nguyên văn là:**
   $$\ell_2^{t-e}(I_m)_{ij} = \frac{1}{2}\left\|\hat{F}_t^k(I_m)_{ij} - \hat{F}_e^k(I_m)_{ij}\right\|_{\ell_2}^2 \tag{Eq. 4}$$
   $$\ell_{\cos}^{t-e}(I_m)_{ij} = 1 - \frac{\left(\hat{F}_t^k\right)_{ij}^T \cdot \hat{F}_e^k(I_m)_{ij}}{\left\|\hat{F}_t^k(I_m)_{ij}\right\| \left\|\hat{F}_e^k(I_m)_{ij}\right\|} \tag{Eq. 5}$$
   $$M_{t-e}^k(I_m)_{ij} = \lambda \ell_2^{t-e}(I_m)_{ij} + \ell_{\cos}^{t-e}(I_m)_{ij} \tag{Eq. 6}$$

---

### 1.2. Chứng minh / Bác bỏ tính tương đương giữa Euclidean Loss (`Eq. 4`) và Cosine Loss (`Eq. 5`) trong hai trường hợp

Để đơn giản hóa ký hiệu tại một vị trí không gian $(i, j)$ và tầng $k$, đặt $u = F_t^k(I_m)_{ij} \in \mathbb{R}^{c_k} \setminus \{0\}$, $v = F_s^k(I_m)_{ij} \in \mathbb{R}^{c_k} \setminus \{0\}$ ($s \in \{e, d\}$), chuẩn Euclidean bậc một $r_u = \|u\|_2 = \sqrt{u^T u} > 0$, $r_v = \|v\|_2 = \sqrt{v^T v} > 0$, và góc $\theta$ giữa $u, v$ thỏa mãn $\cos\theta = \frac{u^T v}{r_u r_v}$.

#### Trường hợp 1: Literal `Eq. (3)` — Chia cho bình phương chuẩn $\ell_2$ ($\hat{u}_{\text{lit}} = \frac{u}{\|u\|_2^2}$, $\hat{v}_{\text{lit}} = \frac{v}{\|v\|_2^2}$)

1. **Độ dài chuẩn $\ell_2$ của vector sau khi chia:**
   $$\left\|\hat{u}_{\text{lit}}\right\|_2 = \left\|\frac{u}{\|u\|_2^2}\right\|_2 = \frac{\|u\|_2}{\|u\|_2^2} = \frac{1}{\|u\|_2} = \frac{1}{r_u} \ne 1 \quad (\text{trừ khi } r_u = 1)$$
   Tương tự, $\left\|\hat{v}_{\text{lit}}\right\|_2 = \frac{1}{r_v} \ne 1$. Phép biến đổi $\hat{u}_{\text{lit}} = u / \|u\|_2^2$ chính là **phép nghịch đảo cầu (spherical inversion / Kelvin transform)** qua mặt cầu đơn vị, biến vector có độ lớn nhỏ ($r_u \to 0$) thành vector có độ lớn tiến ra vô cùng ($\|\hat{u}_{\text{lit}}\|_2 \to \infty$)!
2. **Thay vào `Eq. (4)` (Euclidean distance):**
   $$\ell_{2,\text{lit}}^{t-s} = \frac{1}{2}\left\|\hat{u}_{\text{lit}} - \hat{v}_{\text{lit}}\right\|_2^2 = \frac{1}{2}\left(\left\|\hat{u}_{\text{lit}}\right\|_2^2 + \left\|\hat{v}_{\text{lit}}\right\|_2^2 - 2\hat{u}_{\text{lit}}^T \hat{v}_{\text{lit}}\right) = \frac{1}{2}\left(\frac{1}{r_u^2} + \frac{1}{r_v^2} - \frac{2\cos\theta}{r_u r_v}\right)$$
   Biểu thức này có thể viết lại thành tổng của hai thành phần không âm:
   $$\ell_{2,\text{lit}}^{t-s} = \underbrace{\frac{1}{2}\left(\frac{1}{r_u} - \frac{1}{r_v}\right)^2}_{\text{Sai lệch nghịch đảo biên độ}} + \underbrace{\frac{1 - \cos\theta}{r_u r_v}}_{\text{Sai lệch góc bị nhân với } (r_u r_v)^{-1}} \tag{1.1}$$
3. **Thay vào `Eq. (5)` (Cosine distance):**
   Vì `Eq. (5)` có mẫu số chia lại cho tích hai chuẩn $\left\|\hat{u}_{\text{lit}}\right\|_2 \left\|\hat{v}_{\text{lit}}\right\|_2 = \frac{1}{r_u r_v}$:
   $$\ell_{\cos,\text{lit}}^{t-s} = 1 - \frac{\hat{u}_{\text{lit}}^T \hat{v}_{\text{lit}}}{\left\|\hat{u}_{\text{lit}}\right\|_2 \left\|\hat{v}_{\text{lit}}\right\|_2} = 1 - \frac{\frac{u^T v}{r_u^2 r_v^2}}{\frac{1}{r_u} \cdot \frac{1}{r_v}} = 1 - \frac{u^T v}{r_u r_v} = 1 - \cos\theta \tag{1.2}$$
4. **`[FACT - Mathematical Proof]` Kết luận cho Trường hợp 1 (Literal `Eq. 3`):**
   - **BÁC BỎ TÍNH TƯƠNG ĐƯƠNG:** Trong Trường hợp 1, $\ell_{2,\text{lit}}^{t-s} \ne \ell_{\cos,\text{lit}}^{t-s}$. Các tài liệu Vòng 1 (`dskd_paper_analysis.md`, `method_a_proposal.md`) kết luận $\ell_2 \equiv \ell_{\cos}$ là **chỉ đúng dưới giả định Trường hợp 2 ($\|\hat{F}\|_2 = 1$)**, hoàn toàn **không đúng nếu thực thi theo đúng chữ in của `Eq. (3)`**.
   - **Phân tích tác động lên đạo hàm (Gradient) và thang đo Loss (`Eq. 6`):**
     Tính đạo hàm của $\ell_{2,\text{lit}}^{t-s}$ theo vector đặc trưng học sinh $v = F_s^k(I_m)_{ij}$:
     $$\nabla_v \hat{v}_{\text{lit}} = \frac{1}{r_v^2}\left(I - \frac{2 v v^T}{r_v^2}\right) \implies \nabla_v \ell_{2,\text{lit}}^{t-s} = \frac{1}{r_v^2}\left(I - \frac{2 v v^T}{r_v^2}\right)\left(\frac{v}{r_v^2} - \frac{u}{r_u^2}\right) = -\frac{v}{r_v^4} - \frac{1}{r_u^2 r_v^2}\left(u - \frac{2(u^T v)}{r_v^2}v\right) \tag{1.3}$$
     Từ `(1.1)` và `(1.3)`, ta thấy hai hệ quả số học & tối ưu hóa cần lưu ý của Literal `Eq. (3)`:
     - **Đặc tính số học với $\epsilon > 0$ và bùng nổ đạo hàm bậc $O(1/\epsilon)$:**
       - Trong giải tích thuần túy không có số hạng bảo vệ ($\epsilon = 0$), phép nghịch đảo Kelvin $\hat{v} = v / \|v\|_2^2$ có điểm kỳ dị tại $r_v = 0$ khiến $\|\hat{v}\|_2 = 1/r_v \to \infty$ và đạo hàm $\|\nabla_v \ell_{2,\text{lit}}^{t-s}\|_2 \propto r_v^{-3} \to \infty$.
       - Trong cài đặt thực tế với số hạng ổn định $\epsilon > 0$, tức $\hat{v}_{\text{lit}} = \frac{v}{r_v^2 + \epsilon}$, khi $r_v \to 0$ thì $\hat{v}_{\text{lit}} \to 0$ liên tục (không tiến ra vô cùng). Độ lớn đạt cực đại tại $r_v = \sqrt{\epsilon}$ với $\|\hat{v}_{\text{lit}}\|_{\max} = \frac{1}{2\sqrt{\epsilon}}$ và độ lớn đạo hàm bị chặn trên bởi $O(1/\epsilon)$ tại $r_v = 0$.
       - Tuy nhiên, trong độ chính xác số thực (`FP32`/`FP16`), nếu chọn $\epsilon = 10^{-12}$, đỉnh đạo hàm vọt lên $\sim 10^{12}$ gây sốc gradient; nếu chọn $\epsilon = 10^{-4}$, độ lớn vector bị kéo dãn tới $50.0$ tại các vùng gần $0$ trong khi các vùng cấu trúc $r_v \approx 10$ bị co lại còn $0.1$.
     - **Mất ổn định thang đo trong `Eq. (6)` ($M = \lambda \ell_2 + \ell_{\cos}$):** Trong khi $\ell_{\cos} \in [0, 2]$ luôn bị chặn, số hạng $\lambda \ell_{2,\text{lit}}^{t-s}$ bị chi phối bởi các điểm ảnh có chuẩn $\|F\|_2$ nhỏ nhất thay vì các điểm ảnh có sai lệch đặc trưng thực sự. Thêm vào đó, $\ell_{2,\text{lit}} \not\equiv \ell_{\cos,\text{lit}}$ nên hai hàm mất mát triệt tiêu lẫn nhau về mặt tỷ lệ.

---

#### Trường hợp 2: Conventional Unit $\ell_2$ Normalization — Chia cho chuẩn $\ell_2$ bậc một ($\hat{u}_{\text{unit}} = \frac{u}{\|u\|_2}$, $\hat{v}_{\text{unit}} = \frac{v}{\|v\|_2}$)

1. **Độ dài chuẩn $\ell_2$ của vector sau khi chia:**
   $$\left\|\hat{u}_{\text{unit}}\right\|_2 = \left\|\frac{u}{\|u\|_2}\right\|_2 = 1, \qquad \left\|\hat{v}_{\text{unit}}\right\|_2 = \left\|\frac{v}{\|v\|_2}\right\|_2 = 1$$
2. **Thay vào `Eq. (4)` và `Eq. (5)`:**
   $$\ell_{2,\text{unit}}^{t-s} = \frac{1}{2}\left\|\hat{u}_{\text{unit}} - \hat{v}_{\text{unit}}\right\|_2^2 = \frac{1}{2}\left(1 + 1 - 2\hat{u}_{\text{unit}}^T \hat{v}_{\text{unit}}\right) = 1 - \cos\theta \equiv \ell_{\cos,\text{unit}}^{t-s} \tag{1.4}$$
3. **Thay vào `Eq. (6)`:**
   $$M_{t-s,\text{unit}}^k(I_m)_{ij} = \lambda \ell_{2,\text{unit}}^{t-s} + \ell_{\cos,\text{unit}}^{t-s} = (1 + \lambda)(1 - \cos\theta) = 1.1(1 - \cos\theta) \quad (\text{với } \lambda = 0.1) \tag{1.5}$$
4. **Đạo hàm theo $v$:**
   $$\nabla_v \ell_{2,\text{unit}}^{t-s} = \nabla_v (1 - \cos\theta) = -\frac{1}{r_v}\left(\hat{u}_{\text{unit}} - (\cos\theta)\hat{v}_{\text{unit}}\right), \qquad \left\|\nabla_v \ell_{2,\text{unit}}^{t-s}\right\|_2 = \frac{|\sin\theta|}{r_v}$$
5. **`[FACT - Mathematical Proof]` Kết luận cho Trường hợp 2 (Conventional Unit $\ell_2$ Norm):**
   - **CHỨNG MINH TÍNH TƯƠNG ĐƯƠNG:** Khi và chỉ khi chuẩn hóa theo chuẩn $\ell_2$ bậc một ($\|\hat{F}\|_2 = 1$), `Eq. (4)` và `Eq. (5)` mới đồng nhất bằng $1 - \cos\theta$, và `Eq. (6)` mới rút gọn thành $(1+\lambda)(1-\cos\theta) \in [0, 2.2]$.

---

### 1.3. Kiểm tra bằng chứng nội tại trong `papers/2402.00448v1.pdf`: Có chỗ nào giải thích hoặc mâu thuẫn với `Eq. (3)` không?

Kiểm toán viên đã rà soát toàn bộ 12 trang của `papers/2402.00448v1.pdf` và ghi nhận **hai nhóm bằng chứng nội tại đối lập nhau**:

1. **Bằng chứng ủng hộ `Eq. (3)` là chuẩn hóa $\ell_2$ thông thường (`l2_normalization`) hoặc lấy cảm hứng từ STPM [28]:**
   - Tại Trang 4 (`Algorithm 1`, dòng 7–9) và Trang 6 (`Algorithm 2`, dòng 5–8), mã giả gọi hàm `l2_normalization(T(I_j^t))`, `l2_normalization(S_e(I_j^t))`, `l2_normalization(S_d(F_emb))`. Trong mọi thư viện chuẩn (PyTorch `F.normalize(x, p=2)`, TensorFlow `tf.math.l2_normalize`, Scikit-learn), `l2_normalization` đều chia cho $\|x\|_2$ chứ không chia cho $\|x\|_2^2$.
   - Tại Trang 5 (đoạn văn ngay trên `Eq. 3`), tác giả viết: *"Similar to the approach in Wang et al. [28]... These vectors are normalized by $\ell_2$-distance as follows"*. Trong bài báo gốc của Wang et al. [28] (*STPM*, BMVC 2021, Eq. 1), công thức chuẩn hóa là $\hat{F} = F / \|F\|_2$ và khoảng cách là $\frac{1}{2}\|\hat{F}_t - \hat{F}_s\|_2^2$.
2. **Bằng chứng cho thấy tác giả KHÔNG coi $\|\hat{F}\|_2 = 1$ khi viết `Eq. (5)` và `Eq. (6)`:**
   - Tại Trang 6, trong `Eq. (5)`, tác giả viết rõ mẫu số $\left\|\hat{F}_t^k(I_m)_{ij}\right\| \left\|\hat{F}_e^k(I_m)_{ij}\right\|$. Nếu tác giả mặc định $\|\hat{F}\|_2 = 1$ từ `Eq. (3)`, mẫu số của `Eq. (5)` là dư thừa hoàn toàn.
   - Tại Trang 6, đoạn ngay trên `Eq. (5)`, tác giả viết: *"Notice that only adopts $\ell_2$-distance may overlook the high-dimensional structural information. To capture the correlation inside feature map more comprehensively, we utilize cosine distance to measure the anomaly further."* Tiếp đó, tại `Eq. (6)`, tác giả kết hợp tuyến tính $M = \lambda \ell_2 + \ell_{\cos}$ với hệ số cân bằng $\lambda = 0.1$ (tương tự bài báo MKD — Salehi et al. [20], CVPR 2021, nơi $\ell_2$ được tính trên **vector chưa chuẩn hóa** $F_t, F_s$ còn $\ell_{\cos}$ tính góc, nên mới cần $\lambda = 0.1$ để cân bằng thang đo giữa MSE và Cosine!).

**`[AUDIT FINDING - ERROR/MISMATCH]` Kết luận Kiểm toán về `Eq. (3)`:**
- **Không được phép tự ý khẳng định `Eq. (3)` chỉ là lỗi đánh máy (typo) mà bỏ qua kiểm chứng!** Có đúng **3 khả năng cài đặt thực tế** của tác giả Yao & Gao do không có mã nguồn công khai:
  - **Biến thể `N1 (Unit-L2 Norm)`:** Chuẩn hóa đơn vị $\hat{F} = \frac{F}{\|F\|_2 + \epsilon}$ rồi tính `Eq. (4) + Eq. (5)` $\implies M = (1+\lambda)(1-\cos\theta)$ (theo STPM [28] + Algorithm 1/2).
  - **Biến thể `N2 (Raw-L2 + Cosine / MKD-style)`:** Tính `Eq. (4)` trên đặc trưng gốc (hoặc chuẩn hóa kênh nhẹ) kết hợp với `Eq. (5)` Cosine có mẫu số $\|\hat{F}_t\|\|\hat{F}_s\|$ với $\lambda = 0.1$ (giải thích tại sao `Eq. 5` vẫn còn mẫu số chuẩn và tại sao cần $\lambda=0.1$ để cân bằng độ lớn cộng với hướng như MKD [20]).
  - **Biến thể `N3 (Literal Eq. 3 Inv-Sq Norm)`:** Chia đúng theo chữ in $\hat{F} = \frac{F}{\|F\|_2^2 + \epsilon}$ (cần $\epsilon \ge 10^{-4}$ để không bị `NaN`).
- **`[MANDATORY FIX]` Tách biệt hoàn toàn hai phép kiểm tra tiền thực nghiệm ở Group 0:**
  1. **`S0_eq3_norm` (Kiểm tra Công thức Chuẩn hóa Đặc trưng `Eq. 3`):** Đối chiếu thực nghiệm 3 biến thể `N1 (Unit-L2)` vs. `N2 (Raw-L2 + Cosine)` vs. `N3 (Literal Inv-Sq L2)` trên `carpet` và `bottle`.
  2. **`S0_input_norm` (Kiểm tra Chuẩn hóa Ảnh Đầu vào Mục 4.1.2 tr. 7):** Tách biệt hoàn toàn với `Eq. (3)`, đối chiếu tiền xử lý ảnh đầu vào giữa `P_imagenet` (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`) và `P_img_l2` (chuẩn hóa $\ell_2$ trên ảnh resize như câu chữ ở Mục 4.1.2 tr. 7).
  3. **Phân biệt rõ trong báo cáo:**
     - `B0-Paper`: Kết quả và công thức in trong bài báo gốc (`Table 1, 2, 6`).
     - `B0-Clean`: Bản tái lập chuẩn hóa của nhóm sau khi xác nhận qua `S0_eq3_norm` và `S0_input_norm`.

---

## 2. Kiểm Toán Kỹ Thuật Phương Pháp A (`FC-DSKD`)

### 2.1. Kiểm toán Mô-đun A1 (Spatial-Focal & Contextual Patch Discrepancy)

1. **Tính hợp lệ toán học của khoảng cách điểm ($d_{\text{pt}}$) và ngữ cảnh ($d_{\text{ctx}}$):**
   - Với $\hat{F}_z^k(I)_{ij} = \frac{F_z^k(I)_{ij}}{\|F_z^k(I)_{ij}\|_2 + \epsilon}$ ($\epsilon = 10^{-12}$), chuẩn $\|\hat{F}_z^k(I)_{ij}\|_2 = 1$ nên $d_{\text{pt}} \in [0, 2.2]$ hoàn toàn xác định và khả vi.
   - Vector ngữ cảnh $\bar{F}_z^k(I)_{ij} = \text{AvgPool}_{3\times 3}(\hat{F}_z^k(I))_{ij}$ là trung bình của 9 vector đơn vị trong cửa sổ $3\times 3$. Khi 9 vector lân cận ngược hướng nhau, $\|\bar{F}_z^k(I)_{ij}\|_2$ có thể tiến gần $0$; do đó phép tái chuẩn hóa $\hat{G}_z^k(I)_{ij} = \frac{\bar{F}_z^k(I)_{ij}}{\|\bar{F}_z^k(I)_{ij}\|_2 + \epsilon}$ với $\epsilon = 10^{-12}$ (hoặc $\text{clamp}(\|\cdot\|_2, \min=10^{-8})$ trong `FP16/FP32`) là **bắt buộc** để tránh chia cho $0$.
2. **Đánh giá rủi ro trọng số tiêu cự (Spatial-Focal Weighting) khuếch đại nhiễu nền:**
   - Trọng số $w_s^k(I_m)_{ij} = \min\!\left(w_{\max}, \left(1 + \text{sg}[\tilde{M}_{t-s}^k(I_m)_{ij}]\right)^\gamma\right)$ với $\tilde{M}_{t-s}^k(I_m)_{ij} = \frac{M_{t-s}^k(I_m)_{ij}}{\text{mean}_{u,v}(M_{t-s}^k(I_m)_{uv}) + \epsilon}$ có trung bình không gian xấp xỉ $1$.
   - **Rủi ro thực tế:** Trong các danh mục vật thể có vùng nền lớn hoặc bụi/phản quang nhẹ trên ảnh `train/good` (như `pill`, `metal_nut`, `screw`), nếu dùng $\gamma \ge 2.0$ không kẹp trần, các điểm nhiễu ngoại lai trong tập `train/good` sẽ chiếm toàn bộ gradient, khiến học sinh học thuộc lòng nhiễu nền.
   - **`[MANDATORY FIX]`:** Giữ cố định $\gamma = 1.0$, kẹp trần $w_{\max} = 5.0$, và **bắt buộc tách riêng 2 bước Ablation**: `Cfg-1 (Only A1a: Contextual Patch, \gamma=0)` và `Cfg-2 (Full A1: Contextual + Spatial-Focal, \gamma=1.0)`. Nếu trên các lớp có nền nhiễu mà `Cfg-2` thấp hơn `Cfg-1`, kết quả âm tính của thành phần tiêu cự phải được báo cáo trung thực.
3. **Kiểm tra rò rỉ tập `test`:** Мо-đun A1 chỉ tính trong pha huấn luyện trên `split="train"` và khi trích xuất bản đồ từng ảnh độc lập, hoàn toàn không dùng bất kỳ thông tin nào từ tập `test`.

---

### 2.2. Kiểm toán Mô-đun A2 (Asymmetric Stop-Gradient Cross-Student Consistency)

1. **`[AUDIT FINDING - ERROR/MISMATCH]` Phát hiện lỗi đồ thị đạo hàm ngầm từ $\ell_d$ qua `DFE` về $S_e$:**
   - Trong `method_a_proposal.md` (dòng 248–252) và `method_comparison.md` (Mục 4.2), các agents lập luận rằng chỉ cần đặt $\text{sg}[\hat{F}_e^k]$ trong $\ell_{e-d}$ (`Eq. A8`) thì $S_e$ sẽ không nhận đạo hàm từ nhánh giải mã.
   - **Kiểm toán viên bác bỏ giả định này nếu không ngắt đạo hàm tại đầu vào của `DFE`!** Hãy xem `Algorithm A1` dòng 4 (`method_a_proposal.md` dòng 323):
     $$F_{\text{emb}} = \text{DFE}(F_e^1, F_e^2, F_e^3), \qquad F_d^k = S_d(F_{\text{emb}})$$
     Nếu $F_e^1, F_e^2, F_e^3$ truyền thẳng vào `DFE` mà **không `.detach()`** (hoặc nếu dùng chung một lệnh `loss_total.backward()` trước khi `optimizer_e.step()`), thì theo quy tắc dây chuyền (chain rule):
     $$\frac{\partial \ell_d}{\partial \theta_e} = \sum_{k=1}^3 \frac{\partial \ell_d}{\partial F_d} \frac{\partial F_d}{\partial F_{\text{emb}}} \frac{\partial F_{\text{emb}}}{\partial F_e^k} \frac{\partial F_e^k}{\partial \theta_e} \ne 0$$
     Tức là ngay cả khi $\ell_{e-d}$ đã có $\text{sg}[\hat{F}_e^k]$, chính hàm mất mát $\ell_d$ (và $\ell_{e-d}$ thông qua $F_d^k = S_d(\text{DFE}(F_e))$) **vẫn truyền ngược đạo hàm xuyên qua `DFE` về làm hỏng $S_e$**, trái ngược với phát biểu tại Mục 3.1 (tr. 3) của bài báo DSKD: *"The distillation processes between two students and the teacher is performed separately and independently"* và `Algorithm 1` dòng 16–17!
2. **`[MANDATORY FIX]` Quy tắc Cô lập Đạo hàm Hai Nhánh (`Strict Dual-Branch Gradient Isolation`):**
   - Trong `DSKDModel.forward()`, đầu vào của `DFE` bắt buộc phải ngắt đồ thị đạo hàm từ $S_e$:
     ```python
     f_emb = self.dfe([f.detach() for f in raw_f_e])
     ```
   - Đồng thời, `optimizer_e` chỉ cập nhật $\theta_e$ từ `loss_e.backward()`, và `optimizer_d` chỉ cập nhật $(\theta_{\text{dfe}}, \theta_d)$ từ `(loss_d + beta_t * loss_ed).backward()`.
   - Khi đó, $\nabla_{\theta_e} \mathcal{L}_{\text{FC}} = \nabla_{\theta_e} \ell_{\text{focal}}^e$ được bảo toàn **tuyệt đối 100%** (không bị nhiễu từ cả $\ell_d$ lẫn $\ell_{e-d}$), trong khi $(\theta_{\text{dfe}}, \theta_d)$ nhận được lực dẫn hướng phụ trợ từ $\text{sg}[\hat{F}_e^k]$.
3. **Đánh giá tính cần thiết thực sự của A2 ($\ell_{e-d}$):**
   - Vì cả $S_e$ và $S_d$ đều đã học khớp với cùng một mục tiêu cố định là Teacher $\hat{F}_t^k$, bất đẳng thức tam giác cho ta $\|\hat{F}_e^k - \hat{F}_d^k\|_2 \le \|\hat{F}_e^k - \hat{F}_t^k\|_2 + \|\hat{F}_d^k - \hat{F}_t^k\|_2$. Do đó, lợi ích của $\ell_{e-d}$ khi $\hat{F}_t^k$ đã có mặt là một giả thuyết cần kiểm chứng nghiêm ngặt chứ không hiển nhiên.
   - **`[MANDATORY FIX]`:** Trong ma trận Ablation của Method A, bắt buộc đối chiếu trực tiếp `Cfg-2 (Only A1)` với `Cfg-4 (A1 + A2)`. Nếu `A2` không mang lại mức tăng có ý nghĩa hoặc làm giảm tính đa dạng biểu diễn giữa $S_e$ và $S_d$ khi hợp nhất ở `A3`, cấu hình cuối cùng `A_best` được phép đặt $\beta = 0$ (loại bỏ `A2` dựa trên bằng chứng thực nghiệm).

---

### 2.3. Kiểm toán Mô-đun A3 (Calibrated Complementary Fusion & Top-$\rho$ Scoring)

Kiểm toán viên trả lời trực tiếp và tường minh **5 câu hỏi bắt buộc** đối với công thức hợp nhất bản đồ dị thường của Agent 3:
$$\Phi^k(I^a)_{ij} = (1 - \alpha_k)\,\widehat{M}_{t-e}^k(I^a)_{ij} + \alpha_k\,\widehat{M}_{t-d}^k(I^a)_{ij} + \eta\sqrt{\widehat{M}_{t-e}^k(I^a)_{ij} \cdot \widehat{M}_{t-d}^k(I^a)_{ij} + \epsilon} \tag{Eq. A12}$$

1. **Câu hỏi 1: Thành phần tích $\widehat{M}_{t-e}^k \cdot \widehat{M}_{t-d}^k$ (hoặc $\sqrt{\widehat{M}_{t-e}^k \cdot \widehat{M}_{t-d}^k}$) thực chất là *Agreement (Đồng thuận)* hay *Complementarity (Bổ trợ)*?**
   - **Trả lời dứt khoát (`[FACT - Mathematical Proof]`):** Tích số $\widehat{M}_{t-e}^k \cdot \widehat{M}_{t-d}^k$ (và trung bình nhân $\sqrt{\widehat{M}_{t-e}^k \cdot \widehat{M}_{t-d}^k}$) là toán tử **ĐỒNG THUẬN / GIAO LOGIC (Agreement / Concordance / Soft-AND)**, hoàn toàn **KHÔNG PHẢI** là toán tử bổ trợ (*Complementarity*)!
   - Ngược lại, thành phần **tổ hợp tuyến tính lồi** $(1 - \alpha_k)\widehat{M}_{t-e}^k + \alpha_k\widehat{M}_{t-d}^k$ mới là toán tử **HỢP TUYẾN TÍNH / BỔ TRỢ (Complementary Linear Union / Soft-OR)** vì nó giữ lại tín hiệu khi chỉ một trong hai nhánh phát hiện ra lỗi.
2. **Câu hỏi 2: Công thức này có làm suy giảm các bất thường chỉ được phát hiện bởi một nhánh hay không?**
   - **Phân tích toán học chính xác:**
     - Giả sử tại vùng lõi của một khuyết tật diện tích lớn ở tầng sâu $k=3$, học sinh mã hóa bị đồng nhất hóa ($\widehat{M}_{t-e}^3 = 0$) nhưng học sinh giải mã phát hiện tốt ($\widehat{M}_{t-d}^3 = Z > 0$). Khi đó:
       $$\Phi^3 = (1 - \alpha_3)\cdot 0 + \alpha_3 Z + \eta\sqrt{0 \cdot Z + \epsilon} \approx \alpha_3 Z = 0.75 Z > 0$$
     - Giả sử tại một vết xước biên siêu mảnh ở tầng nông $k=1$, học sinh mã hóa phát hiện sắc nét ($\widehat{M}_{t-e}^1 = Z > 0$) nhưng học sinh giải mã bị mờ ($\widehat{M}_{t-d}^1 = 0$). Khi đó:
       $$\Phi^1 = (1 - \alpha_1) Z + \alpha_1 \cdot 0 + 0 = (1 - \alpha_1) Z = 0.75 Z > 0$$
     - Giả sử tại điểm ảnh mà **cả hai nhánh cùng phát hiện** ($\widehat{M}_{t-e}^k = \widehat{M}_{t-d}^k = Z > 0$):
       $$\Phi^k = (1 - \alpha_k)Z + \alpha_k Z + \eta Z = (1 + \eta) Z = 1.30 Z$$
   - **Kết luận:** Vì số hạng trung bình nhân $\eta\sqrt{\widehat{M}_{t-e}^k \widehat{M}_{t-d}^k}$ được **cộng thêm (additive boost)** vào tổng tuyến tính $(1-\alpha_k)\widehat{M}_{t-e}^k + \alpha_k\widehat{M}_{t-d}^k$ với $\eta = 0.3$, nó **không triệt tiêu về 0** các bất thường chỉ xuất hiện ở một nhánh (nhánh đơn lẻ vẫn giữ được $75\%$ biên độ tại tầng chuyên biệt của nó), nhưng nó làm giảm tỷ số tương đối giữa vùng đơn nhánh và vùng đồng thuận hai nhánh theo hệ số $\frac{\alpha_k}{1+\eta} \approx 0.58$. Nếu ai đó dùng **thuần túy tích số** $\widehat{M}_{t-e}^k \cdot \widehat{M}_{t-d}^k$ thì mọi bất thường đơn nhánh sẽ bị xóa sổ hoàn toàn!
3. **Câu hỏi 3: Công thức hiện tại có phù hợp với mục tiêu khai thác thông tin bổ trợ không, hay cần đổi tên / thiết kế lại?**
   - **`[MANDATORY FIX]` Đổi tên thuật ngữ chính xác & bổ sung đối chứng Ablation:**
     - **Nghiêm cấm** gọi số hạng $\sqrt{\widehat{M}_{t-e}^k \cdot \widehat{M}_{t-d}^k}$ là *"Complementary Gate"*. Phải định danh chính xác hai thành phần của `Eq. (A12)` là:
       $$\Phi^k(I^a)_{ij} = \underbrace{(1 - \alpha_k)\,\widehat{M}_{t-e}^k(I^a)_{ij} + \alpha_k\,\widehat{M}_{t-d}^k(I^a)_{ij}}_{\text{Complementary Linear Union (Bảo toàn bổ trợ đơn nhánh)}} + \underbrace{\eta\sqrt{\widehat{M}_{t-e}^k(I^a)_{ij} \cdot \widehat{M}_{t-d}^k(I^a)_{ij} + \epsilon}}_{\text{Concordance Multiplicative Boost (Khuếch đại đồng thuận hai nhánh)}}$$
     - Bổ sung vào Ablation của Mô-đun A3 (xạy trực tiếp ở pha suy luận, **tốn $0$ phút huấn luyện**) phép so sánh 3 quy tắc hợp nhất trên cùng bản đồ đã hiệu chuẩn Z-score:
       1. `A3-Linear` ($\eta = 0$): Thuần túy hợp nhất bổ trợ tuyến tính $(1-\alpha_k)\widehat{M}_{t-e}^k + \alpha_k\widehat{M}_{t-d}^k$.
       2. `A3-Synergistic` ($\eta = 0.3$): Hợp tuyến tính bổ trợ + Khuếch đại đồng thuận nhân tính (`Eq. A12`).
       3. `A3-SoftMaxUnion`: $\max\!\left((1-\alpha_k)\widehat{M}_{t-e}^k, \alpha_k\widehat{M}_{t-d}^k\right) + \eta\sqrt{\widehat{M}_{t-e}^k \widehat{M}_{t-d}^k + \epsilon}$ (bảo toàn $100\%$ biên độ bất thường đơn nhánh ở vùng lõi khuyết tật lớn mà không bị suy giảm theo $(1-\alpha_k)$).
4. **Câu hỏi 4: Làm rõ dứt khoát `q = 0.001` là tỷ lệ phân số `0.001` (`0.1%` số pixel) hay `0.001%`?**
   - **`[AUDIT FINDING - ERROR/MISMATCH]`:** Trong `method_a_proposal.md` và `research_decision.md`, các agents viết lẫn lộn ký hiệu chữ `Top-q%` với giá trị số `q = 0.001`, gây nguy cơ hiểu nhầm nghiêm trọng khi lập trình!
   - **Tính toán cụ thể trên ảnh $256 \times 256 = 65,536$ pixels:**
     - Nếu hiểu là **phân số diện tích (tail fraction)** $\rho_{\text{tail}} = 0.001$ (tức **$0.1\%$ tổng số pixel**):
       $$K_{\text{tail}} = \max(1, \lfloor 0.001 \times 65,536 \rfloor) = \mathbf{65\text{ pixels}}$$
       Trung bình trên 65 điểm ảnh lớn nhất giúp khử nhiễu xung đơn lẻ nhưng vẫn đủ nhỏ để nhạy với vết xước mảnh (như `screw`, `capsule`).
     - Nếu hiểu nhầm là **$0.001\%$** (tức phân số $10^{-5}$):
       $$K_{\text{tail}} = \max(1, \lfloor 10^{-5} \times 65,536 \rfloor) = \max(1, \lfloor 0.655 \rfloor) = \mathbf{1\text{ pixel}}$$
       Khi $K_{\text{tail}} = 1$, công thức suy biến **100% thành hàm $\max(\widetilde{M})$ của DSKD gốc**, làm vô nghĩa hoàn toàn cải tiến A3b!
   - **`[MANDATORY FIX]`:** Đổi ký hiệu thành **Tỷ lệ phân số đuôi $\rho_{\text{tail}} = 0.001$ (tương đương Top-$0.1\%$ diện tích ảnh = $65$ pixels trên lưới $256 \times 256$)**. Xóa bỏ cách viết dễ gây hiểu nhầm `Top-q% (q=0.001)` trong toàn bộ tài liệu.
5. **Câu hỏi 5: Kiểm tra Z-score calibration trên `val/good`, ngưỡng sàn $\sigma$, và khóa siêu tham số:**
   - $(\mu_s^k, \sigma_s^k)$ được ước lượng duy nhất trên `val/good` ở chế độ `torch.no_grad()` và `model.eval()`.
   - Áp ngưỡng sàn $\sigma_s^k \leftarrow \max(\sigma_s^k, \sigma_{\min})$ với $\sigma_{\min} = 10^{-3}$ trước khi chia Z-score để ngăn chặn sự thổi phồng nhiễu số học ở tầng sâu $k=3$.
   - Toàn bộ siêu tham số hợp nhất ($\alpha = (0.25, 0.50, 0.75)$, $\eta = 0.3$, $\zeta = 0.0$, $\rho_{\text{tail}} = 0.001$) được khóa cố định từ lý thuyết trước khi chạm vào tập `test/`.

---

## 3. Kiểm Toán Kỹ Thuật Phương Pháp B (`Lite-DSKD`)

### 3.1. Kiểm toán Mô-đun B1 (`Lite-DFE`: `DS-MSA` + `BDRF`)

1. **Kiểm tra kích thước Tensor qua từng nhánh của `Lite-DFE` (với ảnh đầu vào $B \times 3 \times 256 \times 256$):**
   - Đầu vào từ $S_e^{(\alpha)}$ (với hệ số độ rộng kênh $\alpha \in \{1.0, 0.75, 0.50\}$, tức $c_k^{(e)} = \alpha c_k$ với $(c_1, c_2, c_3) = (64, 128, 256)$):
     - Nhánh $k=1$: $G_e^1 \in \mathbb{R}^{B \times 64\alpha \times 64 \times 64} \xrightarrow{\text{DSBlock}_{s=2}^{(1,1)}} \mathbb{R}^{B \times 128\alpha \times 32 \times 32} \xrightarrow{\text{DSBlock}_{s=2}^{(1,2)}} \tilde{G}_e^1 \in \mathbb{R}^{B \times 256\alpha \times 16 \times 16}$.
     - Nhánh $k=2$: $G_e^2 \in \mathbb{R}^{B \times 128\alpha \times 32 \times 32} \xrightarrow{\text{DSBlock}_{s=2}^{(2,1)}} \tilde{G}_e^2 \in \mathbb{R}^{B \times 256\alpha \times 16 \times 16}$.
     - Nhánh $k=3$: $\tilde{G}_e^3 = G_e^3 \in \mathbb{R}^{B \times 256\alpha \times 16 \times 16}$.
   - Ghép kênh (`torch.cat(..., dim=1)` tại độ phân giải $16 \times 16$):
     $$U = \text{Concat}(\tilde{G}_e^1, \tilde{G}_e^2, \tilde{G}_e^3) \in \mathbb{R}^{B \times 768\alpha \times 16 \times 16}$$
   - Đi qua khối `BDRF` (với số kênh đầu ra mục tiêu $c_{\text{emb}} = 256\alpha$ và nút cổ chai giữa $d_{\text{mid}} = \rho \cdot 256\alpha$):
     - Nhánh chính: $U (768\alpha \times 16^2) \xrightarrow{\text{PWConv}_{1\times 1} + \text{BN} + \text{ReLU}} Z_{\text{mid}} (d_{\text{mid}} \times 16^2) \xrightarrow{\text{DWConv}_{3\times 3, s=1} + \text{BN} + \text{ReLU}} Z_{\text{dw}} (d_{\text{mid}} \times 16^2) \xrightarrow{\text{PWConv}_{1\times 1} + \text{BN}} Z_{\text{out}} (256\alpha \times 16^2)$.
     - Nhánh tắt (Residual Shortcut): $U (768\alpha \times 16^2) \xrightarrow{\text{PWConv}_{1\times 1} + \text{BN}} Z_{\text{skip}} (256\alpha \times 16^2)$.
     - Đầu ra: $F_{\text{emb}} = \text{ReLU}(Z_{\text{out}} + Z_{\text{skip}}) \in \mathbb{R}^{B \times 256\alpha \times 16 \times 16}$, khớp chính xác 100% với đầu vào tầng đầu tiên (`rev_layer3`) của $S_d^{(\alpha)}$.

2. **Tính toán lại độc lập số lượng Tham số (Parameters) và FLOPs của `Lite-DFE` (tại $\alpha = 1.0$):**
   - Trong mỗi lớp có kèm `BatchNorm2d(C)`, số tham số của `BN` là $2C$ ($\gamma, \beta$):
     - $\text{DSBlock}^{(1,1)} (64 \to 128, 32\times 32)$: $\text{DW}_{3\times 3}(64\times 9) + \text{BN}(128) + \text{PW}_{1\times 1}(64\times 128) + \text{BN}(256) = 576 + 128 + 8,192 + 256 = \mathbf{9,152}$ params; FLOPs $= 2 \times (576 + 8,192) \times 32^2 = \mathbf{17.96\text{M}}$ FLOPs.
     - $\text{DSBlock}^{(1,2)} (128 \to 256, 16\times 16)$: $\text{DW}_{3\times 3}(128\times 9) + \text{BN}(256) + \text{PW}_{1\times 1}(128\times 256) + \text{BN}(512) = 1,152 + 256 + 32,768 + 512 = \mathbf{34,688}$ params; FLOPs $= 2 \times (1,152 + 32,768) \times 16^2 = \mathbf{17.37\text{M}}$ FLOPs.
     - $\text{DSBlock}^{(2,1)} (128 \to 256, 16\times 16)$: $\mathbf{34,688}$ params; FLOPs $= \mathbf{17.37\text{M}}$ FLOPs.
     - **Tổng phần `DS-MSA` ($\alpha=1.0$):** $9,152 + 34,688 + 34,688 = \mathbf{78,528}$ params (`0.0785M`), **`52.70M` FLOPs** (`0.0527G`).
     - **Khối `BDRF` ($\alpha=1.0, \rho=0.5 \implies d_{\text{mid}}=128, c_{\text{emb}}=256$):**
       - $\text{PW}_{\text{reduce}} (768 \to 128) + \text{BN}(128)$: $768\times 128 + 256 = 98,560$ params.
       - $\text{DW}_{\text{ctx}} (128, 3\times 3) + \text{BN}(128)$: $128\times 9 + 256 = 1,408$ params.
       - $\text{PW}_{\text{expand}} (128 \to 256) + \text{BN}(256)$: $128\times 256 + 512 = 33,280$ params.
       - $\text{PW}_{\text{skip}} (768 \to 256) + \text{BN}(256)$: $768\times 256 + 512 = 197,120$ params.
       - **Tổng khối `BDRF` ($\alpha=1.0, \rho=0.5$):** $98,560 + 1,408 + 33,280 + 197,120 = \mathbf{330,368}$ params (`0.3304M`), FLOPs $= 2 \times 328,832 \times 16^2 = \mathbf{168.36\text{M}}$ FLOPs (`0.1684G`).
     - **Phân định hai biến thể thiết kế `Lite-DFE`:**
       - **Biến thể A: `dsmsa_db` (Chuẩn đề xuất trong `method_b_proposal.md` — Canonical Method B):** Nhánh chiếu về $d_b = 64\alpha \implies 3d_b = 192\alpha$ concat kênh. Tại $\alpha=1.0, \rho=0.5$: có **`141,056` params (`0.141M`)** (`0.072G` FLOPs / `0.036G` MACs). Tại $\alpha=0.50, \rho=1.0$: có **`51,904` params (`0.052M`)** (`0.026G` FLOPs / `0.013G` MACs). Tại $\alpha=0.50, \rho=0.5$: có **`36,736` params (`0.037M`)** (`0.019G` FLOPs / `0.009G` MACs) — con số này **đã tính đầy đủ** nhánh tắt $\text{PW}_{\text{skip}}$ ($96 \to 128 \implies 12,288$ conv + $256$ BN).
       - **Biến thể B: `dsmsa_expand` (Biến thể mở rộng kênh kiểm chứng Ablation):** Nhánh hạ mẫu mở rộng lên $256\alpha \implies 768\alpha$ concat kênh. Tại $\alpha=1.0, \rho=0.5$: có **`408,896` params (`0.409M`)** (`0.221G` FLOPs). Tại $\alpha=0.50, \rho=1.0$: có **`137,696` params (`0.138M`)** (`0.073G` FLOPs). Tại $\alpha=0.50, \rho=0.5$: có **`104,096` params (`0.104M`)** (`0.056G` FLOPs).
       - Cả hai biến thể đều giảm sâu từ **`87.3%` đến `98.9%` tham số khối DFE** so với `Dense-DFE` (`3.288M` params). Nhóm chọn `dsmsa_db` làm cấu hình chuẩn chính và giữ `dsmsa_expand` làm bước đối chứng Ablation trong Group 3 `B1`.

3. **Thế nào là phép so sánh công bằng (Fair Baseline Comparison) giữa `Lite-DFE` và `DSKD` gốc?**
   - Trong `papers/2402.00448v1.pdf` (`Fig. 4`, tr. 5), tác giả vẽ khối `DFE` gồm các lớp `3x3 conv, stride=2` nối vào `Concat` rồi qua khối Residual có nhánh tắt `1x1 conv`, nhưng bài báo không ghi rõ các lớp `3x3 conv, stride=2` có nhân đôi số kênh ($64 \to 128 \to 256$, cho ra $C_{\text{cat}} = 768$) hay giữ nguyên số kênh ($64, 128, 256$, cho ra $C_{\text{cat}} = 448$).
   - Với $C_{\text{cat}} = 768 \to 256$: `Dense-DFE` có **`3.222M` params** (`1.651G` FLOPs).
   - Với $C_{\text{cat}} = 448 \to 256$: `Dense-DFE` có **`1.968M` params** (`1.009G` FLOPs).
   - **`[MANDATORY FIX]`:** Báo cáo rõ cấu hình kênh của `Dense-DFE` trong `B0-Clean` ($C_{\text{cat}}=768$, `3.222M` params) và chỉ so sánh phần trăm giảm tham số của `Lite-DFE` trên **cùng quy ước nhân đôi kênh** ($768\alpha \to 256\alpha$) bằng cùng một hàm đo `count_parameters()`.

---

### 3.2. Kiểm toán Mô-đun B2 (`Slim-DSKD`: Thu hẹp Kênh $\alpha$ & Đầu chiếu $\mathcal{P}_e^k, \mathcal{P}_d^k$)

1. **Kiểm tra số kênh ở từng Stage và tính tương thích:**
   | Tầng / Khối | Teacher $T$ (Frozen) | Student $S_e^{(\alpha)}, S_d^{(\alpha)}$ ($\alpha=0.75$) | Student $S_e^{(\alpha)}, S_d^{(\alpha)}$ ($\alpha=0.50$) | Đầu chiếu $\mathcal{P}_e^k, \mathcal{P}_d^k$ ($\alpha c_k \to c_k$) | Độ phân giải ($256\times 256$ input) |
   | :--- | :---: | :---: | :---: | :---: | :---: |
   | `stem` + `layer1` / `rev_layer1` ($k=1$) | $64$ | $48$ | $32$ | $\mathbb{R}^{48 \text{ hoặc } 32} \to \mathbb{R}^{64}$ | $64 \times 64$ |
   | `layer2` / `rev_layer2` ($k=2$) | $128$ | $96$ | $64$ | $\mathbb{R}^{96 \text{ hoặc } 64} \to \mathbb{R}^{128}$ | $32 \times 32$ |
   | `layer3` / `rev_layer3` ($k=3$) | $256$ | $192$ | $128$ | $\mathbb{R}^{192 \text{ hoặc } 128} \to \mathbb{R}^{256}$ | $16 \times 16$ |
   | `DFE` Concat $\to$ Output ($F_{\text{emb}}$) | — | $576 \to 192$ | $384 \to 128$ | — | $16 \times 16$ |

2. **`[AUDIT FINDING - ERROR/MISMATCH]` Phản biện sâu sắc việc thêm `ReLU` sau `BN` ở các đầu chiếu $\mathcal{P}_e^k, \mathcal{P}_d^k$:**
   - Trong Vòng 2, Agent 1 và Agent 4 quyết định dùng $\mathcal{P}_s^k(G_s^k) = \text{ReLU}(\text{BN}(\text{Conv}_{1\times 1}(G_s^k)))$ vì đầu ra khối `BasicBlock` của `torchvision.models.resnet18` đi qua `ReLU` cuối cùng ($F_t^k \ge 0$).
   - **Phân tích toán học độc lập của Kiểm toán viên về rủi ro của `BN + ReLU` trong đầu chiếu mở rộng kênh ($\mathbb{R}^{\alpha c_k} \to \mathbb{R}^{c_k}$):**
     - **Rủi ro 1 (Kênh chết — Dead Projected Channels & Sụp đổ hạng):** Lớp `BatchNorm2d` chuẩn hóa mỗi kênh đầu ra về trung bình $\beta_c$ (khởi tạo bằng $0$). Khi đặt `ReLU(z) = max(0, z)` ngay sau `BN`, một nửa miền giá trị âm bị cắt cụt về $0$ với đạo hàm bằng $0$. Nếu trong những bước đầu huấn luyện, hệ số dịch $\beta_c$ của một kênh giảm xuống âm hoặc các vector chiếu rơi vào nửa không gian âm đối với các mẫu bình thường, kênh đó sẽ bị **chết vĩnh viễn** ($\nabla \text{ReLU} = 0$), làm giảm thêm hạng biểu diễn vốn đã bị nén ($\text{rank} \le \alpha c_k$)!
     - **Rủi ro 2 (Mất thông tin ở miền gần $0$):** Đối với các kênh của Teacher $T$ có kích hoạt dương nhỏ trên ảnh bình thường ($F_{t,c}^k \approx 0.05$) nhưng tăng vọt trên ảnh lỗi, một lớp chiếu tuyến tính `Conv1x1 + BN` (hoặc `Conv1x1 + BN + LeakyReLU(0.01)`) cho phép khớp trơn tru quanh $0$ mà không bị ngắt đạo hàm đột ngột tại $0$.
   - **`[MANDATORY FIX]` Bắt buộc thực hiện Phép Thử Ablation Hàm Kích Hoạt Đầu Chiếu (`Projection Activation Ablation`):**
     Không được mặc định `BN+ReLU` là chân lý duy nhất! Trong Group 3 (Method B), cài đặt tham số cấu hình `proj_activation: Literal["relu", "leaky_relu", "none"]` và chạy đối chứng 3 biến thể trên Core-5:
     1. `B2-BN-Only` (`proj_activation="none"`): $\mathcal{P}_s^k = \text{BN}(\text{Conv}_{1\times 1}(G_s^k))$.
     2. `B2-BN-ReLU` (`proj_activation="relu"`): $\mathcal{P}_s^k = \text{ReLU}(\text{BN}(\text{Conv}_{1\times 1}(G_s^k)))$.
     3. `B2-BN-LeakyReLU` (`proj_activation="leaky_relu"`): $\mathcal{P}_s^k = \text{LeakyReLU}_{0.01}(\text{BN}(\text{Conv}_{1\times 1}(G_s^k)))$ (vừa ép miền âm co nhỏ $100$ lần để tương thích góc với $F_t^k \ge 0$, vừa giữ đạo hàm $0.01 \ne 0$ chống chết kênh tuyệt đối!).

3. **`[MANDATORY FIX]` Bảng Phân Tách Tuyệt Đối 5 Đại Lượng Độ Phức Tạp và Mẫu Số So Sánh:**
   - Kiểm toán viên phát hiện các tài liệu Vòng 1–2 đã trộn lẫn mẫu số khi viết câu *"Giảm 89.1% tổng tham số (từ 39.3M xuống 4.30M)"*. Tại sao phép so sánh này **không công bằng nếu đứng một mình**? Bởi vì con số `39.30M` trong `Table 6` của bài báo gốc nhiều khả năng bao gồm cả tầng `layer4` ($512$ kênh, `8.39M` tham số/mạng $\times 3 = 25.18\text{M}$) không tham gia vào đồ thị tính toán $K=3$ tầng, trong khi con số `4.30M` của Agent 4 đã cắt bỏ `layer4`!
   - Do đó, **bắt buộc** mọi bảng báo cáo của Method B phải tách riêng **3 mẫu số tham chiếu** và **5 chỉ số độ phức tạp** như bảng chuẩn hóa đã kiểm toán dưới đây:

| Cấu hình Mô hình | **1. Total Params (M)** *(T + Se + DFE + Sd)* | **2. Trainable Params (M)** *(Se + DFE + Sd)* | **3. Inference Params (M)** *(Luồng chạy Algo. 2)* | **4. Total Inference FLOPs (G)** *(Ảnh $256\times 256$)* | **5. Student+DFE FLOPs (G)** *(Trừ Teacher $T$)* | **% Giảm Trainable Params vs. `B0-Clean`** *(So sánh Công bằng Nhất)* | **% Giảm Total FLOPs vs. `B0-Clean`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B0-Paper` (`Table 6, p. 10`)** | `39.300 M` *(Reported)* | *Không công bố tách rời* | `39.300 M` *(Reported)* | `6.170 G` *(Reported)* | — | — | — |
| **`B0-FullResNet18` *(Giữ `layer4` tĩnh)*** | `36.753 M` | `8.788 M` | `11.571 M` *(hoặc `36.753M`)* | `6.379 G` | `4.795 G` | `0.0%` | `0.0%` |
| **`B0-Clean` *(Chuẩn 3-Stage + Dense-DFE)*** | **`11.571 M`** | **`8.788 M`** | **`11.571 M`** | **`6.379 G`** | **`4.795 G`** | **`0.0%` (Mẫu số Chuẩn)** | **`0.0%` (Mẫu số Chuẩn)** |
| **`B1: Lite-DFE` ($\alpha=1.0, \rho=0.5$)** | `8.758 M` | `5.975 M` | `8.758 M` | `4.949 G` | `3.365 G` | **`-32.0%`** | **`-22.4%`** |
| **`B2: Slim-DSKD` ($\alpha=0.75$, Dense-DFE)** | `7.822 M` | `5.039 M` | `7.822 M` | `4.642 G` | `3.058 G` | **`-42.7%`** | **`-27.2%`** |
| **`B2: Slim-DSKD` ($\alpha=0.50$, Dense-DFE)** | `5.073 M` | `2.290 M` | `5.073 M` | `3.015 G` | `1.431 G` | **`-73.9%`** | **`-52.7%`** |
| **`B3: Lite-DSKD-0.75` ($\alpha=0.75, \rho=0.5$)** | **`6.240 M`** | **`3.457 M`** | **`6.240 M`** | **`3.838 G`** | **`2.254 G`** | **`-60.7%`** *(và `-84.1%` vs `39.3M`)* | **`-39.8%`** |
| **`B3: Lite-DSKD-0.50` ($\alpha=0.50, \rho=1.0$)** | **`4.369 M`** | **`1.586 M`** | **`4.369 M`** | **`2.658 G`** | **`1.074 G`** | **`-82.0%`** *(và `-88.9%` vs `39.3M`)* | **`-58.3%`** |

*(Ghi chú kiểm toán: Trong pha suy luận gốc `Algo. 2`, chỉ cần tính $M_{t-d}^k$ nên nhánh đầu chiếu $\mathcal{P}_e^k$ của $S_e^{(\alpha)}$ có thể bỏ qua nếu chạy độc lập Method B, tiết kiệm thêm `0.022M` params; khi kết hợp với Method A trong `C1`, cả $T, S_e^{(\alpha)}, \text{Lite-DFE}, S_d^{(\alpha)}$ đều tham gia suy luận nên `Inference Params == Total Params`).*

---

## 4. Kiểm Toán Tổng Quan Tài Liệu (`literature_review.md` & `research_gap_analysis.md`)

### 4.1. Thống kê số lượng và Bảng Xác Minh Toàn Bộ 15 Bài Báo Trong Kho Dữ Liệu

Kiểm toán viên đã mở trực tiếp trang đầu của toàn bộ **15 tệp PDF** trong thư mục `papers/` để xác minh từng tiêu đề, danh sách tác giả, năm, nơi công bố/arXiv ID và trạng thái xuất bản:

| STT | Tiêu đề Chính xác (Đã xác minh từ PDF gốc) | Tác giả Chính xác | Năm | Nơi công bố / Mã arXiv | Phân loại Trạng thái | Tệp PDF Đã Xác Minh trong `papers/` |
| :---: | :--- | :--- | :---: | :--- | :---: | :--- |
| **0** | *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection* (DSKD — **Baseline**) | Liyi Yao, Shaobing Gao | 2024 | `arXiv:2402.00448v1` (01/02/2024) | **arXiv Preprint** | `papers/2402.00448v1.pdf` |
| **1** | *Student-Teacher Feature Pyramid Matching for Anomaly Detection* (STFPM / STPM) | Guodong Wang, Shumin Han, Errui Ding, Di Huang | 2021 | **BMVC 2021** (`arXiv:2103.04257v3`) | **Published Conference** | `papers/2021_BMVC_STFPM_Feature_Pyramid_Matching.pdf` |
| **2** | *Anomaly Detection via Reverse Distillation from One-Class Embedding* (RD4AD) | Hanqiu Deng, Xingyu Li | 2022 | **CVPR 2022** (pp. 9737–9746, `arXiv:2201.10703v2`) | **Published Conference** | `papers/2022_CVPR_RD4AD_Reverse_Distillation.pdf` |
| **3** | *Remembering Normality: Memory-guided Knowledge Distillation for Unsupervised Anomaly Detection* (MemKD) | Zhihao Gu, Liang Liu, Xu Chen, Ran Yi, Jiangning Zhang et al. | 2023 | **ICCV 2023** (pp. 16401–16409, `arXiv:2308.05939v1`) | **Published Conference** | `papers/2023_ICCV_MemKD_Memory_Guided_Knowledge_Distillation.pdf` |
| **4** | *Revisiting Reverse Distillation for Anomaly Detection* (RD++) | Tran Dinh Tien, Anh Tuan Nguyen, Nguyen Hoang Tran et al. | 2023 | **CVPR 2023** (pp. 24511–24520) | **Published Conference** | `papers/2023_CVPR_RDPlusPlus_Revisiting_Reverse_Distillation.pdf` |
| **5** | *Contrastive Knowledge Distillation for Anomaly Detection in Multi-Illumination/Focus Display Images* (CKD / MCD) | Jihyun Lee, Hangil Park, Yongmin Seo, Taewon Min et al. | 2023 | **MVA 2023** (`DOI:10.23919/MVA57639.2023.10215847`; bản lưu arXiv ghi `2609.05520v1`) | **Published Conference** *(Xác nhận tại Ref. [16] tr. 11 của `2509.24448v1`)* | `papers/2026_arXiv_CKD_Contrastive_Knowledge_Distillation_Display_IAD.pdf` |
| **6** | *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies* | Kilian Batzner, Lars Heckler, Rebecca König | 2024 | **WACV 2024** (pp. 128–138, `arXiv:2303.14535v2`) | **Published Conference** | `papers/2024_WACV_EfficientAD_Millisecond_Anomaly_Detection.pdf` |
| **7** | *Context-Aware Anomaly Detection using Knowledge Distillation* (CAD) |Jie Zhang, Masanori Suganuma, Takayuki Okatani *(hoặc bản arXiv `2401.00631v1`)* | 2024 | **WACV 2024** / `arXiv:2401.00631v1` | **arXiv Preprint / Conference** | `papers/2024_arXiv_CAD_Context_Aware_Knowledge_Distillation.pdf` |
| **8** | *Deep Industrial Image Anomaly Detection: A Survey* | Jiaqi Liu, Guoyang Xie, Jinbao Wang, Shangnian Li, Chengjie Wang, Feng Zheng, Yaochu Jin | 2024 | **Machine Intelligence Research (MIR)**, Vol. 21, pp. 104–135 (`arXiv:2301.11514v3`) | **Published Journal** | `papers/2024_Survey_Deep_Industrial_Image_Anomaly_Detection.pdf` |
| **9** | *SK-RD4AD: Skip-Connected Reverse Distillation For Robust One-Class Anomaly Detection* | Eunju Park,omin Kim, Wonjun Hwang | 2025 | **CVPRW 2025 (VAND 3.0 Workshop)** (`arXiv:2507.17266v1`) | **Published Workshop** | `papers/2025_CVPRW_SK_RD4AD_Skip_Connected_Reverse_Distillation.pdf` |
| **10** | *Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection* | Jia Guo, Shuai Lu, Weihang Zhang, Fang Chen, Hongen Liao, Huiqi Li | 2025 | **CVPR 2025** (`arXiv:2405.14325v3`) | **Published Conference** | `papers/2025_CVPR_Dinomaly_MultiClass_Anomaly_Detection.pdf` |
| **11** | *Tuned Reverse Distillation: Enhancing Multimodal Industrial Anomaly Detection with Crossmodal Tuners* (TRD) | Xinyue Liu, Jianyuan Wang, Biao Leng, Shuo Zhang | 2025/2026 | `arXiv:2412.08949v4` (`https://github.com/hito2448/TRD`) | **arXiv Preprint** | `papers/2025_arXiv_Crossmodal_Reverse_Distillation_IAD.pdf` |
| **12** | *Generalist Multi-Class Anomaly Detection via Distillation to Two Heterogeneous Student Networks* | Hangil Park, Yongmin Seo, Tae-Kyun Kim | 2025 | `arXiv:2509.24448v1` (29/09/2025) | **arXiv Preprint** | `papers/2025_arXiv_Dual_Heterogeneous_Student_KD_IAD.pdf` |
| **13** | *PDD: Manifold-Prior Diverse Distillation for Medical Anomaly Detection* | Xijun Lu, Hongying Liu, Fanhua Shang, Yanming Hui, Liang Wan | 2026 | `arXiv:2603.07142v1` (07/03/2026, `https://github.com/OxygenLu/PDD`) | **arXiv Preprint** *(Lưu ý: chỉ là arXiv preprint, chưa có dòng "Published at CVPR" trên PDF)* | `papers/2026_CVPR_PDD_Manifold_Prior_Diverse_Distillation.pdf` |
| **14** | *Dual Distillation for Few-Shot Anomaly Detection* ($\text{D}^2\text{4FAD}$) | Le Dong, Qinzhong Tan, Chunlei Li, Jingliang Hu, Yilei Shi, Weisheng Dong, Xiao Xiang Zhu, Lichao Mou | 2026 | **ICLR 2026** (`arXiv:2603.01713v1`, có header *"Published as a conference paper at ICLR 2026"*) | **Published Conference** | `papers/2026_ICLR_D24FAD_Dual_Distillation_Anomaly_Detection.pdf` |

---

### 4.2. Phát hiện & Sửa lỗi Trích dẫn Nghiêm trọng giữa `literature_review.md` và `research_gap_analysis.md`

**`[AUDIT FINDING - ERROR/MISMATCH]`:** Trong khi tệp `docs/research/literature_review.md` ghi chép **đúng 100%** tiêu đề và tác giả từ các tệp PDF, thì trong tệp `docs/research/research_gap_analysis.md`, Agent 2 đã tự suy diễn tên bài báo dựa theo tên file PDF và mắc **4 lỗi trích dẫn sai lệch sau**:
1. **Lỗi 1 (`research_gap_analysis.md` dòng 5):** Ghi tên bài báo DSKD (`arXiv:2402.00448v1`) thành *"A Smaller Model Learns Better: Dual-Student Knowledge Distillation for Anomaly Detection"* — đây là tên bịa đặt, tên thật ở trang 1 của `2402.00448v1.pdf` là **"Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection"**.
2. **Lỗi 2 (`research_gap_analysis.md` dòng 38 & 99):** Ghi `arXiv:2509.24448v1` là *"Dual Heterogeneous Student Knowledge Distillation for Anomaly Detection"* của *"SeongJe Park et al."* — thực tế trang 1 của PDF là **"Generalist Multi-Class Anomaly Detection via Distillation to Two Heterogeneous Student Networks"** của **Hangil Park, Yongmin Seo, Tae-Kyun Kim (KAIST)**.
3. **Lỗi 3 (`research_gap_analysis.md` dòng 40 & 100):** Ghi `arXiv:2603.07142v1` là *"Progressive Dual-Student Knowledge Distillation for Multi-Class Anomaly Detection (PDD)"* của *"Shun Lu et al., CVPR 2026"* — thực tế trang 1 của PDF là **"PDD: Manifold-Prior Diverse Distillation for Medical Anomaly Detection"** của **Xijun Lu, Hongying Liu, Fanhua Shang, Yanming Hui, Liang Wan (Tianjin University)** và là bản **arXiv Preprint** trên ảnh y tế.
4. **Lỗi 4 (`research_gap_analysis.md` dòng 101):** Ghi `arXiv:2412.08949v4` là *"Triplet Reverse Distillation for Unsupervised Anomaly Detection (TRD)"* của *"Qi Liu et al."* — thực tế trang 1 của PDF là **"Tuned Reverse Distillation: Enhancing Multimodal Industrial Anomaly Detection with Crossmodal Tuners (TRD)"** của **Xinyue Liu, Jianyuan Wang, Biao Leng, Shuo Zhang**.

**`[MANDATORY FIX]`:** Cập nhật trực tiếp `docs/research/research_gap_analysis.md` để sửa toàn bộ 4 lỗi tiêu đề/tác giả này khớp 100% với PDF gốc.

---

### 4.3. Kiểm toán Tính Mới (Novelty Audit) của `FC-DSKD` và `Lite-DSKD` so với các bài báo gần nhất

1. **Đối chiếu `FC-DSKD` (Method A) với Park et al. (`arXiv:2509.24448v1`, 2025) và Liu et al. (`TRD, arXiv:2412.08949v4`):**
   - *Điểm tương đồng cần ghi nhận trung thực:* Cả Park et al. (2025) và Liu et al. (TRD, Eq. 11 tr. 7) đều hợp nhất điểm bất thường từ hai nhánh: Park et al. dùng hàm xác suất `Noisy-OR` trên ViT-Base/14 (`233.1M` params) nối song song (không nối tiếp qua `DFE`), còn Liu et al. (TRD, Eq. 11 tr. 6–7) dùng chuẩn hóa Z-score trên tập validation ($\mathcal{M}_{\text{norm}}^{2D/3D} = \frac{\mathcal{M}^{2D/3D} - \mu^{2D/3D}}{\sigma^{2D/3D}}$) để cộng hai nhánh đa phương thức RGB + 3D Depth!
   - *Điểm khác biệt cốt lõi & đóng góp của `FC-DSKD`:*
     - Chúng tôi **không tuyên bố** chuẩn hóa Z-score trên `val/good` là phát minh hoàn toàn mới trong IAD, mà trích dẫn tường minh Bergmann et al. (IJCV 2022) và Liu et al. (TRD, `arXiv:2412.08949v4`, Eq. 11) làm cơ sở lý thuyết!
     - Đóng góp đặc thù của `FC-DSKD` nằm ở việc **giải quyết sự bất đối xứng cấu trúc nối tiếp $T \to S_e \to \text{DFE} \to S_d$ bên trong DSKD một phương thức RGB**: (i) kết hợp sai biệt điểm + ngữ cảnh $3\times 3$ có trọng số tiêu cự không gian (`A1`), (ii) cô lập đạo hàm $\text{sg}[F_e^k]$ ở cả `DFE` và $\ell_{e-d}$ (`A2`), và (iii) hợp nhất theo độ sâu tầng $\alpha_k$ giữa nhánh mã hóa không qua nút cổ chai ($\widehat{M}_{t-e}^k$, ưu tiên ở tầng nông $k=1$) và nhánh giải mã sau nút cổ chai ($\widehat{M}_{t-d}^k$, ưu tiên ở tầng sâu $k=3$) kèm số hạng khuếch đại đồng thuận nhân tính (`A3`).
2. **Đối chiếu `Lite-DSKD` (Method B) với `RD4AD` (CVPR 2022), `SK-RD4AD` (CVPRW 2025) và `EfficientAD` (WACV 2024):**
   - `RD4AD` và `SK-RD4AD` dùng nút cổ chai đặc `OCBE/MFF` (hàng triệu tham số) và học sinh đối xứng kênh ($\alpha=1.0$); `EfficientAD` dùng mạng PDN nông không có kiến trúc mã hóa-giải mã đảo ngược đa tầng.
   - `Lite-DSKD` là thiết kế đầu tiên giải quyết trực tiếp bài toán **nén nút cổ chai `DFE` bằng tích chập tách chiều sâu (`DS-MSA + BDRF`, giảm $>87\%$ tham số `DFE`) kết hợp thu hẹp kênh bất đối xứng $\alpha \in \{0.50, 0.75\}$ cho hệ thống Học sinh Kép ($S_e^{(\alpha)}, S_d^{(\alpha)}$)**.

---

## 5. Kiểm Toán Thiết Kế Thực Nghiệm & Giao Thức Tái Lập Trên Kaggle (`experiment_design.md` & `kaggle_implementation_plan.md`)

### 5.1. Phát hiện & Sửa lỗi Đảo Nhãn `Method A` / `Method B` trong Tài liệu của Agent 5

- **`[AUDIT FINDING - ERROR/MISMATCH]`:** Trong `docs/research/experiment_design.md` (Mục 4.1–4.3, dòng 136–196) và `docs/research/kaggle_implementation_plan.md` (Mục 1.1–1.2, dòng 39–100), Agent 5 đã viết theo phác thảo tạm thời ở Vòng 1 nên **gán ngược `Method A` là cải tiến kiến trúc nhẹ `DFE` và `Method B` là cải tiến Loss + Calibrated Inference**, ngược hoàn toàn với `method_a_proposal.md` (`Method A = FC-DSKD`), `method_b_proposal.md` (`Method B = Lite-DSKD`), `method_comparison.md`, `research_decision.md` và `project_timeline.md`!
- **`[MANDATORY FIX]`:** Đồng bộ hóa tuyệt đối tên gọi và mã chạy trên toàn bộ tài liệu:
  - **Group 2 = Method A (`FC-DSKD` — Sinh viên 1):** `A1` (Contextual + Spatial-Focal Loss), `A2` (Asymmetric Cross-Student Consistency), `A3` (Normal-Val Calibrated Dual-Branch Fusion & Top-$\rho$ Scoring — chạy Zero-Retrain trên `B0`), và `A_full` (`FC-DSKD` đầy đủ).
  - **Group 3 = Method B (`Lite-DSKD` — Sinh viên 2):** `B1` (`Lite-DFE`), `B2` (`Slim-DSKD` + Ablation kích hoạt đầu chiếu `BN` vs `BN+ReLU` vs `BN+LeakyReLU`), `B3-0.75` (`Lite-DSKD` với $\alpha=0.75$), và `B3-0.50` (`Lite-DSKD` với $\alpha=0.50$).
  - **Group 4 = Combined (`FC-Lite-DSKD`):** `C1-0.75` và `C1-0.50`.

---

### 5.2. Kiểm toán Chia Tập `train/good` vs. `val/good` (`80/20`, `seed=42`) và Ảnh hưởng lên Tái lập Baseline

1. **Phân tích định lượng tác động của việc giữ lại 20% `train/good` làm `val/good`:**
   - Trong bài báo DSKD gốc (Mục 4.1.1, tr. 7), tác giả sử dụng 100% ảnh `train/good` để huấn luyện 200 epochs và không tách tập validation.
   - Khi tách 20% làm `val/good` (`val_split_ratio: 0.2, seed: 42` trong `src/anomaly_detection/data/mvtec.py`), số ảnh huấn luyện mỗi category giảm từ ~240 ảnh xuống ~192 ảnh; đặc biệt với category nhỏ nhất là `toothbrush` (tổng 60 ảnh `train/good`), tập `split="train"` chỉ còn **48 ảnh** và `split="val"` có **12 ảnh** (vẫn cung cấp $12 \times 16 \times 16 = 3,072$ vector điểm ảnh ở tầng $k=3$ và $12 \times 64 \times 64 = 49,152$ điểm ảnh ở tầng $k=1$ để ước lượng thống kê $(\mu_s^k, \sigma_s^k)$ với sai số chuẩn rất nhỏ).
   - Tuy nhiên, để đảm bảo việc mất 20% ảnh train không làm giảm điểm tái lập Baseline `B0` trên các lớp ít ảnh như `toothbrush`, kiểm toán viên bổ sung **Phép Kiểm Tra Độ Nhạy Cỡ Mẫu Huấn Luyện (`S0_split`):**
     - So sánh `B0 (80% train / 20% val)` với `B0-100% (100% train/good)` trên `toothbrush` và `tile`.
     - Lưu ý: Đối với `B0` và `Method B` (khi chạy độc lập theo `Algo. 2`), mô hình dùng checkpoint tại đúng `epoch = 200` và không cần ước lượng $(\mu_s^k, \sigma_s^k)$. Đối với `Method A` và `Combined C1`, thống kê $(\mu_s^k, \sigma_s^k)$ được ước lượng chuẩn trên tập `val/good` (20% tách rời); ngoài ra có thể đối chứng thêm việc ước lượng $(\mu_s^k, \sigma_s^k)$ bằng pass `eval()` trên `train/good` ở trường hợp `100% train` (cả hai đều đảm bảo **100% Zero Test-Set Leakage**).
2. **Hiệu chuẩn theo từng Category (Per-Category Calibration) vs. Khóa Siêu tham số Toàn cục (Global Hyperparameter Lock):**
   - Trong bài toán One-Class Unsupervised AD trên MVTec AD, mỗi category $c \in \{1..15\}$ là một mô hình độc lập được huấn luyện chỉ trên ảnh `train/good` của chính category đó. Do đó, bộ 18 hằng số thống kê $(\mu_{s,c}^k, \sigma_{s,c}^k)$ của Mô-đun A3 **được tính riêng cho từng category $c$ từ chính tập `val/good` của category $c$** (như trọng số BatchNorm của mô hình đó).
   - Ngược lại, mọi siêu tham số công thức ($\gamma=1.0, \omega_{\text{ctx}}=0.5, \beta=0.15, \alpha=(0.25,0.50,0.75), \eta=0.3, \rho_{\text{tail}}=0.001$) được **khóa cứng toàn cục (Globally Locked)** giống hệt nhau trên cả 15 categories, tuyệt đối không tinh chỉnh theo từng category!

---

### 5.3. Phân loại Rõ 3 Nhóm Chi phí Ablation (`Full-Retrain` vs. `Recalibration-Only` vs. `Inference-Only`)

Để minh bạch tuyệt đối trong thực nghiệm và tiết kiệm tối đa ngân sách GPU Kaggle, mọi cấu hình Ablation được phân vào đúng 1 trong 3 nhóm sau:

| Phân loại Chi phí | Định nghĩa Kỹ thuật | Các Thí nghiệm Thuộc Nhóm Này | Thời gian GPU / Category |
| :--- | :--- | :--- | :---: |
| **Loại 1: `Full-Retrain`** | Thay đổi kiến trúc mạng hoặc hàm mất mát huấn luyện $\implies$ Phải chạy đủ 200 epochs từ đầu (`seed=42`). | `S0_eq3_norm`, `S0_input_norm`, `B0-Clean`, `G0_TD`, `G0_noDFE`, `A1a`, `A1`, `A2`, `A1+A2`, `B1`, `B2` (3 biến thể kích hoạt đầu chiếu), `B3-0.75`, `B3-0.50`, `C1-0.75`, `C1-0.50` | `~5.5 – 6.5 phút` (Baseline/Method A)<br>`~3.5 – 4.5 phút` (Method B/C1) |
| **Loại 2: `Recalibration-Only`** | Giữ nguyên trọng số `.pth` đã train, chạy 1 pass `torch.no_grad()` qua `val/good` (để tính $\mu_s^k, \sigma_s^k$) rồi đánh giá trên `test/`. | `A3-Linear`, `A3-Synergistic`, `A3-SoftMaxUnion` chạy trên checkpoint của `B0-Clean` (`Cfg-5`) hoặc trên checkpoint của `A1+A2` (`Cfg-6, Cfg-7`) | `~1.0 phút` (**0 epochs train**) |
| **Loại 3: `Inference-Only`** | Không cần cả `val/good` lẫn train lại; chỉ thay đổi nhánh hoặc tầng lấy ra ở `Algo. 2` và chạy 1 pass trên `test/`. | `G0_TE` (chỉ $M_{t-e}$), `G0_ED` (chỉ $M_{e-d}$), `G0_M1`, `G0_M2`, `G0_M3` (Bảng 7 & Bảng 9 của bài báo), và khảo sát $\rho_{\text{tail}} \in \{0.0, 0.0005, 0.001, 0.005\}$ | `~45 giây` (**0 epochs train**) |

---

### 5.4. Kiểm toán Định nghĩa Toán học và Cài đặt của 10 Chỉ số Đánh giá

Kiểm toán viên đã đọc trực tiếp [`src/anomaly_detection/evaluation/metrics.py`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/src/anomaly_detection/evaluation/metrics.py) và [`src/anomaly_detection/utils/profiling.py`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/src/anomaly_detection/utils/profiling.py) để xác nhận:
1. **`Image-AUROC` (`auroc_img`) & `Image-AP` (`ap_img`):** Tính bằng `roc_auc_score(image_labels, image_scores)` và `average_precision_score(image_labels, image_scores)` trên toàn bộ ảnh kiểm thử của category.
2. **`Pixel-AUROC` (`auroc_pix`) & `Pixel-AP` (`ap_pix`):** Tính trên mảng phẳng toàn bộ pixel $(N \times 256 \times 256)$ của tập kiểm thử.
3. **`AUPRO@0.30` (`aupro_30`) & `AUPRO@0.05` (`aupro_05`):**
   - Trong [`metrics.py`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/src/anomaly_detection/evaluation/metrics.py#L142-L152), hàm `compute_aupro(anomaly_maps, gt_masks, max_fpr=...)` tìm các thành phần liên thông 8-connected (`scipy.ndimage.label(mask, structure=np.ones((3,3)))`), tính tỷ lệ phủ trung bình trên mọi vùng lỗi $\text{PRO}(\tau) = \frac{1}{|\mathcal{R}|}\sum_{r \in \mathcal{R}} \frac{|P(\tau) \cap C_r|}{|C_r|}$ theo từng ngưỡng $\tau$, cắt đường cong tại `max_fpr` ($0.30$ hoặc $0.05$), nội suy tuyến tính điểm biên tại đúng `max_fpr`, và **chia chuẩn hóa diện tích tích phân hình thang cho `max_fpr`**:
     $$\text{AUPRO@}\alpha = \frac{1}{\alpha}\int_0^\alpha \text{PRO}(\text{FPR})\,d\text{FPR} \in [0, 1], \quad \alpha \in \{0.30, 0.05\}$$
   - Công thức cài đặt trong repo hoàn toàn chính xác theo định nghĩa của Bergmann et al. (IJCV 2021).
4. **4 Chỉ số Hiệu quả Tính toán (`Params`, `FLOPs`, `Latency`, `Peak VRAM`):**
   - **`[MANDATORY FIX]` Cập nhật `src/anomaly_detection/utils/profiling.py` trong Giai đoạn Cài đặt:** Hàm `measure_student_flops()` hiện tại trong `profiling.py` (dòng 32–34) đang giả định đầu vào 3 tensor `e1(256ch), e2(512ch), e3(1024ch)` của WideResNet50 cũ. Khi bước vào Giai đoạn Cài đặt, bắt buộc bổ sung hàm `profile_dskd_model(model, img_size=256)` nhận đầu vào ảnh $(1, 3, 256, 256)$ để đo đồng thời `Total Params`, `Trainable Params`, `Inference Params`, `Total FLOPs`, `Student+DFE FLOPs`, `GPU/CPU Latency (ms/img)` và `Peak GPU VRAM (MB)`.

---

### 5.5. Kiểm toán Schema Checkpoint / Resume (`CHECKPOINT_SCHEMA_V2`) trên Kaggle

Để đảm bảo khả năng khôi phục tất định 100% khi phiên Kaggle bị ngắt giữa chừng (bao gồm cả trường hợp dùng Mixed Precision `AMP` hoặc DataLoader shuffle), kiểm toán viên nâng cấp `CHECKPOINT_SCHEMA_V1` của Agent 5 lên **`CHECKPOINT_SCHEMA_V2`** với 3 bổ sung bắt buộc:
1. Đóng gói các đầu chiếu `proj_e` và `proj_d` của Method B nằm bên trong `student_e` và `student_d` để `student_e.state_dict()` và `student_d.state_dict()` tự động lưu trọn vẹn trọng số và `running_mean / running_var` của các đầu chiếu.
2. Lưu thêm `amp_scaler_state_dict` (nếu bật `use_amp=True`) và `dataloader_seed_offset = base_seed + epoch` cho `torch.Generator` của `train_loader` để thứ tự mini-batch ở epoch $e+1$ sau khi resume trùng khớp 100% với khi chạy liên tục.
3. Lưu thêm `git_commit_sha` bên cạnh `config_sha256` và `val_calibration_stats` (chứa $\mu_s^k, \sigma_s^k$ của từng tầng $k \in \{1, 2, 3\}$ và từng nhánh $s \in \{t\text{-}e, t\text{-}d, e\text{-}d\}$).

---

## 6. Bảng Quyết Định Kiểm Toán Chính Thức (Formal Audit Decisions)

| Hạng mục Kiểm toán | Phán quyết Kiểm toán | Điều kiện / Sửa đổi Bắt buộc Đã Khóa (`Mandatory Conditions`) |
| :--- | :---: | :--- |
| **1. Baseline DSKD & Kiểm định `Eq. (3)`** | **`APPROVED WITH CONDITIONS`** | • Tách biệt phép kiểm tra chuẩn hóa đặc trưng `S0_eq3_norm` (`Unit-L2` vs. `Raw-L2+Cosine` vs. `Literal Inv-Sq L2`) và phép kiểm tra tiền xử lý ảnh đầu vào `S0_input_norm` (`ImageNet Z-score` vs. `Image L2`).<br>• Báo cáo minh bạch cả `B0-Paper` (`39.30M` params, `6.170G` MACs) và `B0-Clean` (`12.180M` total / `9.397M` trainable params, `13.485G` FlopCounter FLOPs / `6.742G` MACs). |
| **2. Method A — Mô-đun A1 (Contextual + Spatial-Focal Loss)** | **`APPROVED WITH CONDITIONS`** | • Dùng $\epsilon = 10^{-12}$ khi tái chuẩn hóa $\hat{G}_z^k$.<br>• Khóa $\gamma = 1.0$, kẹp trần $w_{\max} = 5.0$, và tách riêng Ablation `Only Contextual (A1a)` vs. `Contextual + Focal (A1)` để kiểm soát rủi ro khuếch đại nhiễu nền. |
| **3. Method A — Mô-đun A2 (Cross-Student Consistency $\ell_{e-d}$)** | **`APPROVED WITH CONDITIONS`** | • **Bắt buộc** ngắt đạo hàm tại đầu vào `DFE` (`[f.detach() for f in raw_f_e]`) VÀ tách riêng 2 bộ tối ưu `optimizer_e`, `optimizer_d` để $\ell_d$ và $\ell_{e-d}$ tuyệt đối không truyền ngược về $S_e$.<br>• Kiểm chứng độc lập qua Ablation `A1` vs. `A1+A2`; chỉ giữ $\beta > 0$ nếu cải thiện hoặc không làm giảm kết quả trên Core-5. |
| **4. Method A — Mô-đun A3 (Calibrated Fusion & Top-$\rho$ Score)** | **`APPROVED WITH CONDITIONS`** | • Đổi tên chính xác thành **Synergistic Fusion = Complementary Linear Union + Concordance Multiplicative Boost** (không gọi tích số là complementarity).<br>• Chuẩn hóa ký hiệu tỷ lệ đuôi là **phân số diện tích $\rho_{\text{tail}} = 0.001$ ($0.1\%$ số pixel = 65 pixels ở $256\times 256$)**.<br>• Đối chứng 3 quy tắc hợp nhất (`Linear`, `Synergistic`, `SoftMaxUnion`) bằng `Recalibration-Only` tốn $0$ phút train. |
| **5. Method B — Mô-đun B1 (`Lite-DFE`: `DS-MSA + BDRF`)** | **`APPROVED WITH CONDITIONS`** | • Khóa biến thể canonical `dsmsa_db` ($192\alpha$ concat, $141\text{k}$ params ở $\alpha=1.0$; $51.9\text{k}$ ở $\alpha=0.5, \rho=1.0$; $36.7\text{k}$ ở $\alpha=0.5, \rho=0.5$ đã gồm $\text{PW}_{\text{skip}}$) và giữ `dsmsa_expand` ($768\alpha$ concat, $408.9\text{k}$ / $137.7\text{k}$ / $104.1\text{k}$) làm bước đối chứng ablation trong `B1`.<br>• Luôn lấy `B0-Clean` làm mẫu số chuẩn nội bộ. |
| **6. Method B — Mô-đun B2 (`Slim-DSKD` & Đầu chiếu $\mathcal{P}_{e,d}^k$)** | **`APPROVED WITH CONDITIONS`** | • Không áp đặt mù quáng `BN+ReLU`; bắt buộc chạy Ablation 3 hàm kích hoạt đầu chiếu (`BN-only` vs. `BN+ReLU` vs. `BN+LeakyReLU(0.01)`) trên Core-5 để phòng ngừa chết kênh.<br>• Báo cáo tách bạch 5 chỉ số độ phức tạp (`Total Params`, `Trainable Params`, `Inference Params`, `FLOPs`, `Latency`) với mẫu số rõ ràng. |
| **7. Literature Review & Research Gap Analysis** | **`APPROVED WITH CONDITIONS`** | • Đã xác minh trực tiếp 15 PDF trong `papers/` và sửa toàn bộ 4 lỗi sai tiêu đề/tác giả trong `research_gap_analysis.md`. |
| **8. Experiment Design & Kaggle Implementation Plan** | **`APPROVED WITH CONDITIONS`** | • Đã sửa lỗi đảo nhãn `Method A` / `Method B` trong `experiment_design.md` và `kaggle_implementation_plan.md`.<br>• Bổ sung phép kiểm tra `S0_split` (`80/20` vs `100% train`) trên `toothbrush` và nâng cấp lên `CHECKPOINT_SCHEMA_V2`. |

---

## 7. Kết Luận Đóng Băng Thiết Kế Nghiên Cứu (Research Freeze Status)

Toàn bộ các lỗ hổng toán học, rủi ro đạo hàm ngầm, sự nhập nhằng ký hiệu (`Eq. 3`, `Agreement vs. Complementarity`, `q = 0.001`), rủi ro chết kênh ở đầu chiếu, sự không nhất quán mẫu số tham số và lỗi trích dẫn trong các tài liệu trước đã được phát hiện và khắc phục triệt để.

**Trạng thái cuối cùng:** **`READY FOR USER APPROVAL (IMPLEMENTATION PROPOSAL READY)`** *(Đã đối soát toàn bộ điểm nghẽn kỹ thuật tại `post_audit_verification.md` và đồng bộ hóa thông số cài đặt; dừng chờ người dùng phê duyệt trước khi viết code).*
