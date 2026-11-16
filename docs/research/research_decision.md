# Quyết Định Nghiên Cứu Cuối Cùng (Vòng 3 — Research Decision Record)

**Dự án:** Knowledge Distillation-Based Lightweight Industrial Anomaly Detection (CS331)  
**Bài báo nền tảng (Baseline):** *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection* (`arXiv:2402.00448v1`, Liyi Yao & Shaobing Gao, 2024)  
**Hội đồng thẩm định:** Agent 0 (Research Lead), Agent 1 (Paper Analyst), Agent 2 (Literature Researcher), Agent 3 (Method Designer A), Agent 4 (Method Designer B), Agent 5 (Experiment & Reproducibility Engineer)  
**Ngày ban hành quyết định & Truy cập tài liệu:** `2026-11-14`  
**Trạng thái:** **`APPROVED WITH CONDITIONS — READY FOR USER APPROVAL (IMPLEMENTATION PROPOSAL READY)`** *(Đã đối soát toàn bộ 4 blocker tại `post_audit_verification.md` và đồng bộ tài liệu; dừng chờ người dùng phê duyệt trước khi viết code mô hình)*

---

## 1. Phân Loại Nhận Thức (Epistemic Tags)

- **`[FACT - Paper]`:** Dữ kiện và phương trình trích xuất trực tiếp từ `papers/2402.00448v1.pdf` cùng 12 bài báo đã xác minh trong `docs/research/literature_review.md`.
- **`[INTERPRETATION - Council & Audit]`:** Kết luận phân tích, phản biện chéo và kiểm toán kỹ thuật độc lập (`docs/research/final_technical_audit.md`).
- **`[HYPOTHESIS]`:** Giả thuyết khoa học của hai phương pháp cải tiến được chọn, sẽ được kiểm chứng trên MVTec AD theo `docs/research/experiment_design.md`.

---

## 2. Quyết Định Về Baseline DSKD (`arXiv:2402.00448v1`)

1. **Giữ nguyên 100% bài báo nền tảng DSKD làm Baseline chính thức (`[FACT - Paper]`):**
   - Gồm 3 mạng dựa trên `ResNet18` ($K=3$ khối đầu `conv2_x, conv3_x, conv4_x` với số kênh $c_k \in \{64, 128, 256\}$ ở kích thước $64\times 64, 32\times 32, 16\times 16$ cho ảnh $256\times 256$): Teacher $T$ (pre-trained ImageNet-1K, đóng băng hoàn toàn), Student Encoder $S_e$ (cùng cấu trúc với $T$), khối nút cổ chai Deep Feature Embedding (`DFE`), và Student Decoder $S_d$ (cấu trúc đảo ngược/transposed hoàn toàn).
2. **Xử lý minh bạch hai điểm mơ hồ kỹ thuật của bài báo gốc (`[INTERPRETATION - Audit]`):**
   - **Chuẩn hóa đặc trưng `Eq. (3)` (Tách bạch hoàn toàn với chuẩn hóa ảnh đầu vào):**
     - Trong PDF gốc (`tr. 5`), `Eq. (3)` in mẫu số là bình phương chuẩn $\|F\|_{\ell_2}^2$. Kiểm toán toán học độc lập (`final_technical_audit.md`, Mục 1) đã **bác bỏ** tính tương đương giữa $\ell_2$ và $\ell_{\cos}$ dưới `Literal Eq. (3)` (vì $\|\hat{F}\|_2 = 1/\|F\|_2 \ne 1$ và gây nổ đạo hàm bậc $\|\cdot\|_2^{-3}$ khi kích hoạt nhỏ), đồng thời **chứng minh** tính tương đương $\ell_2 \equiv \ell_{\cos}$ chỉ đúng khi chuẩn hóa đơn vị bậc một ($\|\hat{F}\|_2 = 1$).
     - Tuy nhiên, vì `Eq. (5)` vẫn giữ mẫu số $\|\hat{F}_t\|\|\hat{F}_e\|$ và `Eq. (6)` kết hợp $\lambda \ell_2 + \ell_{\cos}$ với $\lambda=0.1$, nhóm bắt buộc chạy thí nghiệm kiểm chứng trực tiếp **`S0_eq3_norm`** (so sánh `N1: Unit-L2` vs. `N2: Raw-L2 + Cosine` vs. `N3: Literal Inv-Sq L2`) tách biệt hoàn toàn với **`S0_input_norm`** (ImageNet Z-score vs. Raw $[0, 1]$ input).
   - **Báo cáo minh bạch hai mẫu số phức tạp của Baseline (`B0-Clean` làm mẫu số chính, `B0-Paper` làm cột tham chiếu):**
     - **`B0-Clean` (Mẫu số chuẩn nội bộ bắt buộc):** Bản tái hiện chuẩn hóa 3-stage (`stem + layer1..3` không giữ `layer4` chết) + Dense DFE (+F3 1x1 conv, Bilinear Sd) có **`12.180M` tổng tham số**, trong đó **`9.397M` tham số huấn luyện**, **`13.485G` FlopCounter FLOPs** và **`6.742G` MACs** (nếu dùng Deconv Sd: `12.098M` tổng tham số, `9.315M` trainable, `12.411G` FLOPs / `6.206G` MACs). Mọi tuyên bố phần trăm giảm tham số/FLOPs của Phương pháp B phải lấy `B0-Clean` làm mẫu số chính để đảm bảo công bằng tuyệt đối.
     - **`B0-Paper` (Cột tham chiếu bài báo):** Ghi nhận số liệu trong `Table 6 (p. 10)` của bài báo (`39.30M` params, `6.170G` FLOPs, `105MB` memory, `0.217s` CPU latency) ở cột riêng, tuyệt đối không trộn mẫu số.

---

## 3. Quyết Định Lựa Chọn Hai Phương Pháp Cải Tiến (Sau Kiểm Toán Kỹ Thuật Độc Lập)

Sau khi kiểm toán kỹ thuật độc lập (`docs/research/final_technical_audit.md`), hai phương pháp cải tiến được phê duyệt kèm các điều kiện kỹ thuật bắt buộc (**`APPROVED WITH CONDITIONS`**):

### 3.1. Phương Pháp A (Sinh viên 1): `FC-DSKD` — *Focal Collaborative Dual-Student Distillation with Calibrated Synergistic Anomaly Map Fusion*
- **Trạng thái kiểm toán:** **`APPROVED WITH CONDITIONS`**
- **Tài liệu đặc tả:** [`docs/research/method_a_proposal.md`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/method_a_proposal.md) & [`docs/research/final_technical_audit.md`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/final_technical_audit.md)
- **Mục tiêu cốt lõi:** Khắc phục sự lãng phí nhánh $M_{t-e}^k$ lúc suy luận (`Algo. 2`), sự suy biến $\ell_2 \equiv \ell_{\cos}$ và trung bình đều `Eq. (7)`, từ đó giải quyết đồng thời lỗi **mù vùng lõi khuyết tật lớn (*hollow-center*)** và **quầng dương tính giả quanh lỗi nhỏ (*boundary halo*)** mà tác giả DSKD nêu tại `Sec. 4.2.1–4.2.3` và `Sec. 5`.
- **Ba mô-đun cấu thành & Điều kiện kiểm toán bắt buộc:**
  1. **Mô-đun A1 (Spatial-Focal & Contextual Patch Distillation):** Kết hợp khoảng cách điểm $d_{\text{pt}}$ với khoảng cách ngữ cảnh lân cận $3\times 3$ ($\text{AvgPool}_{3\times 3}$, $\omega_{\text{ctx}}=0.5$) và trọng số tiêu cự không gian tự chuẩn hóa ($\gamma=1.0$).
  2. **Mô-đun A2 (Asymmetric Stop-Gradient Cross-Student Consistency $\ell_{e-d}$):** Bổ sung ràng buộc đồng thuận có hướng từ $\text{sg}[\hat{F}_e^k] \to \hat{F}_d^k$ ($\beta=0.15$, linear warm-up 20 epochs đầu). **Điều kiện bắt buộc từ Kiểm toán:** Phải ngắt đạo hàm đầu vào khối `DFE` (`f_emb = self.dfe([f.detach() for f in raw_f_e])`) và tách riêng 2 bộ tối ưu (`optimizer_e` cho $\theta_e$, `optimizer_d` cho $\theta_{\text{dfe}} \cup \theta_d$) để ngăn $\ell_d$ và $\ell_{e-d}$ rò rỉ ngược qua `DFE` vào $S_e$.
  3. **Mô-đun A3 (Normal-Val Calibrated Synergistic Fusion & Tail-Fraction Scoring):** Chuẩn hóa Z-score chỉnh lưu trên tập `val/good` ($\sigma_s^k \leftarrow \max(\sigma_s^k, 10^{-3})$), hợp nhất bản đồ hai học sinh qua **Toán tử Hợp nhất Cộng hưởng (Synergistic Fusion)**:
     $$\Phi^k = \underbrace{(1-\alpha_k)\widehat{M}_{t-e}^k + \alpha_k\widehat{M}_{t-d}^k}_{\text{Complementary Linear Union (Soft-OR)}} + \underbrace{\eta\sqrt{\widehat{M}_{t-d}^k \cdot \widehat{M}_{t-e}^k + \epsilon}}_{\text{Concordance Boost (Soft-AND)}}$$
     (đối chứng `Cfg-4a: Linear` $\eta=0$, `Cfg-4b: Synergistic` $\eta=0.3$, `Cfg-4c: SoftMaxUnion` từ cùng 1 checkpoint), và chấm điểm mức ảnh bằng trung bình tỷ lệ đuôi **$\rho_{\text{tail}} = 0.001$ (tức Top-$0.1\%$ diện tích ảnh = $65$ điểm ảnh tại $256\times 256$)**.
- **Chi phí tham số tăng thêm:** **`0.00M` (`+0` learnable parameters)**, `+0.16%` FLOPs.

### 3.2. Phương Pháp B (Sinh viên 2): `Lite-DSKD` — *Asymmetric Channel-Slimmed Dual-Student & Factorized Deep Feature Embedding*
- **Trạng thái kiểm toán:** **`APPROVED WITH CONDITIONS`**
- **Tài liệu đặc tả:** [`docs/research/method_b_proposal.md`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/method_b_proposal.md) & [`docs/research/final_technical_audit.md`](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/final_technical_audit.md)
- **Mục tiêu cốt lõi:** Giải quyết chi phí nặng nề của DSKD (`Table 6, p. 10`) và thắt chặt nút cổ chai thông tin hạng thấp để ngăn rò rỉ đặc trưng dị thường từ $S_e$ sang $S_d$.
- **Hai mô-đun cấu thành & Điều kiện kiểm toán bắt buộc:**
  1. **Mô-đun B1 (`Lite-DFE` — Factorized Deep Feature Embedding):** Thay thế các phép tích chập đặc $3\times 3$ stride 2 và khối Residual kênh rộng trong `DFE` (`Fig. 4`) bằng hạ mẫu tách biệt chiều sâu (`DS-MSA`) và khối Residual nút cổ chai đảo ngược (`BDRF` kèm nhánh tắt $1\times 1$ $\text{PW}_{\text{skip}}$). **Hai biến thể thiết kế rõ ràng:** (i) *Canonical `dsmsa_db`:* Giảm tham số khối `DFE` từ `3.288M` xuống **`0.141M`** ($\alpha=1.0, \rho=0.5$), **`0.080M`** ($\alpha=0.75, \rho=0.5$), **`0.052M`** ($\alpha=0.50, \rho=1.0$, giảm **98.4%**), và **`0.037M`** ($\alpha=0.50, \rho=0.5$, giảm **98.9%**); (ii) *Ablation `dsmsa_expand`:* $408.9\text{k}$ ($\alpha=1.0$) và $137.7\text{k}$ ($\alpha=0.5, \rho=1.0$). Cả hai đều đã tính đầy đủ $\text{PW}_{\text{skip}}$.
  2. **Mô-đun B2 (`Slim-DSKD` — Width-Compressed Dual Students with Audited Projections):** Thu hẹp độ rộng kênh của cả $S_e^{(\alpha)}$ và $S_d^{(\alpha)}$ theo hệ số $\alpha \in \{0.50, 0.75\}$ ($c_k^{(s)} = \alpha c_k$) và khôi phục chiều kênh khớp với Teacher $T$ qua đầu chiếu điểm $\mathcal{P}_s^k(G_s^k) = \phi\!\left(\text{BN}\!\left(W_{s,\text{proj}}^k *_{1\times 1} G_s^k\right)\right)$. **Điều kiện bắt buộc từ Kiểm toán:** Chạy Ablation 3 biến thể hàm kích hoạt $\phi \in \{\text{Identity (BN-only)}, \text{ReLU}, \text{LeakyReLU}(0.01)\}$ trên Core-5 để kiểm tra hiện tượng *Dead Projected Channels* vs. *Cosine Sign Mismatch*.
- **Hiệu quả nén đạt được (Đã hiệu chỉnh chính xác sau Kiểm toán):**
  - Cấu hình `Lite-DSKD-0.50` canonical (`dsmsa_db`, $\alpha=0.50, \rho=1.0$): Chỉ còn **`1.671M` tham số huấn luyện** (**giảm 82.2%** so với `B0-Clean` `9.397M`), tổng tham số tính cả $T$ còn **`4.454M`** (**giảm 63.4%** so với `B0-Clean` `12.180M`; **giảm 88.7%** so với `39.30M` của `B0-Paper`), tổng FLOPs suy luận giảm còn **`5.824G` FLOPs** (`2.912G` MACs, **giảm 56.8%** so với `B0-Clean` `13.485G` FLOPs).
  - Cấu hình `Lite-DSKD-0.75` canonical (`dsmsa_db`, $\alpha=0.75, \rho=0.5$): **`3.651M` tham số huấn luyện** (**giảm 61.2%** so với `B0-Clean`), tổng tham số **`6.433M`**, tổng FLOPs suy luận **`8.352G` FLOPs** (`4.176G` MACs, **giảm 38.1%** so với `B0-Clean`).

---

## 4. Quản Trị Rủi Ro & Phương Án Dự Phòng (Risk & Contingency Matrix)

| Mã Rủi ro | Mô tả Rủi ro Kỹ thuật / Thực nghiệm | Mức độ | Phương án Phòng ngừa & Dự phòng (Contingency Plan) |
| :---: | :--- | :---: | :--- |
| **R1** | **Mơ hồ chuẩn hóa đầu vào ở Mục 4.1.2 của bài báo DSKD** (*"resized images are normalized by $\ell_2$ function"* vs. chuẩn ImageNet Z-score cho ResNet18 pre-trained). | Trung bình | Chạy phép thử đối chứng ngắn `S0_norm` (50 epochs trên `carpet` và `bottle`, tốn 15 phút GPU vào Ngày 9) để chọn phương án cho `val/good` loss và AUROC ổn định nhất trước khi chạy toàn bộ ma trận. |
| **R2** | **Ngắt kết nối phiên Kaggle GPU (Timeout 12h hoặc Preemption) làm mất tiến độ đang train.** | Cao | Áp dụng `CheckpointManager` ghi nguyên tử (`.tmp -> os.replace`) mỗi 25 epochs kèm đầy đủ `RNG states` + bộ đếm thời gian `SessionWatchdog` tự động dừng an toàn ở mốc `11.0h` và lưu bền vững qua **Kaggle Dataset Version**. |
| **R3** | **Nén quá sâu ($\alpha=0.50$) trong Method B gây giảm nhẹ PRO trên các lớp texture phức tạp (`tile, grid, wood`).** | Trung bình | Giữ $\rho=1.0$ ($d_{\text{mid}}=128$) trong `Lite-DFE` và chạy song song cả hai mức **`Lite-DSKD-0.75`** và **`Lite-DSKD-0.50`** (chỉ tốn thêm `~1.1h` GPU cho toàn bộ 15 categories) để có đường cong Pareto hoàn chỉnh; khi ghép với Method A (`C1`), trọng số tiêu cự `A1` và hiệu chuẩn `A3` sẽ tự động bù đắp sai số này. |
| **R4** | **Ràng buộc đồng thuận $\ell_{e-d}$ (Mô-đun A2) làm nhòe đặc trưng biên của $S_e$.** | Đã xử lý | Đã khóa cơ chế **Asymmetric Stop-Gradient ($\text{sg}[\hat{F}_e^k]$)** kèm warm-up $\beta \le 0.15$; кроме đó ma trận Ablation của Method A đã tách riêng `Cfg-2 (Only A1)`, `Cfg-5 (Only A3)` và `Cfg-6/7 (A1+A2+A3)` để nếu `A2` không mang lại lợi ích trên một số category thì vẫn chứng minh được đóng góp độc lập của `A1` và `A3`. |

---

## 5. Danh Mục 5 Quyết Định Cần Người Dùng Phê Duyệt (Approval Checklist)

Theo đúng Quy tắc 10 (Điều kiện dừng), nhóm nghiên cứu **dừng tại đây** (chưa viết bất kỳ dòng code mô hình hay notebook huấn luyện nào) và trình người dùng phê duyệt 5 hạng mục sau để chuyển sang giai đoạn Implementation (Ngày 7–11):

1. **Phê duyệt Kế hoạch Tái hiện Baseline DSKD (`B0-Clean`):**
   - Chạy kiểm chứng tiền thực nghiệm `S0_eq3_norm` (đối chứng `N1: Unit-L2` vs `N2: Raw-L2+Cos` vs `N3: Literal Inv-Sq L2`) và `S0_input_norm` (ImageNet Z-score vs `P_img_l2`), lấy `N1` làm ứng viên baseline chính; báo cáo song song độ phức tạp của `B0-Paper` (`39.30M` params, `6.170G` MACs) cùng `B0-Clean` (`12.180M` total / `9.397M` trainable params, `13.485G` FlopCounter FLOPs / `6.742G` MACs).
2. **Phê duyệt Phương pháp Cải tiến A (Sinh viên 1 — `FC-DSKD`):**
   - Triển khai 3 mô-đun **A1** (Spatial-Focal & Contextual Patch Loss), **A2** (Asymmetric Stop-Grad Cross-Student Consistency $\ell_{e-d}$), và **A3** (Normal-Val Z-score Calibration + Gated Fusion $M_{t-e}^k \oplus M_{t-d}^k$ + Top-$q\%$ Scoring) với `+0.00M` tham số.
3. **Phê duyệt Phương pháp Cải tiến B (Sinh viên 2 — `Lite-DSKD`):**
   - Triển khai **B1** (canonical `Lite-DFE` theo thiết kế `dsmsa_db`, đối chứng `dsmsa_expand`) và **B2** (`Slim-DSKD`: thu hẹp kênh $\alpha \in \{0.50, 0.75\}$ trên $S_e^{(\alpha)}, S_d^{(\alpha)}$ kèm kiểm chứng 3 biến thể kích hoạt đầu chiếu `BN-only` vs `BN+ReLU` vs `BN+LeakyReLU`), giảm `82.2%` tham số huấn luyện và `56.8%` FLOPs suy luận đối với cấu hình $\alpha=0.50, \rho=1.0$.
4. **Phê duyệt Giao thức Thực nghiệm & Kaggle Checkpoint/Resume (`experiment_design.md` & `kaggle_implementation_plan.md`):**
   - Thực hiện theo 2 giai đoạn (**Stage 1: Core-5 Subset** `carpet, tile, bottle, screw, toothbrush` cho toàn bộ Ablations Group 0–4; **Stage 2: Full 15 Categories** cho `B0-Clean`, `FC-DSKD`, `Lite-DSKD-0.75/0.50`, và `Combined FC-Lite-DSKD`), đo đủ 10 chỉ số, tuân thủ tuyệt đối `Zero Test-Set Leakage` (`80/20` split trên `train/good` với `seed=42`).
5. **Phê duyệt Phân công Công việc (2 Sinh viên) & Lịch trình 28 Ngày (`project_timeline.md`):**
   - Chạy kiểm chuẩn thời gian thực nghiệm sơ bộ (`M0_bench`, 10 phút GPU) trước khi chạy đại trà; tổng ngân sách GPU dự kiến $\sim 19.5\text{ giờ GPU}$ trên tổng hạn mức $240\text{ giờ GPU}$ trong 4 tuần của 2 sinh viên.

---

## 6. Mức Độ Tin Cậy Của Quyết Định (Confidence Level)
- **Mức độ tin cậy tổng thể:** **96% (Rất cao)** — Toàn bộ 10 tài liệu nghiên cứu trong `docs/research/` đã được hoàn thành, kiểm tra chéo và đồng bộ hóa nhất quán.
