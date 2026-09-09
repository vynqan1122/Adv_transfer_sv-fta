# SV-FCA: Source-View Frequency-Coordinated Attack

Code nghiên cứu tấn công đối kháng chuyển giao trên ImageNet-1k, gồm **SV-FCA**, bảy baseline, các thí nghiệm Tables I–VIII và công cụ vẽ hình. Tên repository còn chứa `sv-fta` để tương thích; thuật toán hiện tại là **SV-FCA**.

README này mô tả phép tính được triển khai trong code. Các bảng CSV chỉ được tạo sau khi chạy thí nghiệm; repository không kèm kết quả ASR đã xác nhận, dữ liệu ImageNet hay trọng số model.

## 1. Phạm vi và quy trình

```text
ImageNet + nhãn chuẩn
    -> chọn ảnh mà tất cả surrogate được chỉ định dự đoán đúng
    -> tạo adversarial images chỉ bằng gradient của surrogate
    -> lưu ảnh hoặc perturbation + manifest + cấu hình
    -> đánh giá độc lập trên target
    -> tổng hợp ASR / chất lượng ảnh / thời gian / hình minh họa
```

- Tấn công **untargeted**, ràng buộc chuẩn vô cùng trên pixel trong khoảng `[0, 1]`.
- Surrogate cho phép lấy gradient; target chỉ dùng lúc đánh giá. Việc chọn tập con mặc định dùng bốn surrogate, không dùng target.
- Tables I–IV dùng **một surrogate cho mỗi lần tạo ảnh**. Tables V–VIII có thể dùng nhiều surrogate; không so trực tiếp hai cấu hình như thể cùng ngân sách gradient.
- Không có nhánh token, truy vấn target hoặc tối ưu theo confidence của target trong SV-FCA.
- Baseline `vit_aware` là cách chọn patch bằng input gradient của dự án; không phải triển khai một thuật toán attention attack chuẩn đã được xác nhận theo paper.

## 2. Cài đặt

Khuyến nghị Linux hoặc WSL2, Python 3.10 trở lên. Các file `.sh` cần **Bash**, không chạy trực tiếp bằng PowerShell hoặc `sh`.

```bash
git clone https://github.com/vynqan1122/Adv_transfer_sv-fta.git
cd Adv_transfer_sv-fta
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Cài cặp PyTorch/torchvision phù hợp với driver từ [PyTorch](https://pytorch.org/get-started/locally/). Ví dụ môi trường CPU dùng cho kiểm thử:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Để tính LPIPS và chạy target RobustBench:

```bash
python -m pip install -r requirements-optional.txt
```

LPIPS dùng AlexNet và có thể tải trọng số ở lần chạy đầu. RobustBench cần checkpoint tương thích với phiên bản thư viện. Ghi lại môi trường của thí nghiệm bằng `python -m pip freeze > runs/environment.txt` sau khi tạo thư mục `runs`.

Mặc định batch size là **16** cho chọn ảnh, tạo attack, đánh giá và đo chất lượng/runtime. Khi CUDA hết bộ nhớ trong lúc xử lý ảnh, chương trình tự giảm batch thực thi và thử lại phần chưa hoàn thành; chi tiết ở mục 7.2. Nếu riêng trọng số model đã vượt VRAM, giảm batch không giải quyết được: cần giảm số surrogate hoặc dùng GPU nhiều bộ nhớ hơn. CPU dùng được cho kiểm thử nhỏ; không cần chạy toàn bộ thí nghiệm để kiểm tra cấu hình.

## 3. Dữ liệu và model

### 3.1. Nhãn ImageNet

CSV cần ít nhất hai cột:

```csv
relpath,label
n01440764/ILSVRC2012_val_00000293.JPEG,0
```

`relpath` trỏ đến ảnh theo gốc dữ liệu đã cấu hình; `label` là class index ImageNet-1k **0–999** theo checkpoint. Không suy nhãn bằng cách đánh số lại một tập con các thư mục class. Loader hỗ trợ cả dấu `/` và `\`, CSV UTF-8 có hoặc không có BOM, và hai cách đặt gốc `/data/imagenet` hoặc `/data/imagenet/val` với tiền tố `val/` tương ứng. Với thư mục validation ở vị trí khác, CSV phải trỏ đúng ảnh từ `DATA_DIR`; `VAL_DIR` chỉ dùng khi chọn ảnh bằng ImageFolder.

Với validation set đã chia đủ 1.000 thư mục WordNet ID:

```bash
python scripts/make_imagenet_val_labels_csv.py \
  --val-dir /data/imagenet/val \
  --out-csv /data/imagenet/imagenet_val_labels.csv
```

Với ảnh validation đặt phẳng, cần chuẩn bị CSV từ ground truth chính thức. Script trên không suy được nhãn từ tên ảnh phẳng. Có thể dùng ImageFolder khi bỏ CSV và cấu trúc thư mục đã có đủ mapping ImageNet.

### 3.2. Tiền xử lý

Các ảnh đều được `Resize(256)`, `CenterCrop(224)`, chuyển thành RGB tensor `[0, 1]`. Perturbation được tối ưu trong không gian crop **224 × 224** này, không phải ảnh JPEG gốc. Trong CLI `run_attack.py`, `--image-size` mặc định là 224; có thể bỏ cờ này, nhưng nếu truyền thì chỉ chấp nhận 224 để khớp tiền xử lý khi tạo ảnh, đánh giá và vẽ hình.

Wrapper timm lấy `mean`, `std` và kích thước đầu vào từ cấu hình checkpoint. Khi cần, tensor được resize khả vi trước khi chuẩn hóa, ví dụ crop 224 được resize cho Inception-v3 dùng đầu vào 299. Đây là quy trình chung của thí nghiệm transfer, không đồng nhất với mọi quy trình đánh giá nguyên bản của từng model. RobustBench sử dụng preprocessing của model do RobustBench trả về, tránh chuẩn hóa hai lần.

### 3.3. Nhóm model mặc định

| Vai trò | Model |
| --- | --- |
| CNN surrogate | `resnet50`, `densenet121` |
| ViT surrogate | `vit_base_patch16_224`, `deit_small_patch16_224` |
| CNN target | `resnet152`, `inception_v3` |
| ViT target | `vit_large_patch16_224`, `deit_base_patch16_224`, `swin_tiny_patch4_window7_224` |
| Robust target | `Salman2020Do_R50`, `Mo2022When_ViT-B`, `Liu2023Comprehensive_Swin-B` |

Tải model timm bằng `bash sh/download_models.sh`. Cache mặc định nằm trong `pretrained_models`; các biến cache đã được đặt từ bên ngoài có thể được ưu tiên. Tên model timm không có hậu tố checkpoint có thể đổi trọng số mặc định giữa các phiên bản; cần lưu phiên bản và checkpoint để tái lập.

Table VIII yêu cầu các checkpoint cục bộ:

```text
models/
└── imagenet/
    └── Linf/
        ├── Salman2020Do_R50.pt
        ├── Mo2022When_ViT-B.pt
        └── Liu2023Comprehensive_Swin-B.pt
```

```bash
export ROBUSTBENCH_MODEL_DIR="$PWD/models"
python scripts/check_defense_models.py --model-dir "$ROBUSTBENCH_MODEL_DIR"
```

Biến trên phải trỏ đến thư mục **chứa** `imagenet`, không trỏ thẳng vào `imagenet/Linf`. Preflight kiểm tra file tồn tại; chỉ khi load và suy luận thực tế mới xác nhận được checkpoint dùng được. Có thể truyền `--targets` cho `check_defense_models.py` để kiểm tra đúng danh sách target tùy chỉnh; launcher Table VIII tự chuyển tiếp `DEFENSE_TARGETS`. Ngân sách tấn công của dự án là 16/255 mặc định, có thể khác ngân sách leaderboard của RobustBench.

## 4. Ký hiệu và phép cập nhật chung

Gọi ảnh sạch là $x$, nhãn đúng là $y$, ảnh ở bước $t$ là $x_t$, có $K$ surrogate $f_k$, số bước là $T$. Hàm mất mát là cross-entropy $\ell(f_k(x),y)$; tấn công untargeted **tăng** loss này.

Các công thức bên dưới mô tả một ảnh; chiều batch được lược bỏ. Gradient và các tensor band có kích thước $C\times H\times W$; các phép chuẩn hóa và norm đều tính trên cả channel lẫn hai chiều không gian của từng ảnh. Mỗi $f_k$ bao gồm wrapper resize/normalization của model, nên đạo hàm đi qua cả tiền xử lý đó.

```math
x_0=x
```

```math
x_t\in[0,1]^{C\times H\times W}
```

```math
\lVert x_t-x\rVert_\infty\leq\varepsilon.
```

Mỗi thuật toán tạo hướng $d_t$ rồi cập nhật:

```math
x_{t+1}=\mathrm{clip}_{[0,1]}\left(
x+\mathrm{clip}_{[-\varepsilon,\varepsilon]}
\left(x_t+\alpha\mathrm{sign}(d_t)-x\right)
\right).
```

Mặc định $\varepsilon=16/255$, $\alpha=1.6/255$, $T=10$, không random start. Chỉnh `EPS`, `ALPHA`, `STEPS` trong `experiment.venv` để áp dụng cho các launcher. File cấu hình nhận cả số thập phân lẫn phân số như `16/255`, `1.6/255`; `common.sh` đổi sang số thực trước khi gọi Python. Khi gọi trực tiếp CLI, `--eps` và `--alpha` vẫn nhận số thập phân theo thang pixel `[0, 1]`.

Hàm `normalize_grad_l1` thực tế chia cho **trung bình trị tuyệt đối** trên mỗi ảnh. Với $D=CHW$ và $\eta=10^{-12}$:

```math
\mathcal N(g)=\frac{g}{\max\left(D^{-1}\sum_{j=1}^{D}|g_j|,\eta\right)}.
```

Điểm này cần giữ nguyên khi tái lập: chuẩn hóa theo tổng trị tuyệt đối khác hệ số $D$ và có thể làm thay đổi vị trí lookahead trong SI-NI-FGSM. Gradient được lấy từ loss trung bình batch; cùng batch size thì hệ số này chung cho các model.

## 5. Các baseline trong code

### I-FGSM — `attacks/ifgsm.py`

Lấy trung bình gradient loss của các surrogate rồi dùng dấu của gradient:

```math
g_t=\frac{1}{K}\sum_{k=1}^{K}\nabla_{x_t}\ell(f_k(x_t),y)
```

```math
d_t=g_t.
```

Khi có nhiều model, đây là trung bình **gradient của từng loss**, không phải cross-entropy của logits đã lấy trung bình.

### MI-FGSM — `attacks/mifgsm.py`

```math
m_0=0
```

```math
m_{t+1}=\mu m_t+\mathcal N(g_t)
```

```math
d_t=m_{t+1}.
```

Hệ số momentum mặc định $\mu=1$.

### DI-FGSM — `attacks/difgsm.py`

Thay đầu vào mỗi surrogate bằng biến đổi khả vi $\mathcal T$: resize ngẫu nhiên từ 224 đến 256, zero-padding đến 256, rồi resize về 224. Với xác suất $1-p$, dùng identity.

```math
d_t=\frac{1}{K}\sum_{k=1}^{K}
\nabla_{x_t}\ell(f_k(\mathcal T_{k,t}(x_t)),y).
```

Biến đổi được lấy mẫu theo model và theo bước, dùng chung trong batch. CLI và pipeline mặc định truyền $p=1$; class dùng trực tiếp mặc định $p=0.7$. Phiên bản DI này không cộng momentum.

### TI-FGSM — `attacks/tifgsm.py`

```math
d_t=W* g_t
```

```math
W_{u,v}=\frac{\exp(-(u^2+v^2)/(2\sigma^2))}
{\sum_{a,b}\exp(-(a^2+b^2)/(2\sigma^2))}.
```

Code dùng convolution riêng từng channel, Gaussian kernel 15 × 15, $\sigma=3$, zero-padding. Phiên bản TI này không ghép DI hoặc momentum.

### SI-NI-FGSM — `attacks/sinifgsm.py`

Khởi tạo $m_0=0$. Lookahead theo momentum, lấy trung bình loss trên $S=5$ mức cường độ ảnh, rồi cộng momentum:

```math
z_t=x_t+\mu\alpha m_t
```

```math
g_t=\frac{1}{KS}\sum_{k=1}^{K}\sum_{i=0}^{S-1}
\nabla_{z_t}\ell(f_k(z_t/2^i),y)
```

```math
m_{t+1}=\mu m_t+\mathcal N(g_t).
```

Dùng $d_t=m_{t+1}$. Scale ở đây là **cường độ pixel**, không phải kích thước hình học. Đạo hàm đi qua phép chia nên có hệ số chain rule tương ứng. Lookahead không bị clip; ảnh sau cập nhật vẫn được chiếu về miền hợp lệ.

Các baseline là biến thể được định nghĩa bởi những công thức trên. Đặc biệt, SI-NI dùng chuẩn hóa theo trung bình trị tuyệt đối, DI có bước resize trở lại ảnh gốc, còn DI/TI không ghép momentum; không mặc nhiên coi chúng là bản tái lập nguyên trạng mọi thiết lập trong paper gốc.

### Freq-Only — `attacks/freq_only.py`, `src/frequency.py`

```math
d_t=\mathrm{Re}\left(\mathcal F^{-1}
\left(M\odot\mathcal F(g_t)\right)\right).
```

FFT được thực hiện riêng trên mỗi channel. Với bán kính tần số chuẩn hóa $r=\sqrt{f_x^2+f_y^2}/\sqrt{0.5^2+0.5^2}$, chế độ `low_mid` dùng trọng số 1 khi $r\leq0.25$, 0.75 khi $0.25<r\leq0.55$, và $0.10+0.25(a+1)/2$ ở vùng cao; $a$ là cosine agreement trung bình giữa gradient của surrogate, giới hạn trong `[-1, 1]`.

$g_t$ ở đây là gradient thô được lấy trung bình như I-FGSM. Agreement lấy trung bình trên các cặp surrogate và các ảnh trong batch; khi chỉ có một surrogate, code quy ước $a=1$. Các tọa độ tần số lấy từ `fftfreq`, sau đó `fftshift` cùng phổ, để mask giữ đối xứng liên hợp cho ảnh thực.

Các mode khác: `low` giữ $r\leq0.25$; `mid` giữ $0.20<r\leq0.55$; `high` giữ $r>0.55$; `all` giữ toàn bộ. Đây là FFT, không phải DCT và không phải cơ chế phối hợp dải tần của SV-FCA.

### ViT-Aware — `attacks/vit_aware.py`

Lấy trung bình gradient của các ViT surrogate; nếu không có ViT thì dùng tất cả surrogate. Tính saliency bằng trung bình trị tuyệt đối qua channel, average-pool thành lưới patch, giữ top 25% patch (làm tròn xuống, ít nhất một patch), rồi phóng mask bằng nearest-neighbor:

```math
s=\mathrm{Pool}_{\mathrm{patch}}
\left(\mathrm{Mean}_{c}|g^{\mathrm{selected}}_t|\right)
```

```math
d_t=\mathrm{Upsample}(\mathrm{TopKMask}(s))\odot g_t.
```

Patch mặc định 16 pixel. Với crop 224, lưới có 14 × 14 patch và giữ đúng 49 patch; các patch đồng hạng tại ngưỡng được chọn theo `torch.topk`. Hướng cuối dùng mask nhân với gradient thô $g_t$ trung bình trên **tất cả** surrogate, như I-FGSM. Mask dựa trên input gradient, không đọc attention/token gradient nội bộ.

## 6. SV-FCA chi tiết — `attacks/sv_fca.py`

### 6.1. Source-view gradient pool

Mỗi surrogate có $R$ view; view đầu là identity, các view sau dùng phép DI ở trên. Với $P=KR$:

```math
g_{k,r,t}=\nabla_{x_t}\ell(f_k(\mathcal T_{k,r,t}(x_t)),y)
```

```math
\widehat g_{k,r,t}=\mathcal N(g_{k,r,t}).
```

View ngẫu nhiên được lấy mẫu trong vòng lặp từng surrogate. Chuẩn hóa **từng** source-view gradient trước khi lấy trung bình, khác baseline lấy trung bình gradient thô.

### 6.2. Phân rã dải tần

Chia bán kính chuẩn hóa thành $B$ khoảng đều nhau. Các mask $M_b$ phủ toàn bộ lưới rFFT, không chồng lấn; khoảng cuối bao gồm biên trên.

Với $r=\sqrt{f_x^2+f_y^2}/\sqrt{0.5^2+0.5^2}$, band $b$ giữ $b/B\leq r<(b+1)/B$; band cuối giữ cả $r=1$. Trục dọc lấy từ `fftfreq(H)`, trục ngang lấy từ `rfftfreq(W)`. Mask có kích thước $H\times(\lfloor W/2\rfloor+1)$ và dùng chung cho các channel/ảnh. Đánh số các phần tử pool bằng $i=1,\ldots,P$:

```math
h_{i,b,t}=\mathrm{irFFT2}\left(
M_b\odot\mathrm{rFFT2}(\widehat g_{i,t})\right)
```

```math
\overline h_{b,t}=\frac1P\sum_{i=1}^{P}h_{i,b,t}.
```

Cả hai phép biến đổi dùng `norm="ortho"`, inverse nhận lại đúng `(H, W)`, nên áp dụng được cho kích thước chẵn hoặc lẻ. Tổng các band khôi phục gradient đã chuẩn hóa trong sai số số học.

### 6.3. Đồng thuận giữa các gradient trong từng band

Với mỗi ảnh, mỗi band:

```math
u_{i,b,t}=\frac{h_{i,b,t}}{\max(\lVert h_{i,b,t}\rVert_2,\eta)}
```

```math
U_{b,t}=\sum_i u_{i,b,t}
```

```math
Q_{b,t}=\sum_i\lVert u_{i,b,t}\rVert_2^2.
```

```math
c_{b,t}=\frac{\lVert U_{b,t}\rVert_2^2-Q_{b,t}}{P(P-1)}\quad(P>1).
```

Đây là trung bình tích vô hướng của mọi cặp vector đã chuẩn hóa, tương đương cosine khi gradient khác 0 và lớn hơn ngưỡng số học. Band có gradient bằng 0 đóng góp 0 vào các cặp. Khi $P=1$, code quy ước consensus bằng 1; consensus được giới hạn trong `[-1, 1]` để xử lý sai số làm tròn. Công thức trừ $Q$, không luôn trừ $P$, để xử lý đúng band bằng 0.

Code chỉ tích lũy tổng band, tổng vector chuẩn hóa và tổng bình phương chuẩn. Không cần lưu toàn bộ $K\times R$ gradient đồng thời.

### 6.4. Low-mid prior và trọng số tức thời

Với $b=0,\ldots,B-1$, tâm band là $q_b=(b+0.5)/B$:

```math
v_b=0.15+\exp\left(-\frac12\left(\frac{q_b-0.30}{0.30}\right)^2\right)
```

```math
p_b=\left[\max\left(\frac{v_b}{\max_j v_j},0.05\right)\right]^\gamma.
```

```math
w_{b,t}=\frac{\exp(c_{b,t}/\tau+\log\max(p_b,\eta))}
{\sum_j\exp(c_{j,t}/\tau+\log\max(p_j,\eta))}.
```

$\tau$ là temperature, $\gamma$ là độ mạnh prior. Prior thiên về tần số thấp–trung bình; phép chặn dưới bằng $\eta$ trước khi lấy log tránh `log(0)` nếu lũy thừa bị underflow. Với $\gamma=0$, prior đồng đều. Code dùng `torch.softmax` để tính trọng số ổn định; trọng số được tính riêng cho từng ảnh.

### 6.5. Spectral EMA, momentum và projection

Ở bước đầu $\widetilde w_{b,0}=w_{b,0}$. Các bước tiếp theo:

```math
\widetilde w_{b,t}=\beta\widetilde w_{b,t-1}+(1-\beta)w_{b,t}
```

```math
\sum_b\widetilde w_{b,t}=1.
```

Code chuẩn hóa lại tổng trọng số để hạn chế sai số. Sau đó:

```math
v_t=\sum_b\widetilde w_{b,t}\overline h_{b,t}
```

```math
m_{t+1}=\mu m_t+\mathcal N(v_t)
```

```math
d_t=m_{t+1}.
```

Cập nhật ảnh bằng phép projection ở mục 4. Spectral memory và momentum được khởi tạo lại cho mỗi batch, không truyền từ ảnh này sang ảnh khác.

### 6.6. Tham số mặc định và chi phí

| Tham số | CLI | Biến Bash | Mặc định |
| --- | --- | --- | --- |
| Số view | `--num-views` | `SVFCA_NUM_VIEWS` | 4 |
| Số band | `--spectral-bands` | `SVFCA_SPECTRAL_BANDS` | 6 |
| Xác suất DI | `--diversity-prob` | `SVFCA_DIVERSITY_PROB` | 1.0 |
| Temperature | `--band-temperature` | `SVFCA_BAND_TEMPERATURE` | 0.35 |
| Prior strength | `--low-mid-strength` | `SVFCA_LOW_MID_STRENGTH` | 1.0 |
| EMA decay | `--spectral-decay` | `SVFCA_SPECTRAL_DECAY` | 0.75 |
| Momentum decay | `--decay` | `SVFCA_DECAY` | 1.0 |
| CUDA autocast | `--amp` | `SVFCA_AMP` | 0, tắt |
| Kiểu dữ liệu AMP | `--amp-dtype` | `SVFCA_AMP_DTYPE` | `fp16`; có thể chọn `bf16` |

SV-FCA cần $TKR$ lần lấy gradient. Mỗi source-view cần một rFFT và $B$ inverse rFFT. Bộ nhớ tích lũy band tăng theo $B$ và batch size; streaming giảm việc giữ toàn bộ pool nhưng các surrogate vẫn được load đồng thời. Khi bật AMP, forward của SV-FCA dùng fp16/bf16 trên CUDA; FFT và thống kê giữ float32. Không tự động áp AMP cho bảy baseline.

Số view $R$ và số band $B$ là số nguyên, với $R\geq1$ và $B\geq2$ khi dùng phân rã tần số. Các tham số thực phải hữu hạn và thỏa $0\leq p\leq1$, $\tau>0$, $\gamma\geq0$, $0\leq\beta<1$, $\mu\geq0$; $p$ là xác suất DI. Các giá trị temperature/EMA ngoài miền hợp lệ bị báo lỗi, không được tự sửa ngầm. AMP không bảo đảm tương đương số học với float32; gradient không hữu hạn sẽ làm SV-FCA báo lỗi để người chạy đổi precision hoặc kiểm tra model.

### 6.7. Ablation Table V

| Variant | Thay đổi so với full model |
| --- | --- |
| `full_model` | Toàn bộ SV-FCA |
| `without_sv_pool` | Đặt R = 1, chỉ identity; vẫn giữ tất cả K surrogate |
| `without_frequency_coordination` | Dùng trung bình gradient đã chuẩn hóa trong không gian ảnh; bỏ toàn bộ band, prior, consensus và spectral EMA |
| `without_band_consensus` | Bỏ hạng consensus khỏi logits trọng số; vẫn dùng prior |
| `without_low_mid_prior` | Bỏ log prior; chỉ dùng consensus |
| `without_spectral_memory` | Dùng trọng số tức thời, không EMA |

Với prior cố định, variant không consensus có trọng số không đổi theo thời gian; EMA khi đó không tạo thêm thay đổi. Đây là hệ quả của công thức, không phải một nguồn ngẫu nhiên khác.

## 7. Chạy thí nghiệm

### 7.1. Một file cấu hình: `experiment.venv`

Mở và sửa [experiment.venv](experiment.venv). Tất cả launcher tự đọc file này; không cần copy template hoặc `source` một file `.env` khác. Đây là **file cấu hình Bash** có đuôi `.venv`, khác với **thư mục môi trường Python** `.venv/` tạo ở bước cài đặt.

Các tham số được nhóm theo dữ liệu, thiết bị/bộ nhớ, ngân sách tấn công, source/target, SV-FCA, defense và hình. Trong mỗi dòng dạng `: "${BATCH_SIZE:=16}"`, sửa giá trị sau `:=`. Biến đã truyền từ terminal được ưu tiên hơn giá trị mặc định trong file.

```bash
# Xem cấu hình đã được giải quyết và kiểm tra tham số; không load model/dataset.
bash sh/common.sh

# Ghi đè tạm cho một lần chạy.
EPS=4/255 ALPHA=0.4/255 STEPS=10 bash sh/run_table5_ablation.sh

# Dùng thêm cấu hình riêng nếu cần; mặc định chỉ cần experiment.venv.
CONFIG_FILE=/path/to/server.venv bash sh/run_tables_1_8.sh
```

File do `CONFIG_FILE` chỉ định được đọc trước; các giá trị còn thiếu được lấy từ `experiment.venv`. Dùng cùng cú pháp mặc định `: "${NAME:=value}"` trong file riêng để giữ ưu tiên của biến terminal. Tên `*.local.venv` được Git bỏ qua nếu muốn lưu cấu hình máy cá nhân.

Đường dẫn tương đối được hiểu theo gốc repository, kể cả khi gọi launcher từ thư mục khác. `sh/common.sh` phụ trách đọc/kiểm tra cấu hình và các hàm chạy chung; giá trị dùng để chỉnh thí nghiệm nằm trong `experiment.venv`. Các lớp Python không tự đọc file Bash này: khi gọi Python trực tiếp, truyền các cờ CLI tương ứng.

| Nhóm | Tham số | Mặc định / ý nghĩa |
| --- | --- | --- |
| Dữ liệu | `DATA_DIR`, `LABELS_CSV`, `VAL_DIR` | Gốc ảnh, CSV nhãn; `LABELS_CSV=""` để dùng ImageFolder |
| Đầu ra | `MODEL_DIR`, `OUT_DIR`, `ADV_BATCH_ROOT` | Cache model, kết quả và batch tạm |
| Thiết bị | `PY`, `DEVICE`, `NUM_WORKERS`, `SEED` | `python3`, `auto`, 2, 0 |
| Batch | `BATCH_SIZE` | 16; kích thước batch logic cho chọn ảnh/attack/runtime |
| Batch đánh giá | `EVAL_BATCH_SIZE`, `QUALITY_BATCH_SIZE` | Kế thừa `BATCH_SIZE` nếu chưa ghi đè |
| OOM | `AUTO_BATCH`, `MIN_BATCH_SIZE` | 1 và 1; bật giảm batch, tối thiểu một ảnh |
| Ngân sách attack | `EPS`, `ALPHA`, `STEPS` | `16/255`, `1.6/255`, 10 |
| Ngân sách defense | `DEFENSE_EPS`, `DEFENSE_ALPHA`, `DEFENSE_STEPS` | Kế thừa ngân sách chung; chỉ ghi đè cho Table VIII |
| Số mẫu | `NUM_IMAGES`, `NUM_BATCHES`, `SELECTED_CSV` | 1.000 ảnh mặc định; CSV cố định cho thí nghiệm |
| Lưu tensor | `ADV_STORAGE_MODE`, `DELETE_ADV_AFTER_USE` | `delta_fp16`, 1 |
| Chạy lại | `FORCE_ATTACKS` | 1 để bỏ cache kết quả trong launcher có hỗ trợ reuse |

`EPS`, `ALPHA` phải hữu hạn và không âm; `STEPS`, các batch size và số mẫu phải là số nguyên dương. `STEPS` độc lập với `ALPHA`: khi đổi số bước, hãy chọn lại bước cập nhật phù hợp. Table VIII không tự đổi ngân sách thành 4/255; ví dụ muốn dùng ngân sách đó:

```bash
DEFENSE_EPS=4/255 DEFENSE_ALPHA=0.4/255 DEFENSE_STEPS=10 \
bash sh/run_table8_defense.sh
```

Cache tính cả ngân sách attack, cấu hình batch, CSV tập con và code. Tập con đã chọn vẫn là một phần của thí nghiệm: khi đổi dữ liệu, nhãn, seed, checkpoint hoặc model dùng chọn mẫu, dùng CSV/output mới để chọn lại ảnh sạch đúng. Cache không băm nội dung toàn bộ checkpoint và ảnh.

### 7.2. Batch tự giảm khi CUDA OOM

Mỗi lệnh bắt đầu với batch đã yêu cầu, mặc định 16. Khi gặp lỗi **CUDA out of memory trong xử lý ảnh**, chương trình giải phóng tensor của lần lỗi rồi giảm một nửa số ảnh của phần xử lý đang lỗi, làm tròn xuống và chặn ở `MIN_BATCH_SIZE`. Với batch đủ 16 ảnh, chuỗi điển hình là 16 → 8 → 4 → 2 → 1; batch cuối có 5 ảnh sẽ giảm 5 → 2 → 1 nếu tiếp tục OOM. Chỉ phần chưa hoàn thành được thử lại; kết quả thành công trước đó được giữ. Batch đã giảm được dùng tiếp trong cùng lần chạy; lệnh/method/target mới có thể bắt đầu lại ở batch đã cấu hình.

Batch logic và batch thực thi được tách biệt. Ví dụ `BATCH_SIZE=16 NUM_BATCHES=10` vẫn yêu cầu tối đa **160 ảnh**, kể cả GPU chỉ chạy được 4 ảnh mỗi lần. Một file batch attack vẫn chứa tối đa 16 ảnh; evaluation đọc đủ các ảnh này theo các phần nhỏ hơn. Không bỏ ảnh và không giảm ngân sách thí nghiệm khi giảm batch thực thi. Khi có ít hơn số ảnh yêu cầu trong dataset, chỉ xử lý số ảnh thực có.

Nếu chưa đặt `NUM_IMAGES` và đã đặt `NUM_BATCHES`, số ảnh cần chọn được suy ra bằng `NUM_BATCHES × BATCH_SIZE` **ban đầu**. Nếu đặt cả hai, `NUM_IMAGES` quyết định tập con được chọn và `NUM_BATCHES` giới hạn số batch logic được chạy.

Để tắt cơ chế này hoặc đặt kích thước nhỏ ngay từ đầu:

```bash
AUTO_BATCH=0 BATCH_SIZE=8 bash sh/run_tables_1_8.sh
```

CLI Python tương ứng dùng `--no-auto-batch`, `--min-batch-size` và cờ batch size của từng script. Lỗi khác CUDA OOM được báo nguyên vẹn; nếu một ảnh vẫn OOM hoặc OOM ngay lúc nạp model, chương trình dừng với thông báo rõ ràng.

Kích thước batch thực thi và số lần OOM được ghi cùng kết quả: tập con có `<selected.csv>.metadata.json`, attack có `attack_config.json` và cột microbatch trong CSV, evaluation có `eval_summary.json`, chất lượng ảnh có `quality_execution.json` trong mỗi thư mục attack, runtime có thêm các cột batch thực tế và số lần retry. Các cột peak VRAM của attack ghi cực đại trong các microbatch thành công, không tính lần chạy OOM đã bị hủy.

Đổi batch có thể thay đổi phép lấy mẫu view ngẫu nhiên và thống kê phụ thuộc batch; không bảo đảm ảnh adversarial giống từng bit giữa hai GPU. Khi so sánh thuật toán hoặc runtime chính thức, nên chọn batch cố định mà mọi phương pháp chạy được, lưu cấu hình và kiểm tra metadata thực thi.

Các lệnh chạy:

```bash
# Lần thử nhỏ trên máy thực nghiệm: 16 ảnh, một cấu hình Table V.
NUM_IMAGES=16 NUM_BATCHES=1 \
TABLE5P_SETTINGS=cnn_to_vit TABLE5P_VARIANTS=full_model \
bash sh/run_table5_ablation.sh

# Tables I–VIII, mặc định 1.000 ảnh.
bash sh/run_tables_1_8.sh

# Chưa có robust checkpoints: bỏ riêng Table VIII.
SKIP_TABLE8=1 bash sh/run_tables_1_8.sh

# Tables và hình.
bash sh/run_all_outputs.sh

# Table V với 5.000 ảnh.
TABLE5P_NUM_IMAGES=5000 bash sh/run_table5_ablation.sh
```

### 7.3. CLI độc lập

```bash
python scripts/select_imagenet_subset.py \
  --data-dir /data/imagenet/val \
  --labels-csv /data/imagenet/imagenet_val_labels.csv \
  --surrogates resnet50,densenet121,vit_base_patch16_224,deit_small_patch16_224 \
  --num-images 1000 --batch-size 16 --device cuda \
  --out-csv runs/selected_1000.csv

python scripts/run_attack.py \
  --data-dir /data/imagenet/val --selected-csv runs/selected_1000.csv \
  --surrogates resnet50 --attack sv_fca --variant full_model \
  --eps 0.06274509803921569 --alpha 0.006274509803921569 --steps 10 \
  --num-views 4 --spectral-bands 6 --batch-size 16 --device cuda \
  --storage-mode adv_fp32 --out-dir runs/example

python scripts/evaluate.py \
  --attack-dir runs/example --data-dir /data/imagenet/val \
  --targets resnet152,inception_v3 --eval-batch-size 16 --device cuda
```

### 7.4. Các launcher và đầu ra

| Launcher | Công việc | CSV chính dưới OUT_DIR |
| --- | --- | --- |
| `run_table1.sh` | Tables I–IV, tám attack, mỗi source riêng | `table1_single_source_transfer/table{1,2,3,4}_*.csv` |
| `run_table5_ablation.sh` | Sáu ablation trên bốn nhóm transfer | `table5_prime/table5_prime.csv` |
| `run_table6_quality.sh` | Chất lượng ảnh, nhóm CNN và ViT source | `table6_quality/table6_quality.csv` |
| `run_table7_runtime.sh` | Runtime, mặc định nhóm mixed source | `table7_runtime/table7_runtime.csv` |
| `run_table8_defense.sh` | ASR trên robust targets | `table8_defense/table8_defense.csv` |
| `run_tables_5_8.sh` | Tables V–VIII | Các file tương ứng ở trên |
| `run_tables_1_8.sh` | Tables I–VIII | Các file tương ứng ở trên |
| `run_figures.sh` | Triplet, FFT, attention/saliency | `figures/*.png`, radial profile CSV |
| `run_all_outputs.sh` | Tables và hình | Cả hai nhóm |
| `select_5000.sh` | Chọn 5.000 ảnh clean-correct | CSV tập con |
| `download_models.sh` | Tải/cache model timm | Cache MODEL_DIR |

Thư mục `sh/` còn 12 file, gồm `common.sh` và 11 launcher trong bảng trên. Các wrapper trùng chức năng, launcher single-source cũ và ablation token/fusion/precision weighting lỗi thời đã được bỏ. Dùng `run_table1.sh` để tạo Tables I–IV và `run_table5_ablation.sh` cho Table V.

Table V mặc định dùng `NUM_IMAGES` trong cấu hình chung. Đặt `TABLE5P_NUM_IMAGES=1000` hoặc `5000` khi cần số ảnh riêng; kết quả tương ứng được ghi vào `table5_prime_1000` hoặc `table5_prime_5000`. Các tên thư mục output và biến `TABLE5P_*` được giữ để tương thích với kết quả đã có.

`ours`, `sv_fca`, `svfca`, `sv_fta`, `svfta`, `ddc` đều trỏ đến cùng class SV-FCA. Các cờ legacy như `--fusion`, `--rho`, `--lambda-grid`, `--eps-c`, `--c-min`, `--c-max`, `--disable-precision-weighting`, `--soft-sigma`, `--soft-power`, `--low-cutoff`, `--mid-cutoff`, `--low-prior`, `--mid-prior`, `--high-prior`, `--log-spectral-energy` được giữ để đọc lệnh cũ nhưng không điều khiển thuật toán SV-FCA hiện tại.

## 8. Định nghĩa kết quả

### 8.1. ASR trên target

Gọi $N$ là số ảnh đã đánh giá, $C_i$ chỉ việc target đoán đúng ảnh sạch, $A_i$ chỉ việc target đoán sai ảnh adversarial:

```math
\mathrm{ASR}_{\mathrm{clean\ correct}}
=100\frac{\sum_{i=1}^{N} C_i A_i}{\sum_{i=1}^{N} C_i}
```

```math
\mathrm{ASR}_{\mathrm{all}}
=100\frac{\sum_{i=1}^{N} A_i}{N}.
```

`asr` là alias của `asr_clean_correct`, không phải `asr_all`. Tập ảnh sạch đúng trên surrogate chưa chắc đúng trên target; mẫu số cần tính riêng cho từng target. Nếu không có ảnh sạch đúng, ASR có điều kiện không xác định và phải biểu diễn thiếu dữ liệu, không diễn giải thành 0%.

`clean_acc` và `adv_acc_all` là accuracy phần trăm trên toàn bộ mẫu; `robust_acc_clean_correct` là accuracy adversarial trên tập clean-correct của target. Hàng `AVG` trong `eval_results.csv` gộp các số đếm trên cặp ảnh–target, nên ASR có điều kiện được cân theo số clean-correct của từng target. Table V dùng hàng `AVG` này cho từng setting rồi lấy trung bình đều các setting ở cột `Average`. Table VIII lấy trung bình đều ASR từng target ở cột `Average` (macro). Hai cách có thể cho số khác nhau. Chỉ xuất `Average` khi đủ các setting/target yêu cầu; ô trống biểu thị thiếu/không xác định, không phải attack thất bại 0%.

### 8.2. PSNR, SSIM, LPIPS

Với ảnh chuẩn hóa `[0, 1]`:

```math
\mathrm{MSE}=\frac1D\sum_j(x_j-x^{\mathrm{adv}}_j)^2
```

```math
\mathrm{PSNR}=10\log_{10}\frac{1}{\mathrm{MSE}}.
```

Hai ảnh bằng nhau có MSE bằng 0 và PSNR bằng **dương vô cùng** (`inf` trong CSV).

Trong công thức SSIM, $z=x^{\mathrm{adv}}$ là ảnh đối kháng; $\mu_x$, $\mu_z$ là trung bình cục bộ, $\sigma_x^2$, $\sigma_z^2$ là phương sai cục bộ và $\sigma_{xz}$ là hiệp phương sai cục bộ.

```math
\mathrm{SSIM}=\frac{(2\mu_x\mu_z+C_1)(2\sigma_{xz}+C_2)}
{(\mu_x^2+\mu_z^2+C_1)(\sigma_x^2+\sigma_z^2+C_2)}
```

```math
C_1=0.01^2,\quad C_2=0.03^2.
```

SSIM ở đây dùng cửa sổ **trung bình đều 11 × 11**, zero-padding, lấy trung bình theo channel và pixel; không đồng nhất với Gaussian-window SSIM của mọi thư viện khác. LPIPS dùng `lpips.LPIPS(net="alex")`, chuyển ảnh sang `[-1, 1]`. Nếu không có thư viện LPIPS, cột được để trống và chương trình thông báo. Các metric được tính theo ảnh rồi lấy trung bình; PSNR/SSIM càng cao và LPIPS càng thấp thì perturbation càng ít khác biệt theo metric đó.

### 8.3. Thời gian

```math
\mathrm{ms/image}=1000\frac{\text{tổng thời gian attack đã đo}}{\text{số ảnh thực sự đã đo}}
```

```math
\mathrm{relative\ cost}=\frac{\mathrm{ms/image}_{\mathrm{method}}}{\mathrm{ms/image}_{\mathrm{DI\text{-}FGSM}}}.
```

Warmup chạy riêng và không tính vào số đo. CUDA được synchronize trước/sau đoạn đo. Thời gian bao gồm việc thực thi attack và logging bên trong attack; không tính tải checkpoint, đọc dataset, chuyển tensor lên GPU, lưu batch, đánh giá target hoặc thời gian của lần chạy bị OOM. Chỉ thời gian các microbatch thành công đóng góp vào ms/image; số lần thử lại được báo riêng. Baseline relative cost là DI-FGSM; nếu danh sách không có DI-FGSM thì dùng method đầu tiên. So sánh phải dùng cùng phần cứng, source set, batch size, số bước và chế độ precision. `RUNTIME_IMAGES` mặc định 128. Khi đặt `NUM_BATCHES`, giới hạn này **thay thế** `RUNTIME_IMAGES`: đo theo số batch, không bao gồm warmup. Khi không đặt `NUM_BATCHES`, code mới giới hạn theo số ảnh và cắt batch cuối trước khi đo.

### 8.4. Batch lưu trên đĩa và hình minh họa

Mỗi attack ghi `attack_config.json`, `attack_batches.csv`, `attack_steps.csv` và các file `.pt` trong thư mục batch trung tâm. Evaluation ghi `eval_results.csv`. Manifest mới lưu đường dẫn batch tương đối với thư mục attack, nên có thể di chuyển cả cây kết quả cùng nhau. Manifest là nguồn xác định các batch hợp lệ; mất một file phải báo lỗi để không âm thầm thay đổi cỡ mẫu. Nếu chỉ đánh giá và xóa một phần batch, manifest giữ các batch chưa dùng.

- `adv_fp32`: lưu ảnh adversarial float32.
- `delta_fp16`: lưu perturbation float16 và tái dựng từ ảnh sạch; giảm gần một nửa dung lượng tensor nhưng có sai số lượng tử hóa. Batch mới lưu epsilon để tái chiếu khi đọc, tránh sai số làm tròn vượt ngân sách. Khi cần so sánh chính xác, dùng `adv_fp32`.
- Xóa batch sau sử dụng làm mất khả năng đánh giá/vẽ lại từ tensor cũ; đặt `DELETE_ADV_AFTER_USE=0` để giữ. Thông số và CSV kết quả vẫn được lưu.
- Phổ hình minh họa lấy trung bình ba channel thành ảnh xám, tính `log1p(abs(fftshift(fft2(image))))`, rồi chuẩn hóa riêng mỗi panel. Giá trị này không biểu diễn năng lượng phổ tuyệt đối.
- Attention rollout chỉ có ở model hỗ trợ; trường hợp còn lại dùng gradient saliency và ghi rõ tên phương pháp trên hình.

## 9. Kiểm tra và cấu trúc mã nguồn

```bash
python -m compileall -q attacks src scripts tests
bash -n experiment.venv
for script in sh/*.sh; do bash -n "$script"; done
for script in scripts/*.py; do python "$script" --help > /dev/null; done
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m pytest -q
```

Các kiểm thử nhỏ dùng tensor/model giả để kiểm tra phép chiếu, FFT, consensus, metric và luồng đọc/ghi mà không tải ImageNet hoặc checkpoint. GitHub Actions chạy cùng nhóm kiểm tra CPU. Kiểm thử này không thay thế việc chạy đầy đủ Tables I–VIII trên checkpoint thật và GPU.

```text
attacks/                 Baseline, factory và SV-FCA
src/                     Dataset, wrapper model, FFT, metric và tiện ích
scripts/                 CLI chọn ảnh, attack, evaluate, bảng và hình
sh/                      Bash launcher và cấu hình dùng chung
experiment.venv          Cấu hình chung duy nhất cho tất cả launcher
tests/                   Kiểm thử hồi quy nhỏ, không cần trọng số pretrained
.github/workflows/       Kiểm tra tự động trên GitHub
requirements*.txt        Dependencies chính, tùy chọn và kiểm thử
```

Các `PATCH_NOTES_*.md` là lịch sử của những bản trước; một số mô tả SV-FTA/token/weighting đã hết hiệu lực. Dùng README này và code hiện tại làm tham chiếu. Repository không chứa bản paper hoặc dữ liệu chứng minh ưu thế của phương pháp; khi công bố cần bổ sung kết quả thực nghiệm, thông tin checkpoint, seed, cấu hình và trích dẫn phù hợp.

## 10. Công thức trên GitHub

Công thức trong dòng dùng `$...$`; công thức riêng dùng khối có nhãn `math`, được GitHub hỗ trợ. Các tên hàm viết bằng `\mathrm{...}`. Công thức dài được tách thành các biểu thức ngắn để đọc được trong cột README.

Khi sửa tài liệu, cần kiểm tra cả trang README đã render trên GitHub: kiểm tra LaTeX bằng MathJax cục bộ chưa đủ để phát hiện macro bị GitHub từ chối hoặc ký tự bị Markdown biến đổi. Trình xem Markdown khác cần hỗ trợ toán học để hiển thị các công thức này.

## 11. Tham khảo baseline

- I-FGSM/BIM: [Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533).
- MI-FGSM: [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081).
- DI: [Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978).
- TI: [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884).
- SI-NI-FGSM: [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281).

Các liên kết là nguồn phương pháp gốc; các khác biệt của implementation trong repository được nêu ở mục 5. `freq_only`, `vit_aware` và SV-FCA được mô tả theo code của dự án, không gán nhầm cho những paper trên.
