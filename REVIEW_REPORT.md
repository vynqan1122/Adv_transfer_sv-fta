# Rà soát code và Bash — 2026-09-09

## Phạm vi

Đọc toàn bộ module trong `attacks/`, `src/`, `scripts/`, toàn bộ 23 file Bash trong `sh/` và cấu hình mẫu. Viết lại README chính bằng tiếng Việt, mô tả phép tính thực tế và dùng cú pháp toán GitHub.

## Các lỗi đã sửa

| Nhóm | Vấn đề trước đây | Kết quả sau sửa |
| --- | --- | --- |
| Import | Comment không phải UTF-8 trong ViT-aware làm import package attacks lỗi | File Python dùng UTF-8 hợp lệ |
| Model | Mọi timm checkpoint dùng cùng mean/std và không thích ứng input size | Lấy preprocessing từ cấu hình checkpoint, resize có gradient |
| SV-FCA | Consensus trừ số phần tử ngay cả khi band gradient bằng 0 | Trừ tổng bình phương norm thực tế, khớp tính từng cặp |
| FFT | Lưới linspace làm sai tần số và đối xứng liên hợp | Dùng fftfreq đúng bin, kiểm tra khôi phục và sinusoid |
| Baseline | Nesterov lookahead bị clip; top-k patch chọn dư khi đồng hạng | Lookahead đúng công thức, chọn đúng số patch |
| Số học | Gradient fp16 bằng 0 có thể NaN; tham số sai bị dùng âm thầm | Chuẩn hóa ổn định và báo lỗi tham số không hợp lệ |
| Dữ liệu | Dataset subset có thể bị đánh lại index class; đường dẫn không thống nhất | Kiểm tra nhãn ImageNet, CSV không rỗng, dùng chung cách resolve đường dẫn |
| Batch | Figure chỉ đọc ảnh đầy đủ, không đọc delta_fp16; manifest bỏ qua file thiếu | Reader dùng chung cho eval, quality, figures; hỗ trợ cả hai format và báo file thiếu |
| Ngân sách | Làm tròn delta float16 có thể vượt epsilon | Lưu epsilon và chiếu lại perturbation khi tái dựng |
| ASR | Không có clean-correct vẫn xuất 0% | ASR có điều kiện không xác định được để trống/null |
| Metric | PSNR ảnh trùng nhau bị chặn ở 120 dB | Trả dương vô cùng; giữ định nghĩa SSIM uniform của dự án |
| Runtime | Warmup ăn mất batch đo; batch cuối chỉ sửa mẫu số thời gian | Warmup riêng, đo đúng tensor và số ảnh thực tế |
| Tổng hợp | Có thể lấy nhầm metric hoặc dùng target cuối thay AVG | Đọc đúng metric/AVG; không tạo Average từ bảng thiếu cấu hình |
| Bash | Chạy từ thư mục khác lỗi source; cache dùng batch đã xóa | Đường dẫn theo vị trí script; kiểm tra file và chữ ký cấu hình trước reuse |
| Bash | Fusion/token/precision weighting cũ không còn tác dụng nhưng vẫn tạo hàng thí nghiệm | Launcher cảnh báo và chuyển sang ablation hiện có |
| Bash | Cleanup figure sai nơi lưu tensor; figures có thể chạy hai lần | Cleanup đúng batch trung tâm và chạy figures một lần |
| Cấu hình | Script chọn 5000 nhưng mặc định lấy 1000; đường dẫn cá nhân hardcode | Đồng bộ số ảnh, biến môi trường và đường dẫn dùng được trên máy khác |
| Tài liệu | README thiếu thuật toán và có tên legacy gây nhầm | README chính đầy đủ, hai README cũ chuyển thành liên kết tương thích |

## Kết quả kiểm tra

| Kiểm tra | Kết quả |
| --- | --- |
| Python compileall | PASS toàn bộ attacks, src, scripts và tests |
| Bash syntax và workflow mô phỏng | PASS toàn bộ 23 file SH; kiểm tra đường dẫn có khoảng trắng/dấu phẩy, cache và cleanup |
| Python script entry points | PASS gọi `--help` trên 18 file scripts |
| Pytest CPU | **35 passed, 18 subtests passed** |
| Pipeline tổng hợp | PASS chọn ảnh → SV-FCA → đọc hai batch (batch cuối thiếu) → evaluate micro-batch → quality, cả adv_fp32/delta_fp16 |
| timm API thực | 9 tên model mặc định được đăng ký; ResNet50, Inception-v3 và ViT-B chạy forward/backward với trọng số khởi tạo ngẫu nhiên, gradient hữu hạn |
| Dependency consistency | `pip check`: No broken requirements found |
| README MathJax | **92 công thức, 0 lỗi** |

Môi trường kiểm tra: WSL Linux, Python 3.12.3, PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu, timm 1.0.29, NumPy 1.26.4, Pillow 10.2.0, Matplotlib 3.11.1. Có một warning Matplotlib về Axes3D do môi trường dùng chung package hệ thống; không có test thất bại. Các hình của dự án dùng trục 2D.

## Giới hạn của lần kiểm tra

Kiểm thử sử dụng model nhỏ và dữ liệu tổng hợp, không tải trọng số nghiên cứu. Chưa chạy đầy đủ ImageNet, Tables I–VIII, CUDA/AMP, LPIPS pretrained hoặc các checkpoint RobustBench thật. Vì sửa preprocessing, consensus, FFT và SI-NI, kết quả từ bản cũ cần chọn lại tập con và chạy lại vào thư mục output mới.

Các công thức được kiểm tra bằng MathJax. Kiểm tra cú pháp/định dạng không chứng minh tính mới của phương pháp hoặc chất lượng chuyển giao; các kết luận đó cần số liệu thực nghiệm.

## Cập nhật cấu hình và xử lý OOM

- Gom cấu hình vào `experiment.venv`; `sh/common.sh` tự đọc, kiểm tra tham số và cung cấp các hàm gọi chung. Thay mẫu `configs/sv_fca_tables_1_8.env.example` bằng file cấu hình này.
- Mặc định batch 16 cho chọn ảnh, attack, evaluation, quality và runtime. Khi xử lý ảnh gặp CUDA OOM, giảm kích thước phần thực thi và thử lại đúng phần lỗi; giữ batch logic, thứ tự ảnh và giới hạn số mẫu.
- `EPS`, `ALPHA`, `STEPS` chỉnh được ở cấu hình chung; hỗ trợ phân số như `4/255`. Table VIII có `DEFENSE_EPS`, `DEFENSE_ALPHA`, `DEFENSE_STEPS` kế thừa hoặc ghi đè ngân sách riêng.
- Khôi phục trạng thái RNG sau lần lỗi, giải phóng tensor trước khi retry và chỉ cộng số liệu của phần đã hoàn thành. Ghi metadata batch thực tế/OOM để so sánh thí nghiệm.
- CPU OOM, lỗi model khác và OOM khi tải trọng số không bị che bằng vòng retry batch. Đổi cách chia batch có thể thay đổi các biến đổi ngẫu nhiên hoặc thống kê phụ thuộc batch; không khẳng định tương đương từng bit với batch ban đầu.

Kiểm thử cập nhật: **51 tests và 29 subtests đạt trong 77,38 giây**, một luồng CPU, vô hiệu hóa GPU; dùng tensor/model giả và OOM mô phỏng. Không tải checkpoint, không chạy ImageNet hoặc phép đo hiệu năng thật. Thư viện kiểm thử được tái sử dụng từ cache cục bộ. Python/Bash syntax, CLI mới và `pip check` đều đạt. Warning Matplotlib Axes3D của môi trường tạm không ảnh hưởng các kiểm thử này.

README hiện có 123 công thức. Bản sửa hiển thị trước đó đã được xác nhận trực tiếp trên GitHub (123/123 render, không lỗi macro); cập nhật cấu hình giữ nguyên các công thức và kiểm tra lại cú pháp MathJax.
