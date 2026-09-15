# NCEV AO/ASM1 Simulator — Cloud v3.0

Bản v3.0 giữ nguyên engine ASM1 + hóa chất + bùn, nhưng viết lại lớp giao diện từ đầu để tránh xung đột theme đã gặp ở nhánh v2.x.

## Chức năng chính
- Tiếng Việt / English / 中文简体
- ASM1 13 biến trạng thái, 8 quá trình, SciPy BDF
- Methanol / Ethanol: tối ưu liều carbon và đưa ngược vào ASM1
- NaOH 10%: tính theo thiếu hụt độ kiềm
- PAC / Polytetsu: tính khử T-P
- Bùn sinh học + bùn hóa học + tổng DS + bùn ướt
- Chế độ nhập WAS hoặc SRT mục tiêu
- So sánh kịch bản
- CSV / JSON

## Giao diện v3.0
- Nền trắng cố định
- Chữ xanh đen
- Input/select nền sáng, viền 1 px đồng đều
- Nút +/- xanh nhạt
- KPI tự dựng, không cắt thành "..."
- Biểu đồ nền trắng, trục tối
- Bảng Scenario HTML nền sáng
- Logo cục bộ, không phụ thuộc URL ngoài

Xem `HUONG_DAN_CAP_NHAT_V3.md` để deploy.
