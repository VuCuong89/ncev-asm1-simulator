# Cloud v3.0 — Release Notes

## Thay đổi lớn
- Viết lại UI từ đầu, không kế thừa CSS v2.x.
- Giữ nguyên engine ASM1/chemistry/sludge.
- Dùng cấu hình theme tối giản, chỉ dùng các khóa ổn định.
- CSS chỉ tập trung vào các control bắt buộc: nền trang, input, select, button, KPI, chart/table.
- Logo được đóng gói local tại `assets/ncev_logo.png`.
- Không dùng `st.metric` cho KPI chính để tránh cắt đơn vị bằng `...`.
- Bảng Scenario dùng HTML table thay cho dataframe canvas.

## Engine test
Smoke test giữ kết quả chuẩn:
- Base TN ~11.226 mgN/L
- NH4 ~0.679 mgN/L
- Carbon optimization đạt TN ~9.996 mgN/L trong test target 10 mgN/L
- Target SRT 15 d → WAS ~10.827 m3/d, SRT ~14.991 d
