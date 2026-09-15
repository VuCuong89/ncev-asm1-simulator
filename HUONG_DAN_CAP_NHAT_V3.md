# HƯỚNG DẪN DEPLOY V3.0 — TỪNG BƯỚC

## Mục tiêu
Tạo một app test v3.0 riêng, không ảnh hưởng app chính và các nhánh v2.x.

## Bước 1 — Giải nén
Giải nén `NCEV_AO_ASM1_Cloud_v3_0.zip`.
Mở thư mục `ncev_asm1_cloud_deploy_v3_0`.

## Bước 2 — Tạo branch test mới trên GitHub
1. Mở repo `VuCuong89/ncev-asm1-simulator`.
2. Bấm ô branch đang ghi `main`.
3. Gõ `v3-0-test`.
4. Chọn `Create branch: v3-0-test from main`.
5. Kiểm tra góc trên đang là `v3-0-test`.

## Bước 3 — Upload v3.0 vào root branch
1. Trong branch `v3-0-test`, bấm `Add file` → `Upload files`.
2. Mở BÊN TRONG thư mục `ncev_asm1_cloud_deploy_v3_0` trên máy.
3. Chọn toàn bộ NỘI DUNG bên trong, kéo vào GitHub.
4. Không kéo nguyên thư mục ngoài cùng.

Cấu trúc đúng phải là:

```text
app.py
asm1_engine.py
chemistry.py
fractionation.py
i18n.py
requirements.txt
assets/
    ncev_logo.png
.streamlit/
    config.toml
...
```

## Bước 4 — Commit
Commit message: `Deploy NCEV Cloud v3.0 test`
Sau đó bấm `Commit changes`.

## Bước 5 — Kiểm tra file quan trọng
Trước khi sang Streamlit, xác nhận tại root branch có:
- `app.py`
- `requirements.txt`
- `assets/ncev_logo.png`
- `.streamlit/config.toml`

Nếu `.streamlit` không hiện, v3.0 vẫn có CSS ép nền sáng, nhưng nên upload đủ để giao diện ổn định hơn.

## Bước 6 — Tạo app test Streamlit
1. Vào Streamlit Community Cloud → `My apps` → `Create app`.
2. Repository: `VuCuong89/ncev-asm1-simulator`
3. Branch: `v3-0-test`
4. Main file path: `app.py`
5. Python: chọn 3.13 nếu được hỏi.
6. URL gợi ý: `ncev-asm1-v30-test`
7. Bấm `Deploy`.

## Bước 7 — Checklist sau deploy
Chưa chạy mô phỏng, kiểm tra trước:
1. Nền toàn trang trắng.
2. Sidebar nền xanh rất nhạt hoặc trắng.
3. Ô số nền trắng, chữ tối, 4 cạnh viền đều.
4. Nút +/- xanh nhạt.
5. Language nền trắng, chữ tối, đủ 4 cạnh.
6. Methanol/Ethanol và PAC/Polytetsu nền trắng, chữ tối.
7. Nút `Chạy mô phỏng` nền xanh NCEV, chữ trắng.
8. Không có mảng input màu đen.

Sau đó chạy mô phỏng và kiểm tra:
9. KPI không có dấu `...`.
10. Biểu đồ nền trắng, trục X/Y đọc rõ.
11. Download CSV/JSON nhìn rõ.
12. Bảng Scenario nền trắng, header xanh nhạt, chữ tối.

Nếu bất kỳ mục nào sai, chụp đúng vùng lỗi và gửi lại để sửa trên cùng branch `v3-0-test` — không tạo thêm branch mới.
