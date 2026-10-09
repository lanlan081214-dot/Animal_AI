# ANIMAL AI – Nhận diện động vật bằng AI

Dự án xây dựng ứng dụng nhận diện động vật từ hình ảnh bằng mô hình học sâu **EfficientNet-B0**, kết hợp **Streamlit** để tạo giao diện web và chatbot **Gemini** để hỗ trợ hỏi đáp về động vật bằng tiếng Việt.

> **Lưu ý:** Ứng dụng chỉ có thể dự đoán các loài thuộc danh sách lớp mà mô hình đã được huấn luyện. Kết quả AI có thể sai, vì vậy không nên xem dự đoán là xác nhận tuyệt đối.

## 1. Chức năng chính

- Tải ảnh động vật lên giao diện web.
- Dùng mô hình EfficientNet-B0 để dự đoán loài động vật.
- Hiển thị các dự đoán có xác suất cao, tùy theo cách triển khai trong `app.py`.
- Hỗ trợ chatbot tiếng Việt để hỏi thêm về động vật, chẳng hạn đặc điểm, thức ăn, môi trường sống và tập tính (nếu phần chatbot đã được cấu hình).
- Đánh giá mô hình bằng tập kiểm thử và xuất báo cáo kết quả.

## 2. Công nghệ sử dụng

- **Python** – ngôn ngữ lập trình chính.
- **PyTorch / Torchvision** – xây dựng và chạy mô hình học sâu.
- **EfficientNet-B0** – mô hình phân loại ảnh.
- **Streamlit** – xây dựng giao diện web.
- **Google Gemini API** – chatbot hỏi đáp (cần API key nếu sử dụng tính năng này).
- **Pandas, Pillow, python-dotenv** – xử lý dữ liệu, ảnh và biến môi trường.

## 3. Cấu trúc thư mục

Cấu trúc dưới đây mô tả các thành phần chính; tên hoặc một số thư mục phụ có thể khác tùy phiên bản dự án hiện tại.

```text
Animal_AI/
├── app.py
├── train_cnn.py
├── test_cnn_200_v6_fixed.py
├── models/
│   ├── animal_ai_efficientnet_b0_200_v6_best.pth
│   └── class_names_200_v6_train.json
├── data_200_v6/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset/
├── reports/
│   ├── test_v6_summary.txt
│   └── test_v6_per_class.csv
├── .env
└── requirements.txt
```

- `app.py`: chạy giao diện ứng dụng.
- `train_cnn.py`: mã huấn luyện mô hình.
- `test_cnn_200_v6_fixed.py`: đánh giá mô hình V6 trên tập test.
- `models/`: lưu trọng số mô hình và ánh xạ tên lớp.
- `data_200_v6/`: dữ liệu đã chia thành tập train, validation và test.
- `reports/`: báo cáo đánh giá mô hình.
- `.env`: lưu API key cục bộ; **không chia sẻ file này công khai**.

## 4. Yêu cầu trước khi chạy

- Windows 10/11 hoặc môi trường có thể chạy Python.
- Python tương thích với các thư viện trong dự án.
- Các thư viện được khai báo trong `requirements.txt`.
- Đã có các file mô hình và class map cần thiết trong thư mục `models/`.
- Nếu chạy chức năng chatbot Gemini, cần có API key hợp lệ.

Mô hình V6 được đánh giá trên CPU trong môi trường phát triển. GPU không bắt buộc để chạy suy luận, nhưng có thể giúp huấn luyện nhanh hơn nếu được cấu hình phù hợp.

## 5. Cài đặt và chạy dự án trên Windows

Mở **PowerShell** tại thư mục dự án.

### Bước 1: Tạo môi trường ảo

```powershell
python -m venv .venv
```

### Bước 2: Kích hoạt môi trường ảo

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn việc chạy script, có thể cho phép trong phiên hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Bước 3: Cài thư viện

Nếu dự án có `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chưa có file này, hãy cài các thư viện theo danh sách import trong mã nguồn hoặc tạo `requirements.txt` phù hợp trước khi cài đặt.

### Bước 4: Kiểm tra file mô hình

Đảm bảo các file mà `app.py` đang cấu hình tồn tại, ví dụ:

```text
models/animal_ai_efficientnet_b0_200_v6_best.pth
models/class_names_200_v6_train.json
```

Tên file trong `app.py` phải khớp với file thực tế. Class map phải đúng thứ tự lớp đầu ra tương ứng với checkpoint; nếu không, ứng dụng có thể hiển thị sai tên loài dù mô hình đã dự đoán một chỉ số lớp.

### Bước 5: Cấu hình Gemini (nếu sử dụng)

Tạo file `.env` tại thư mục gốc dự án và đặt API key theo đúng tên biến mà `app.py` đọc. Ví dụ, nếu mã nguồn dùng biến `GEMINI_API_KEY`:

```env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Thay `YOUR_API_KEY_HERE` bằng key của bạn. Không đưa API key thật vào README, GitHub hoặc file gửi cho người khác. Giáo viên có thể chạy phần nhận diện ảnh mà không cần Gemini nếu ứng dụng cho phép bỏ qua chatbot; muốn dùng chatbot thì cần tự cấu hình key.

### Bước 6: Khởi chạy ứng dụng

```powershell
streamlit run app.py
```

Streamlit thường hiển thị địa chỉ cục bộ trong terminal, ví dụ `http://localhost:8501`. Mở địa chỉ đó trên trình duyệt.

## 6. Kiểm tra mô hình V6

Nếu tập dữ liệu kiểm thử và script đánh giá đã có sẵn, có thể chạy:

```powershell
python test_cnn_200_v6_fixed.py
```

Theo lần đánh giá V6 gần nhất của dự án:

| Chỉ số | Kết quả |
|---|---:|
| Số ảnh kiểm thử | 3.600 |
| Số lớp | 200 |
| Test Loss | 1.9914 |
| Top-1 Accuracy | 52.94% |
| Top-5 Accuracy | 76.94% |

**Ý nghĩa:**
- **Top-1 Accuracy**: tỷ lệ ảnh mà dự đoán đứng đầu tiên đúng với nhãn thật.
- **Top-5 Accuracy**: tỷ lệ ảnh mà nhãn thật xuất hiện trong 5 dự đoán đứng đầu.

Đây là kết quả của lần đánh giá đã ghi nhận, không phải cam kết rằng mọi ảnh tải lên ứng dụng đều có độ chính xác tương tự. Kết quả thực tế còn phụ thuộc vào chất lượng ảnh và loài cần nhận diện.

Sau khi chạy, script tạo các báo cáo trong thư mục `reports/` nếu đường dẫn đầu ra được cấu hình như trên:
- `test_v6_summary.txt`
- `test_v6_per_class.csv`

## 7. Dữ liệu và mô hình

Dự án sử dụng dữ liệu động vật từ bộ dữ liệu iNat-Anim. Thông tin bộ dữ liệu gốc:

- Tên: **iNat-Anim**
- Nguồn: https://zenodo.org/records/6703088

Tập dữ liệu gốc có dung lượng lớn. Vì vậy, thư mục dữ liệu và trọng số mô hình có thể không được đính kèm khi gửi mã nguồn. Nếu thiếu các file này, cần bổ sung chúng trước khi chạy ứng dụng hoặc đánh giá mô hình.

## 8. Một số lỗi thường gặp

**Không tìm thấy file `.pth`**
- Kiểm tra file trọng số có nằm trong `models/` không.
- Đối chiếu tên file trong `app.py` với tên file thực tế.

**Ứng dụng dự đoán sai tên loài bất thường**
- Kiểm tra checkpoint và class map có thuộc cùng phiên bản huấn luyện không.
- Đảm bảo thứ tự tên lớp khớp với `class_to_idx` dùng khi huấn luyện.
- Kiểm tra xem loài đó có nằm trong danh sách 200 lớp hay không.

**Không chạy được `streamlit`**
- Kích hoạt `.venv`.
- Cài các thư viện còn thiếu.
- Chạy lệnh từ thư mục chứa `app.py`.

**Chatbot không hoạt động**
- Kiểm tra API key và tên biến môi trường mà mã nguồn sử dụng.
- Kiểm tra kết nối mạng và quyền sử dụng API.
- Đảm bảo file `.env` được đọc đúng.

## 9. Hướng dẫn cho người chấm

1. Mở thư mục dự án `Animal_AI`.
2. Cài đặt môi trường và thư viện theo Mục 5.
3. Kiểm tra file checkpoint và class map trong `models/`.
4. Chạy `streamlit run app.py`.
5. Tải một ảnh động vật lên để thử chức năng nhận diện.
6. Nếu cần kiểm tra chỉ số mô hình, chạy script test ở Mục 6 và xem báo cáo trong `reports/`.

**Trước khi nộp bài:** kiểm tra lại tên file và thư mục trong README so với dự án thực tế; gửi kèm các file mô hình cần thiết hoặc hướng dẫn nơi lấy chúng. Không gửi `.env` có chứa API key thật.
