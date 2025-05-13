# Hệ thống Tìm kiếm và Soạn thảo Pháp lý

## Giới thiệu

Dự án này là một ứng dụng web pháp lý được xây dựng để hỗ trợ tư vấn và soạn thảo bản án hành chính tại Việt Nam. Hệ thống sử dụng các kỹ thuật tìm kiếm hybrid (FAISS và BM25) để truy vấn các bản án và án lệ tương đồng, kết hợp với mô hình ngôn ngữ lớn (LLM) qua Gemini API để tạo ra câu trả lời pháp lý chi tiết và bản án theo chuẩn **Mẫu số 22-HC** (Nghị quyết số 02/2017/NQ-HĐTP).

Mục tiêu của dự án là cung cấp công cụ hỗ trợ pháp lý nhanh chóng, chính xác, phù hợp với các quy định pháp luật Việt Nam, đồng thời đảm bảo hiệu suất và khả năng mở rộng.

## Công nghệ sử dụng

- **Backend**: Flask (Python) - Framework web nhẹ, dễ tùy chỉnh.
- **Tìm kiếm vector**: FAISS - Tìm kiếm gần đúng hiệu quả cho dữ liệu lớn.
- **Tìm kiếm từ khóa**: BM25 (Okapi) - Bổ sung tìm kiếm full-text.
- **Embedding**: SentenceTransformer (`hiieu/halong_embedding`) - Tạo vector biểu diễn văn bản.
- **LLM**: Gemini API (`gemini-2.0-flash-thinking-exp-01-21`) - Xử lý ngôn ngữ tự nhiên và soạn thảo.
- **Frontend**: Flask `render_template` - Phục vụ HTML tĩnh.
- **Logging**: Python `logging` - Ghi log hoạt động.
- **CORS**: Flask-CORS - Xử lý yêu cầu cross-origin.
- **Quản lý cấu hình**: `.env` - Lưu trữ biến môi trường (`GEMINI_API_KEYS`).
- **Phụ thuộc**: NLTK (punkt), NumPy, Pickle.

## Cấu trúc hệ thống

Hệ thống bao gồm các thành phần chính:
1. **API Endpoints**:
   - `/query`: Nhận câu hỏi pháp lý, trả về câu trả lời dựa trên bản án/án lệ tương đồng.
   - `/draft_judgment`: Soạn thảo bản án hành chính sơ thẩm theo mẫu.
   - `/`: Trang chủ hiển thị thời gian hiện tại.
2. **Tìm kiếm Hybrid**:
   - Kết hợp FAISS (tìm kiếm vector) và BM25 (tìm kiếm từ khóa) để tìm bản án/án lệ liên quan.
   - Dữ liệu được lưu trữ trong các file FAISS (`faiss_index`, `faiss_index_anle.index`) và metadata (`faiss_metadata.pkl`, `summarized_faiss_metadata.pkl`, `metadata_anle.pkl`).
3. **LLM Integration**: Sử dụng Gemini API để sinh câu trả lời và bản án dựa trên prompt chi tiết.
4. **Frontend**: HTML tĩnh (`index.html`) hiển thị giao diện cơ bản.

## Hướng dẫn cài đặt

### Yêu cầu
- Python 3.8+
- pip
- Docker (khuyến nghị cho triển khai)
- API Key cho Gemini API (lưu trong `.env`)

### Cài đặt môi trường
1. **Clone repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Tạo môi trường ảo**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```
   Nội dung `requirements.txt`:
   ```
   flask
   flask-cors
   faiss-cpu
   numpy
   sentence-transformers
   python-dotenv
   nltk
   rank-bm25
   ```

4. **Tải dữ liệu NLTK**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

5. **Cấu hình biến môi trường**:
   - Tạo file `.env` trong thư mục gốc:
     ```
     GEMINI_API_KEYS=<your_gemini_api_key>
     ```
   - Đảm bảo file `config.yaml` tồn tại cho `GeminiHandler`.

6. **Tải dữ liệu FAISS và metadata**:
   - Đặt các file `faiss_index`, `faiss_index_anle.index`, `faiss_metadata.pkl`, `summarized_faiss_metadata.pkl`, `metadata_anle.pkl` vào thư mục `source/`.

7. **Chạy ứng dụng**:
   ```bash
   python run.py
   ```
   Truy cập `http://localhost:5000` để kiểm tra.

### Cài đặt với Docker
1. **Tạo Dockerfile**:
   ```dockerfile
   FROM python:3.8-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "run.py"]
   ```

2. **Build và chạy**:
   ```bash
   docker build -t legal-system .
   docker run -p 5000:5000 --env-file .env legal-system
   ```

## Hướng dẫn sử dụng

1. **Truy vấn pháp lý**:
   - Gửi yêu cầu POST tới `/query` với JSON:
     ```json
     {
         "question": "Tư vấn về tranh chấp hợp đồng thuê đất"
     }
     ```
   - Kết quả trả về: Câu trả lời pháp lý, danh sách bản án/án lệ tương đồng.

2. **Soạn thảo bản án**:
   - Gửi yêu cầu POST tới `/draft_judgment` với JSON:
     ```json
     {
         "case_details": "Tranh chấp quyết định hành chính về thu hồi đất tại Hà Nội..."
     }
     ```
   - Kết quả trả về: Bản án định dạng HTML theo Mẫu số 22-HC, kèm bản án/án lệ tham khảo.

3. **Trang chủ**:
   - Truy cập `http://localhost:5000` để xem giao diện cơ bản.

## Bảo trì và khắc phục lỗi

### Bảo trì
- **Kiểm tra log**: Log được ghi qua Python `logging` (mức INFO/DEBUG). Kiểm tra file log hoặc console để phát hiện vấn đề.
- **Cập nhật dữ liệu**:
  - Định kỳ cập nhật file FAISS và metadata trong thư mục `source/`.
  - Kiểm tra tính toàn vẹn của file bằng hash (SHA256).
- **Cập nhật thư viện**:
  ```bash
  pip install --upgrade -r requirements.txt
  ```

### Khắc phục lỗi
- **Lỗi FAISS/BM25**:
  - **Triệu chứng**: Truy vấn trả về kết quả sai hoặc lỗi file.
  - **Khắc phục**: Kiểm tra file `faiss_index`/`metadata.pkl`, rebuild index nếu hỏng.
- **Lỗi Gemini API**:
  - **Triệu chứng**: Lỗi 429 (rate limit) hoặc 500.
  - **Khắc phục**: Kiểm tra `GEMINI_API_KEYS`, đảm bảo `KeyRotationStrategy` hoạt động đúng.
- **Lỗi ứng dụng**:
  - **Triệu chứng**: Crash hoặc lỗi 500.
  - **Khắc phục**: Kiểm tra log, khởi động lại Flask (`python run.py`).

## Kế hoạch mở rộng
Dựa trên tư vấn công nghệ, dự án có thể được cải tiến như sau:
1. **Backend**: Chuyển sang FastAPI để hỗ trợ bất đồng bộ và tài liệu API tự động.
2. **Cơ sở dữ liệu**: Sử dụng PostgreSQL với pgvector thay cho FAISS file-based.
3. **Tìm kiếm**: Tích hợp Elasticsearch để tăng cường tìm kiếm full-text.
4. **LLM**: Sử dụng LangChain để quản lý prompt và caching.
5. **Frontend**: Chuyển sang React để cải thiện giao diện người dùng.
6. **Cơ sở hạ tầng**: Triển khai trên AWS (ECS, RDS, OpenSearch) với CI/CD.
7. **Design Pattern**:
   - **Repository Pattern**: Tách logic truy cập dữ liệu.
   - **Strategy Pattern**: Linh hoạt chọn chiến lược tìm kiếm.
   - **Factory Pattern**: Quản lý tạo đối tượng LLM.

### Chi phí vận hành (ước tính trên AWS)
- **Tháng**: ~$176 (ECS, RDS, OpenSearch, ElastiCache, Gemini API).
- **Quý**: ~$528.
- **Năm**: ~$2,112.
- **Tối ưu hóa**: Sử dụng AWS Savings Plans, caching, Graviton instance.

## Thông tin liên hệ
- **Email**: quocdatforworkv2@gmail.com
- **GitHub**: `<repository_url>`
- **Báo lỗi**: Vui lòng tạo issue trên GitHub hoặc liên hệ qua email.
