from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from datetime import datetime
import faiss
from werkzeug.utils import secure_filename
from langchain.memory import ConversationBufferMemory
import json
from typing import List, Dict
import re
import bcrypt
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import sys
import pandas as pd
import numpy as np
import os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov11_source')))
from ultralytics import YOLO 
import cv2
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from gemini_handler import GeminiHandler, GenerationConfig, Strategy, KeyRotationStrategy
import logging
import pickle

app = Flask(__name__, static_folder='static')
app.secret_key = 'your-secret-key'  # Thay bằng khóa bí mật an toàn

# Kết nối MongoDB
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['plant_disease_db']
users_collection = db['users']
users_collection.create_index('email', unique=True)

# Đường dẫn đến mô hình YOLO và thư mục lưu ảnh
MODEL_PATH = "source/trained_yolo11s_cls.pt"
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# Định nghĩa đường dẫn
MODELS_PLANT= './recommender-models/'
filenameplant = 'LogisticRegresion_plant.pkl'

# Tải scaler, label encoder và mô hình
try:
    with open(MODELS_PLANT + 'scaler_plant.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    with open(MODELS_PLANT + 'label_encoder_plant.pkl', 'rb') as f:
        loaded_label_encoder = pickle.load(f)
    with open(MODELS_PLANT + filenameplant, 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy tệp - {e}")
    exit(1)



# Tải mô hình đã lưu
model_fertilizer = joblib.load('recommender-models/multioutput_xgboost_fertilizer_model.pkl')
print("Model loaded successfully")

# Tải các LabelEncoder
label_encoders_fertilizer = {}
for column in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    label_encoders_fertilizer[column] = joblib.load(f'recommender-models/label_encoder_{column}.pkl')
    print(f"LabelEncoder for {column} loaded successfully")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Tải mô hình YOLO
model = YOLO(MODEL_PATH)

# Khởi tạo ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_message_limit=10,
    max_token_limit=1000
)

# Đường dẫn đến FAISS index và dữ liệu embeddings
INDEX_PATH = "source/index_plant.faiss"
EMBEDDINGS_DATA_PATH = "source/index_plant.pkl"

# Tải FAISS index
def load_faiss_index(index_path):
    try:
        index = faiss.read_index(index_path)
        print(f"Đã tải FAISS index từ {index_path}")
        return index
    except Exception as e:
        print(f"Lỗi khi tải FAISS index: {e}")
        return None

# Tải dữ liệu embeddings
def load_embeddings_data(data_path):
    try:
        with open(data_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        logging.info(f"Đã tải dữ liệu embeddings từ {data_path}")
        return embeddings_data
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu embeddings: {e}")
        return None

# Hàm truy xuất
def retrieve(query, index, embeddings_data, k=10):
    try:
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'file': embeddings_data[idx]['file'],
                'folder': embeddings_data[idx]['folder'],
                'text_path': embeddings_data[idx]['text_path'],
                'text': embeddings_data[idx]['text'],
                'distance': float(distance)
            })
        return results
    except Exception as e:
        print(f"Lỗi trong quá trình truy xuất: {e}")
        return []

# Tải FAISS index và dữ liệu embeddings
index = load_faiss_index(INDEX_PATH)
embeddings_data = load_embeddings_data(EMBEDDINGS_DATA_PATH)
if index is None or embeddings_data is None:
    print("Không thể tải FAISS index hoặc dữ liệu embeddings. Ứng dụng không thể khởi động.")
    exit(1)

def preprocess_related_questions(related_questions_input: str | List[Dict[str, str]]) -> List[Dict[str, str]]:
    fallback_questions = [
        {"question": "Cách xử lý bệnh phổ biến trên cây trồng tại Việt Nam là gì?"},
        {"question": "Làm thế nào để nhận biết sớm các triệu chứng bệnh trên cây cà chua?"},
        {"question": "Những loại thuốc trừ sâu nào được khuyến nghị cho cây lúa?"},
        {"question": "Bệnh nào thường xuất hiện cùng với bệnh nhện đỏ trên cây trồng?"},
        {"question": "Chế độ dinh dưỡng nào giúp cây trồng tăng sức đề kháng với bệnh?"}
    ]

    if isinstance(related_questions_input, str):
        cleaned_input = re.sub(r'^```json\s*|\s*```$', '', related_questions_input).strip()
        try:
            related_questions = json.loads(cleaned_input)
        except json.JSONDecodeError:
            return fallback_questions[:5]
    else:
        related_questions = related_questions_input

    if not isinstance(related_questions, list):
        return fallback_questions[:5]

    valid_questions = [
        q for q in related_questions
        if isinstance(q, dict) and "question" in q and isinstance(q["question"], str) and q["question"].strip()
    ]

    seen = set()
    unique_questions = []
    for q in valid_questions:
        question_text = q["question"].strip()
        if question_text not in seen:
            seen.add(question_text)
            unique_questions.append({"question": question_text})

    agriculture_keywords = r"(bệnh|cây trồng|triệu chứng|thuốc trừ sâu|điều trị|nông nghiệp|cà chua|lúa|nấm|nhện đỏ|phân bón)"
    filtered_questions = [
        q for q in unique_questions
        if re.search(agriculture_keywords, q["question"], re.IGNORECASE)
    ]

    if len(filtered_questions) < 5:
        remaining = 5 - len(filtered_questions)
        for fq in fallback_questions:
            if len(filtered_questions) >= 5:
                break
            if fq["question"] not in seen:
                filtered_questions.append(fq)
                seen.add(fq["question"])

    return filtered_questions[:5]

def format_chat_history(memory):
    messages = memory.chat_memory.messages
    if not messages:
        return "Không có lịch sử hội thoại trước."
    formatted = []
    for m in messages:
        role = getattr(m, "type", "user").capitalize()
        content = getattr(m, "content", "") or ""
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)

# @app.route("/")
# def home():
#     return render_template("home.html", user=session.get("user"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    
    data = request.form if request.form else request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    name = data.get("name", "").strip()

    if not email or not password or not name:
        if request.form:
            return render_template("register.html", error="Email, password, and name are required!")
        return jsonify({"error": "Email, password, and name are required!"}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        users_collection.insert_one({
            "email": email,
            "password": hashed_password,
            "name": name,
            "created_at": datetime.utcnow()
        })
    except DuplicateKeyError:
        if request.form:
            return render_template("register.html", error="Email already exists!")
        return jsonify({"error": "Email already exists!"}), 400

    if request.form:
        return render_template("register.html", message="Registration successful! Please log in.")
    return jsonify({"message": "Registration successful! Please log in."}), 201


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    
    data = request.form if request.form else request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        if request.form:
            return render_template("login.html", error="Email and password are required!")
        return jsonify({"error": "Email and password are required!"}), 400

    user = users_collection.find_one({"email": email})
    if not user:
        if request.form:
            return render_template("login.html", error="Invalid email or password!")
        return jsonify({"error": "Invalid email or password!"}), 401

    if bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        session["user"] = {"email": user["email"], "name": user["name"]}
        if request.form:
            return redirect(url_for("home"))
        return jsonify({"message": "Login successful!", "user": session["user"]}), 200
    else:
        if request.form:
            return render_template("login.html", error="Invalid email or password!")
        return jsonify({"error": "Invalid email or password!"}), 401

@app.route("/logout")
def logout():
    session.pop("user", None)
    if request.method == "GET":
        return redirect(url_for("home"))
    return jsonify({"message": "Logged out successfully"}), 200

@app.route("/query", methods=["GET", "POST"])
def query():
    if request.method == "GET":
        return render_template("query.html", user=session.get("user"))

    if "user" not in session:
        return jsonify({"status": "error", "message": "Vui lòng đăng nhập trước!", "redirect": url_for("login")}), 401

    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"status": "error", "message": "Vui lòng nhập câu hỏi!"}), 400

    results = query(question, index, embeddings_data, k=10)
    top_pdf_docs = [
        {"source": r.metadata["source"], "text": r.text, "distance": r.distance, **r.__dict__}
        for r in results if r.distance is not None and r.distance != 0
    ]

    chat_history_str = format_chat_history(memory)
    user_info = session["user"]

    main_prompt = f"""
Dưới đây là lịch sử hội thoại trước đó:
{chat_history_str}

Bạn là chuyên gia nông nghiệp với hơn 30 năm kinh nghiệm. 
Người dùng: {user_info.get('name', 'Anonymous')} (Email: {user_info.get('email', 'N/A')})
**Câu hỏi:**  
{question}

**Thông tin tham khảo (từ PDF):**  
{top_pdf_docs if top_pdf_docs else "Không có thông tin từ PDF. Phân tích dựa trên kiến thức nông nghiệp."}

Trả lời cần:  
- Tập trung trả lời câu hỏi của nông dân.
- Đưa ra nguyên nhân gây bệnh.
- Đề xuất phương pháp điều trị/phòng ngừa hiệu quả.

**Lưu ý quan trọng:**
- Không cần giới thiệu bản thân, không đề cập đến kinh nghiệm tư vấn.
- Trả lời ngắn gọn, súc tích, đúng trọng tâm.
- Nêu các lưu ý khi áp dụng phương pháp điều trị (thời điểm, an toàn lao động, môi trường).
- Không sử dụng từ "giả sử" hoặc "ví dụ".
- Trình bày rõ ràng, sử dụng định dạng danh sách (-), in đậm (**text**) cho các tiêu đề và điểm quan trọng.
"""
    try:
        handler = GeminiHandler(
            config_path="config.yaml",
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )
        gen = handler.generate_content(
            prompt=main_prompt,
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            return_stats=False
        )
        answer = gen.get("text", "Không có phản hồi từ mô hình.")
    except Exception as e:
        logging.error(f"Lỗi khi gọi Gemini cho câu hỏi chính: {e}")
        return jsonify({
            'error': 'Lỗi xử lý câu hỏi, vui lòng thử lại sau',
            'error_code': 'GEMINI_ERROR',
            'upgrade_url': 'https://legal.loca.lt/'
        }), 500

    related_questions_prompt = f"""
Bạn là chuyên gia nông nghiệp Việt Nam. Dựa trên câu hỏi về bệnh cây trồng được cung cấp, hãy sinh ra 5 câu hỏi liên quan, đảm bảo các câu hỏi:

- Liên quan chặt chẽ đến chủ đề bệnh cây trồng trong câu hỏi gốc.
- Phù hợp với nông nghiệp Việt Nam hiện hành.
- Ngắn gọn, rõ ràng, và mang tính ứng dụng thực tế.
- Tập trung vào tên bệnh, triệu chứng, cách điều trị, hoặc bệnh liên quan.
- Được trình bày dưới dạng danh sách JSON, mỗi câu hỏi là một đối tượng với key `question`.

**Câu hỏi gốc:**  
{question}

**Hướng dẫn thêm:**
- Nếu câu hỏi gốc đề cập đến một cây trồng cụ thể (ví dụ: cà chua, lúa), sinh ra các câu hỏi liên quan đến cây đó.
- Nếu câu hỏi không rõ cây trồng, sinh ra các câu hỏi liên quan đến bệnh phổ biến trong nông nghiệp Việt Nam.
- Không sử dụng từ "giả sử" hoặc "ví dụ".
- Không lặp lại câu hỏi gốc.
- Đảm bảo các câu hỏi không trùng lặp nội dung.

**Định dạng đầu ra (JSON):**  
[
  {{"question": "Câu hỏi 1"}},
  {{"question": "Câu hỏi 2"}},
  {{"question": "Câu hỏi 3"}},
  {{"question": "Câu hỏi 4"}},
  {{"question": "Câu hỏi 5"}}
]
"""
    try:
        handler = GeminiHandler(
            config_path="config.yaml",
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )
        gen = handler.generate_content(
            prompt=related_questions_prompt,
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            return_stats=False
        )
        related_questions = gen.get("text", "Không có phản hồi từ mô hình.")
    except Exception as e:
        logging.error(f"Lỗi khi gọi Gemini cho câu hỏi liên quan: {e}")
        return jsonify({
            'error': 'Lỗi xử lý câu hỏi, vui lòng thử lại sau',
            'error_code': 'GEMINI_ERROR',
            'upgrade_url': 'https://legal.loca.lt/'
        }), 500
    related_questions = preprocess_related_questions(related_questions)

    memory.save_context({"question": question}, {"answer": answer})

    return render_template(
        "query_result.html",
        user=session.get("user"),
        question=question,
        answer=answer,
        top_pdf_docs=top_pdf_docs,
        chat_history=chat_history_str,
        related_questions=related_questions
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Handle GET request: Render the predict page
    if request.method == "GET":
        return render_template("predict.html", user=session.get("user"))

    # Ensure user is authenticated
    if "user" not in session:
        return jsonify({
            "status": "error",
            "message": "Vui lòng đăng nhập trước!",
            "redirect": url_for("login")
        }), 401

    # Check if a file is included in the request
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Vui lòng chọn file ảnh!"
        }), 400

    file = request.files["file"]
    
    # Validate file selection and extension
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Vui lòng chọn file ảnh!"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Định dạng file không được hỗ trợ! Chỉ chấp nhận PNG, JPG, JPEG, GIF."
        }), 400

    try:
        # Verify image integrity
        image = Image.open(file)
        image.verify()  # Check if file is a valid image
        file.seek(0)  # Reset file pointer after verification
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"File tải lên không phải ảnh hợp lệ: {str(e)}"
        }), 400

    # Securely save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi lưu file: {str(e)}"
        }), 500

    # Perform model prediction
    try:
        results = model(file_path)
        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = model.names[top1_idx]
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi xử lý dự đoán: {str(e)}"
        }), 500

    # Construct relative image path for frontend
    relative_image_path = f"static/uploads/{filename}"
    if not os.path.exists(relative_image_path):
        return jsonify({
            "status": "error",
            "message": "Không tìm thấy ảnh đã tải lên!"
        }), 404
    # Load crop data
    json_file_path = "source/crop_data.json"
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            crop_data = json.load(f)
    except FileNotFoundError:
        return jsonify({
            "status": "error",
            "message": "Không tìm thấy file dữ liệu cây trồng!"
        }), 500
    except json.JSONDecodeError:
        return jsonify({
            "status": "error",
            "message": "File dữ liệu cây trồng không hợp lệ!"
        }), 500

    # Find crop information
    crop_info = None
    for crop in crop_data.get('crops', []):
        if crop['key'] == class_name:
            crop_info = {
                "scientific_name": crop.get('scientific_name', 'N/A'),
                "vietnamese_name": crop.get('vietnamese_name', 'N/A'),
                "disease": crop.get('disease', 'N/A'),
                "irrigation_schedule": crop.get('irrigation_schedule', 'N/A'),
                "fertilizer_dosage": crop.get('fertilizer_dosage', 'N/A'),
                "disease_treatment_prevention": crop.get('disease_treatment_prevention', 'N/A')
            }
            break

    if not crop_info:
        crop_info = {"message": "Không tìm thấy thông tin chi tiết cho bệnh cây trồng này"}

    # Return prediction results and crop information
    return jsonify({
        "status": "success",
        "class": class_name,
        "confidence": f"{top1_conf:.2f}",
        "image_path": relative_image_path,
        "crop_info": crop_info
    }), 200

@app.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)


@app.route("/homes")
def homes():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

@app.route("/plant_detection")
def plant_detection():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("plant_detection.html", time=current_time)


@app.route("/plant_recommendation", methods=['GET', 'POST'])
def plant_recommendation():
    if request.method == 'GET':
        return render_template("plant_recommendation.html", user=session.get("user"))   
    else:
        try:
            # Try to get JSON data
            data = request.get_json(silent=True)
            if not data:
                # Fallback to form data if JSON is not provided
                if request.form:
                    data = request.form.to_dict()
                else:
                    return jsonify({"error": "No input data provided (expected JSON or form data)"}), 400

            # Extract and validate input parameters
            required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing field: {field}"}), 400
                try:
                    data[field] = float(data[field])
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value for {field}: must be a number"}), 400

            # Prepare input data for prediction
            input_data = np.array([[
                data['N'],
                data['P'],
                data['K'],
                data['temperature'],
                data['humidity'],
                data['ph'],
                data['rainfall']
            ]])

            # Scale the input data
            input_data_scaled = loaded_scaler.transform(input_data)

            # Make prediction
            prediction = loaded_model.predict(input_data_scaled)
            predicted_crop = loaded_label_encoder.inverse_transform(prediction)[0]

            # Return successful response
            return jsonify({
                "prediction": predicted_crop,
                "timestamp": datetime.now().strftime("%I:%M:%S %p")
            }), 200

        except Exception as e:
            # Log the error for debugging
            print(f"Error in plant_recommendation: {str(e)}")
            return jsonify({"error": f"Server error: {str(e)}"}), 500
        


@app.route("/plant_fertilizer", methods=['GET', 'POST'])
def plant_fertilizer():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu JSON từ yêu cầu
            data = request.get_json()

            # Kiểm tra dữ liệu đầu vào
            required_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
            if not all(feature in data for feature in required_features):
                return jsonify({
                    'error': 'Missing required features. Expected: ' + ', '.join(required_features)
                }), 400

            # Tạo DataFrame từ dữ liệu đầu vào
            new_data = pd.DataFrame({
                'Temperature': [float(data['Temperature'])],
                'Humidity': [float(data['Humidity'])],
                'Moisture': [float(data['Moisture'])],
                'Nitrogen': [float(data['Nitrogen'])],
                'Potassium': [float(data['Potassium'])],
                'Phosphorous': [float(data['Phosphorous'])]
            })

            # Dự đoán trên dữ liệu mới
            predictions = model_fertilizer.predict(new_data)

            # Giải mã kết quả dự đoán về giá trị gốc
            result = {}
            for i, column in enumerate(['Soil Type', 'Crop Type', 'Fertilizer Name']):
                predicted_label = label_encoders_fertilizer[column].inverse_transform([predictions[0][i]])[0]
                result[column] = predicted_label

            # Trả về kết quả dưới dạng JSON
            return jsonify({
                'status': 'success',
                'predictions': result
            }), 200

        except ValueError as ve:
            return jsonify({
                'error': f'Invalid input data: {str(ve)}'
            }), 400
        except Exception as e:
            return jsonify({
                'error': f'Prediction error: {str(e)}'
            }), 500
    else:
        # Xử lý yêu cầu GET: Render form
        current_time = datetime.now().strftime("%I:%M:%S %p")
        return render_template("plant_fertilizer.html", time=current_time)

# Chạy ứng dụng Flask

if __name__ == "__main__":
    app.run(debug=True)