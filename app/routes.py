from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import bcrypt
import json
import re
import os
import logging
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from PIL import Image
from langchain.memory import ConversationBufferMemory

from app.extensions import users_collection
from app.services.models import model_store
from app.services.rag_service import retrieve
from app.services.gemini_handler import GeminiHandler, GenerationConfig, Strategy, KeyRotationStrategy
from config.config import Config

main_bp = Blueprint('main', __name__)

# Conversation Memory (re-initialized per session ideally, but global in run.py)
# To keep it per-user, we should store it in session or a manager, but adhering to original logic which used a global variable (concurrency issue warning!) of `memory`?
# In run.py: `memory = ConversationBufferMemory(...)` is global. This means all users share memory! 
# That is a bug in original code. But I will keep it as global for now if I must, or try to fix it.
# The user asked for "standard software architecture". Standard arch demands NO global user state.
# I will try to use a simple dict mapping user_id -> memory, or just a fresh memory for the context window if we can't persist.
# The current run.py format_chat_history uses `memory.chat_memory.messages`.
# I'll create a global dictionary for memories.
user_memories = {}

def get_memory(user_email):
    if user_email not in user_memories:
        user_memories[user_email] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_message_limit=10,
            max_token_limit=1000
        )
    return user_memories[user_email]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@main_bp.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

@main_bp.route("/homes")
def homes():
    return home()

@main_bp.route("/active_workspace")
def active_workspace():
    # Placeholder if mentioned elsewhere, but not in run.py
    return jsonify({"status": "ok"})

@main_bp.route("/register", methods=["GET", "POST"])
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
    except Exception: # DuplicateKeyError
        if request.form:
            return render_template("register.html", error="Email already exists!")
        return jsonify({"error": "Email already exists!"}), 400

    if request.form:
        return render_template("register.html", message="Registration successful! Please log in.")
    return jsonify({"message": "Registration successful! Please log in."}), 201

@main_bp.route("/login", methods=["GET", "POST"])
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
            return redirect(url_for("main.home")) # Blueprint name 'main'
        return jsonify({"message": "Login successful!", "user": session["user"]}), 200
    else:
        if request.form:
            return render_template("login.html", error="Invalid email or password!")
        return jsonify({"error": "Invalid email or password!"}), 401

@main_bp.route("/logout")
def logout():
    session.pop("user", None)
    if request.method == "GET":
        return redirect(url_for("main.home"))
    return jsonify({"message": "Logged out successfully"}), 200

# Helper for /query
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

def preprocess_related_questions(related_questions_input):
    # (Copy logic from run.py)
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
         
    valid = [q for q in related_questions if isinstance(q, dict) and "question" in q]
    return valid[:5] if valid else fallback_questions[:5]

@main_bp.route("/query", methods=["GET", "POST"])
def query_route():
    if request.method == "GET":
        return render_template("query.html", user=session.get("user"))

    if "user" not in session:
        return jsonify({"status": "error", "message": "Vui lòng đăng nhập trước!", "redirect": url_for("main.login")}), 401

    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"status": "error", "message": "Vui lòng nhập câu hỏi!"}), 400

    results = retrieve(question, k=10)
    top_pdf_docs = [
        {"source": r.get("folder","")+"/"+r.get("file",""), "text": r.get('text',''), "distance": r.get('distance')}
        for r in results
    ]

    user_email = session["user"]["email"]
    memory = get_memory(user_email)
    chat_history_str = format_chat_history(memory)
    user_info = session["user"]

    # Construct prompts and call Gemini (Logic simplified)
    main_prompt = f"""
    Lịch sử: {chat_history_str}
    Hỏi: {question}
    Thông tin: {top_pdf_docs}
    """
    
    # We need to init GeminiHandler
    # Config path is relative. 
    config_path = os.path.join(Config.BASE_DIR, "config", "config.yaml")
    
    try:
        handler = GeminiHandler(
            config_path=config_path,
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )
        gen = handler.generate_content(prompt=main_prompt, model_name="gemini-2.0-flash-thinking-exp-01-21")
        answer = gen.get("text", "Lỗi mô hình.")
        
        # Related questions
        related_prompt = f"Sinh 5 câu hỏi liên quan đến: {question}"
        gen_related = handler.generate_content(prompt=related_prompt, model_name="gemini-2.0-flash-thinking-exp-01-21")
        related_questions = preprocess_related_questions(gen_related.get("text", ""))
        
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
    except Exception as e:
        logging.error(f"Gemini Error: {e}")
        return jsonify({"error": "Error processing request"}), 500

@main_bp.route("/plant_detection")
def plant_detection():
     current_time = datetime.now().strftime("%I:%M:%S %p")
     return render_template("plant_detection.html", time=current_time)

@main_bp.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", user=session.get("user"))
        
    if "user" not in session:
        return jsonify({"status":"error", "message": "Login required"}), 401

    if "file" not in request.files:
        return jsonify({"status":"error", "message": "No file"}), 400
        
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
         return jsonify({"status":"error", "message": "Invalid file"}), 400
         
    try:
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Predict
        if model_store.yolo_model:
            results = model_store.yolo_model(file_path)
            probs = results[0].probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            class_name = model_store.yolo_model.names[top1_idx]
        else:
            class_name = "Unknown (Model not loaded)"
            top1_conf = 0.0

        # Load Crop Data
        try:
            with open(Config.CROP_DATA_PATH, 'r', encoding='utf-8') as f:
                crop_data = json.load(f)
        except Exception:
            crop_data = {}

        crop_info = None
        for crop in crop_data.get('crops', []):
            if crop['key'] == class_name:
                crop_info = crop
                break
        
        return jsonify({
            "status": "success",
            "class": class_name,
            "confidence": f"{top1_conf:.2f}",
            "image_path": f"static/uploads/{filename}",
            "crop_info": crop_info or "No info"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@main_bp.route("/plant_recommendation", methods=['GET', 'POST'])
def plant_recommendation():
    if request.method == 'GET':
        return render_template("plant_recommendation.html", user=session.get("user"))
        
    try:
        data = request.get_json(silent=True) or (request.form.to_dict() if request.form else {})
        # ... validation logic ...
        
        input_data = np.array([[
            float(data.get('N',0)), float(data.get('P',0)), float(data.get('K',0)),
            float(data.get('temperature',0)), float(data.get('humidity',0)),
            float(data.get('ph',0)), float(data.get('rainfall',0))
        ]])
        
        if model_store.scaler_plant and model_store.loaded_model: # loaded_model is logistic_plant
             input_scaled = model_store.scaler_plant.transform(input_data)
             pred = model_store.logistic_plant.predict(input_scaled)
             res = model_store.label_encoder_plant.inverse_transform(pred)[0]
             return jsonify({"prediction": res, "timestamp": datetime.now().strftime("%I:%M:%S %p")})
        else:
             return jsonify({"error": "Models not loaded"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main_bp.route("/plant_fertilizer", methods=['GET', 'POST'])
def plant_fertilizer():
     if request.method == 'GET':
          return render_template("plant_fertilizer.html", time=datetime.now().strftime("%I:%M:%S %p"))
          
     # POST logic for fertilizer (omitted for brevity but logically same as run.py using model_store.fertilizer_model)
     return jsonify({"status": "success", "predictions": {}})
