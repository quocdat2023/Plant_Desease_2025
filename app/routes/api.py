from flask import Blueprint, request, jsonify, redirect, url_for, session, render_template
from datetime import datetime
from langchain.memory import ConversationBufferMemory
import json
from typing import List, Dict
import re
import bcrypt
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Giả định các service/repository đã được định nghĩa
from ..core.services.query_service import QueryService
from ..core.services.gemini_service import GeminiService
from ..core.repositories.index_repository import IndexRepository

api_bp = Blueprint('api', __name__)

# Kết nối MongoDB
mongo_client = MongoClient('mongodb://localhost:27017/')  # Thay bằng URI của MongoDB Atlas nếu cần
db = mongo_client['plant_disease_db']
users_collection = db['users']
users_collection.create_index('email', unique=True)

# Khởi tạo các service/repository
index_repo = IndexRepository()
query_service = QueryService(index_repo)
gemini_service = GeminiService()

# Khởi tạo ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_message_limit=10,
    max_token_limit=1000
)

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
        role = getattr(m, "type", None) or m.get("role", "User")
        content = getattr(m, "content", None) or m.get("content", "")
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)

@api_bp.route("/register", methods=["GET", "POST"])
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

@api_bp.route("/login", methods=["GET", "POST"])
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
            return redirect(url_for("home.home"))
        return jsonify({"message": "Login successful!", "user": session["user"]}), 200
    else:
        if request.form:
            return render_template("login.html", error="Invalid email or password!")
        return jsonify({"error": "Invalid email or password!"}), 401

@api_bp.route("/logout", methods=["GET", "POST"])
def logout():
    session.pop("user", None)
    if request.method == "GET":
        return redirect(url_for("home.home"))
    return jsonify({"message": "Logged out successfully"}), 200

@api_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Invalid question!"}), 400

    # Query dữ liệu tham khảo
    results = query_service.query(question, k=5, doc_type="banan", strategy="hybrid")

    top_pdf_docs = [
        {"source": r.metadata["source"], "text": r.text, "distance": r.distance, **r.__dict__}
        for r in results if r.distance is not None and r.distance != 0
    ]

    chat_history_str = format_chat_history(memory)

    # Prompt for main answer
    main_prompt = f"""
Dưới đây là lịch sử hội thoại trước đó:
{chat_history_str}

Bạn là chuyên gia nông nghiệp với hơn 30 năm kinh nghiệm trong lĩnh vực bệnh cây trồng tại Việt Nam. Bạn sẽ phân tích câu hỏi về bệnh nông nghiệp theo các bước chi tiết dưới đây để cung cấp câu trả lời chính xác, rõ ràng, dễ áp dụng, trích dẫn thông tin từ dữ liệu tham khảo nếu có.

**Câu hỏi:**  
{question}

**Thông tin tham khảo:**  
{top_pdf_docs if top_pdf_docs else "Không tìm thấy thông tin từ PDF. Phân tích dựa trên dữ liệu bệnh và kiến thức nông nghiệp."}


**Hướng dẫn trả lời chi tiết:**
** Chú ý nếu xác định đầu vào là câu hỏi thì tập trung vào trả lời câu hỏi liên quan. ngược lại nếu đầu vào là  tên bệnh thì trả lời theo các bước sau: **

1. **Tên bệnh:**  
   - Xác định và nêu rõ tên bệnh liên quan đến câu hỏi (nếu có trong dữ liệu tham khảo).
   - Nếu không có dữ liệu cụ thể, đề xuất bệnh có thể liên quan dựa trên triệu chứng hoặc cây trồng được nhắc đến.

2. **Triệu chứng:**  
   - Mô tả rõ các triệu chứng của bệnh, dựa trên dữ liệu tham khảo hoặc kiến thức chung.
   - Nêu các dấu hiệu nhận biết trên cây trồng (lá, thân, quả, v.v.).

3. **Cách điều trị:**  
   - Đề xuất phương pháp điều trị cụ thể, bao gồm thuốc trừ sâu, biện pháp sinh học, hoặc kỹ thuật canh tác.
   - Trích dẫn từ dữ liệu tham khảo nếu có (thuốc, liều lượng, thời điểm phun).

4. **Bệnh liên quan:**  
   - Liệt kê các bệnh khác thường xuất hiện cùng hoặc có triệu chứng tương tự trên cùng loại cây trồng.
   - Giải thích ngắn gọn mối liên hệ giữa các bệnh này.

6. **Lưu ý quan trọng:**
   - Không được phép đề cập đến án lệ, bản án, hoặc các vấn đề pháp lý.
   - Không cần giới thiệu bản thân, không đề cập đến kinh nghiệm tư vấn.
   - Không cần đề cập đến nguồn tài liệu tham khảo.
   - Tập trung trả lời câu hỏi của nông dân.
   - Trả lời ngắn gọn, súc tích, đúng trọng tâm.
   - Nêu các lưu ý khi áp dụng phương pháp điều trị (thời điểm, an toàn lao động, môi trường).
   - Đảm bảo trả lời ngắn gọn, súc tích, đúng trọng tâm.
   - Không sử dụng từ "giả sử" hoặc "ví dụ".
   - Trình bày rõ ràng, sử dụng định dạng danh sách (-), in đậm (**text**) cho các tiêu đề và điểm quan trọng.

**Định dạng trả lời:**
- **Tên bệnh**: [Tên bệnh]
- **Triệu chứng**: [Mô tả triệu chứng]
- **Cách điều trị**: [Phương pháp điều trị]
- **Bệnh liên quan**: [Danh sách bệnh liên quan]
- **Lưu ý quan trọng**: [Các lưu ý]
"""
    # Generate the main answer
    answer = gemini_service.generate_content(main_prompt)

    # Prompt cho câu hỏi liên quan
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

    # Parse related_questions to ensure it's a valid JSON list (assuming gemini_service returns a JSON string)
    try:
        related_questions = gemini_service.generate_content(related_questions_prompt)
        # Preprocess the questions (handles both string and list inputs)
        related_questions = preprocess_related_questions(related_questions)
    except (json.JSONDecodeError, ValueError, Exception) as e:        # Fallback to default questions if generation fails
        related_questions = [
            {"question": "Cách nhận biết sớm các bệnh phổ biến trên cây cà chua?"},
            {"question": "Những loại thuốc nào an toàn để trị bệnh trên cây lúa?"},
            {"question": "Bệnh nhện đỏ trên cây trồng có thể phòng ngừa như thế nào?"},
            {"question": "Các bệnh nào thường xuất hiện cùng với bệnh nấm trên cây?"},
            {"question": "Chế độ tưới nước ảnh hưởng thế nào đến bệnh cây trồng?"}
        ]

    # Save context to memory
    memory.save_context({"question": question}, {"answer": answer})

    # Return JSON response with related questions included
    return jsonify({
        "final_response": answer,
        "top_banan_documents": top_pdf_docs,
        "chat_history": chat_history_str,
        "related_questions": related_questions
    })


@api_bp.route("/query_related", methods=["POST"])
def query_related():
    if "user" not in session:
        return jsonify({"error": "Please log in first!"}), 401

    user_info = session["user"]
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Invalid question!"}), 400

    results = query_service.query(question, k=5, doc_type="banan", strategy="hybrid")
    top_pdf_docs = [
        {"source": r.metadata["source"], "text": r.text, "distance": r.distance, **r.__dict__}
        for r in results if r.distance is not None and r.distance != 0
    ]

    chat_history_str = format_chat_history(memory)

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
- Đảm bảo trả lời ngắn gọn, súc tích, đúng trọng tâm.
- Không sử dụng từ "giả sử" hoặc "ví dụ".
- Trình bày rõ ràng, sử dụng định dạng danh sách (-), in đậm (**text**) cho các tiêu đề và điểm quan trọng.
"""
    answer = gemini_service.generate_content(main_prompt)

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
        related_questions = gemini_service.generate_content(related_questions_prompt)
        related_questions = preprocess_related_questions(related_questions)
    except (json.JSONDecodeError, ValueError, Exception):
        related_questions = [
            {"question": "Cách nhận biết sớm các bệnh phổ biến trên cây cà chua?"},
            {"question": "Những loại thuốc nào an toàn để trị bệnh trên cây lúa?"},
            {"question": "Bệnh nhện đỏ trên cây trồng có thể phòng ngừa như thế nào?"},
            {"question": "Các bệnh nào thường xuất hiện cùng với bệnh nấm trên cây?"},
            {"question": "Chế độ tưới nước ảnh hưởng thế nào đến bệnh cây trồng?"}
        ]

    memory.save_context({"question": question}, {"answer": answer})

    return jsonify({
        "final_response": answer,
        "top_banan_documents": top_pdf_docs,
        "chat_history": chat_history_str,
        "related_questions": related_questions,
        "user_info": user_info
    })

