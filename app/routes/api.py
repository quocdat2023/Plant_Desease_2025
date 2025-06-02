from flask import Blueprint, request, jsonify
from datetime import datetime
from langchain.memory import ConversationBufferMemory

# Giả định các service/repository đã được định nghĩa
from ..core.services.query_service import QueryService
from ..core.services.gemini_service import GeminiService
from ..core.repositories.index_repository import IndexRepository

api_bp = Blueprint('api', __name__)

# Khởi tạo các service/repository
index_repo = IndexRepository()
query_service = QueryService(index_repo)
gemini_service = GeminiService()

# Khởi tạo ConversationBufferMemory với giới hạn 10 tin nhắn và 1000 token
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_message_limit=10,  # Giới hạn 10 tin nhắn
    max_token_limit=1000   # Giới hạn 1000 token
)

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



@api_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Invalid question!"}), 400

    # Query dữ liệu tham khảo
    banan_results = query_service.query(question, k=5, doc_type="banan", strategy="hybrid")
    banan_sum_results = query_service.query(question, k=5, doc_type="banan_sum", strategy="hybrid")

    top_banan_docs = [
        {"source": r.metadata["source"], "text": r.text, "distance": r.distance, **r.__dict__}
        for r in banan_results if r.distance is not None and r.distance != 0
    ]
    

    chat_history_str = format_chat_history(memory)

    prompt = f"""
Dưới đây là lịch sử hội thoại trước đó:
{chat_history_str}

Bạn là chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. Bạn sẽ phân tích vấn đề pháp lý theo các bước chi tiết dưới đây để cung cấp câu trả lời chặt chẽ, rõ ràng, và thuyết phục, đảm bảo đề cập đến cả bản án và án lệ khi có sẵn.

**Câu hỏi:**  
{question}

**Thông tin tham khảo (bản án tương đồng):**  
{top_banan_docs if top_banan_docs else "Không tìm thấy bản án phù hợp. Phân tích dựa trên các quy định pháp luật hiện hành và nguyên tắc pháp lý chung."}


**Hướng dẫn trả lời chi tiết:**

1. **Tổng quan về các bản án, án lệ tương đồng:**  
   - Chỉ ra rõ thông tin tham khảo từ bản án, án lệ tương đồng đã được cung cấp.
   - Nếu có thông tin chi tiết về bản án hoặc án lệ, trình bày ngắn gọn tên, bối cảnh, và nguồn gốc, làm rõ sự liên quan đến vấn đề pháp lý được đặt ra.  
   - Xác định loại tranh chấp (hợp đồng, dân sự, thương mại, v.v.) và các yếu tố pháp lý trọng tâm, nhấn mạnh tính phù hợp với câu hỏi.  
   - Nếu thông tin bản án hoặc án lệ chỉ có tên hoặc số hiệu, nêu rõ rằng thông tin chi tiết không khả dụng và chuyển sang phân tích dựa trên quy định pháp luật hiện hành.

2. **Nội dung chi tiết của từng bản án, án lệ:**  
   - Nếu có thông tin chi tiết, tóm lược các sự kiện chính, vấn đề pháp lý, và lập luận của tòa án trong từng bản án, án lệ.  
   - Phân tích các yếu tố quyết định phán quyết, bao gồm hợp đồng, nghĩa vụ bồi thường, hoặc trách nhiệm pháp lý của các bên.  
   - Nếu thiếu chi tiết, nêu rõ hạn chế và thay bằng phân tích các nguyên tắc pháp lý liên quan đến câu hỏi.

3. **Phân tích tình huống pháp lý:**  
   - Nếu có bản án hoặc án lệ, làm rõ các tình huống pháp lý trọng tâm, nêu bật yếu tố ảnh hưởng đến quyết định của tòa án.  
   - So sánh điểm tương đồng và khác biệt giữa các bản án, án lệ, đánh giá mức độ áp dụng vào câu hỏi pháp lý.  
   - Nếu không có thông tin chi tiết, phân tích tình huống dựa trên các quy định pháp luật hiện hành (ví dụ: Bộ luật Dân sự, Luật Thương mại).

4. **Lập luận pháp lý:**  
   - Phân tích chi tiết căn cứ pháp lý, viện dẫn cụ thể các điều luật từ Bộ luật Dân sự, Luật Thương mại, hoặc các văn bản pháp luật liên quan.  
   - Giải thích cách áp dụng các điều luật vào tình huống thực tế, đảm bảo dễ hiểu và minh họa rõ ràng quá trình lập luận.  
   - Nếu có bản án hoặc án lệ, liên hệ với lập luận của tòa án; nếu không, xây dựng lập luận dựa trên luật và nguyên tắc pháp lý.

5. **Kết luận từ các bản án, án lệ:**  
   - Nếu có thông tin chi tiết, tóm tắt phán quyết của từng bản án, án lệ, làm rõ lý do chúng có thể áp dụng vào tình huống của câu hỏi.  
   - Nếu thiếu chi tiết, đưa ra kết luận dựa trên phân tích pháp lý, nhấn mạnh quyền lợi, nghĩa vụ, và hậu quả pháp lý của các bên.  
   - Chỉ ra các yếu tố cần lưu ý khi áp dụng vào tình huống tương tự.

**Lưu ý quan trọng:**  
- Xác định nếu câu hỏi nhận được không thuộc về linh vực pháp lý hoặc không có thông tin bản án, án lệ phù hợp, hãy bỏ qua và trả lời Câu trả lời không nằm trong kiến thức của tôi.
- Trả lời ngắn gọn, súc tích, không dài dòng, không cần giải thích quá nhiều.
- Không dùng từ giả sử, ví dụ. 
- Bỏ phần chào hỏi, giới thiệu mình là ai. 
- Không cần nêu quy trình phân tích, không giới thiệu 30 năm kinh nghiệm.
- Nếu xác định được bản án hay án lệ không phù hợp hãy bỏ qua, không đề cập đến trong câu trả lời.
- Phân tích phải kết hợp chặt chẽ giữa lý thuyết pháp lý và thực tiễn vụ án (nếu có), đảm bảo tính chi tiết và thực tiễn.  
- Nếu thông tin bản án hoặc án lệ không đủ chi tiết, tập trung vào phân tích pháp lý dựa trên các quy định pháp luật hiện hành.  
- Trình bày rõ ràng, súc tích, sử dụng ngôn ngữ pháp lý chính xác, giúp người đọc dễ dàng áp dụng vào tình huống pháp lý thực tế.
"""    # Sửa: sử dụng memory để lấy lịch sử hội thoại
    # Sửa: sử dụng memory để lấy lịch sử hội thoại
    answer = gemini_service.generate_content(prompt)

    # Sửa: truyền đúng dict có đúng 1 key theo yêu cầu save_context
    memory.save_context({"question": question}, {"answer": answer})

    return jsonify({
        "final_response": answer,
        "top_banan_documents": top_banan_docs,
        "chat_history": chat_history_str
    })


@api_bp.route("/draft_judgment", methods=["POST"])
def draft_judgment():
    data = request.get_json(silent=True) or {}
    case_details = data.get("case_details", "").strip()
    if not case_details:
        return jsonify({"error": "Invalid case details!"}), 400

    banan_results = query_service.query(case_details, k=2, doc_type="banan", strategy="faiss")

    top_banan_docs = [{"source": r.metadata["source"], **r.__dict__} for r in banan_results]

    chat_history_str = format_chat_history(memory)

    prompt = f"""
    Dưới đây là lịch sử hội thoại trước đó:
    {chat_history_str}

    Bạn là trợ lý pháp lý thông minh, có nhiệm vụ **hỗ trợ soạn thảo bản án** theo đúng mẫu các quy định của Nghị quyết 351/2017/UBTVQH14, Tòa án nhân dân tối cao đã thể hiện cụ thể về thể thức, kỹ thuật trình bày các mẫu bản án sơ thẩm, phúc thẩm, quyết định giám đốc thẩm về hành chính, dân sự, hôn nhân và gia đình, kinh doanh, thương mại, lao động ban hành kèm theo Nghị quyết số 01/2017/NQ-HĐTP, Nghị quyết số 02/2017/NQ-HĐTP (08 mẫu được gửi kèm theo Công văn này) để các Tòa án áp dụng khi ban hành các văn bản tố tụng này.
    Hãy giúp tôi **soạn bản án đầy đủ, dưới dạng  sườn bản án, không cần cụ thể, ngắn gọn, đầy đủ chữ ký các bên liên quan**, trình bày rõ ràng, theo đúng định dạng mẫu, với các phần cụ thể như sau:
    Dưới đây là **thông tin vụ án** để bạn dựa vào đó và soạn bản án:

    {case_details}

    **Yêu cầu:**  
    - Bỏ **Tuyệt vời! Tôi hiểu yêu cầu của bạn. Dựa trên thông tin bạn cung cấp và các quy định pháp luật hiện hành tại Việt Nam, tôi sẽ soạn thảo sườn Bản án sơ thẩm giải quyết vụ án Ly hôn, tranh chấp về quyền nuôi con một cách ngắn gọn, đầy đủ các phần theo yêu cầu, dưới dạng HTML5 và CSS chuẩn mực bạn cung cấp. Đây là sườn bản án: html**
    - Không cần giải thích, lập luận pháp lý sau khi soạn thaot bản án.
    - Hãy tập trung vào phần soạn thảo bản án, dưới dạng  sườn bản án, không cần cụ thể, ngắn gọn, đầy đủ chữ ký các bên liên quan, không cần giới thiệu
    - Bắt buộc bỏ phần giới thiệu dài dòng khúc đầu của hệ thống, hãy tập trung vào phần soạn thảo bản án, dưới dạng  sườn bản án, không cần cụ thể, ngắn gọn, đầy đủ chữ ký các bên liên quan
    - Dùng thông tin tôi cung cấp để soạn thảo hoàn chỉnh bản án, không tự suy diễn nội dung, tạo ra nội dung mới, bằng cách điền vào các chỗ trống, dưới dạng  sườn bản án, không cần cụ thể, ngắn gọn, đầy đủ chữ ký các bên liên quan
    - Căn cứ, viện dẫn điều luật tại Việt Nam một cách chính xác, dựa trên thông tin tình huống vụ án mà tôi cung cấp để hoàn thành soạn thảo bản án, dưới dạng  sườn bản án, không cần cụ thể, ngắn gọn, đầy đủ chữ ký các bên liên quan
    - Viết đúng định dạng bản án theo văn phong pháp lý, trang trọng, khách quan.  
    - Không viết gộp đoạn, hãy chia từng phần theo đúng nhãn tiêu đề như trong mẫu.
    - Định dạng kết quả trả về trong khổ giấy A4.
    - Bắt buộc trả về kết quả dùng html5, và định dạng css chuẩn mực, không cần css cho thẻ <body>, phù hợp với trang A4 để dễ dàng hiển thị và in ấn, không tự ý css ngoài những gì mà tôi cung cấp dưới đây. Cụ thể như sau: 
    <style>

            .header {{
                display: flex  !important;
                justify-content: space-between  !important;
                margin-bottom: 20px  !important;
            }}

            .header-left {{
                text-align: left  !important;
            }}

            .header-right {{
                text-align: center  !important;
            }}
            .header p{{
                margin: 0  !important;

            }}
            .indented {{
                margin-left: 20px  !important;
            }}

            .center-bold {{
                font-weight: bold  !important;
                text-align: center  !important;
            }}

            .italic {{
                font-style: italic  !important;
            }}
            / Print-specific styles /
                @media print {{
                    @page {{
                        size: A4  !important;
                        margin: 15mm  !important; / Standard margins for A4 printing /
                    }}
                    body {{
                        margin: 0  !important;
                        padding: 0  !important;
                        font-size: 12pt  !important; / Standard for printed documents /
                        line-height: 1.5  !important;
                        text-align: justify  !important;
                        color: #000  !important;
                    }}
                    .header{{
                    display: block  !important;
                    }}
                }}
        </style>
    """
    judgment = gemini_service.generate_content(prompt)

    memory.save_context({"case_details": case_details}, {"judgment": judgment})

    return jsonify({
            "judgment": judgment,
            "top_banan_documents": top_banan_docs,
            "chat_history": chat_history_str
        })
