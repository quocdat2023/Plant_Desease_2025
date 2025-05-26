from flask import Blueprint, request, jsonify, render_template
from datetime import datetime
from ..core.services.query_service import QueryService
from ..core.services.gemini_service import GeminiService
from ..core.repositories.index_repository import IndexRepository
from langchain.memory import ConversationBufferMemory

api_bp = Blueprint('api', __name__)

index_repo = IndexRepository()
query_service = QueryService(index_repo)
gemini_service = GeminiService()

# Khởi tạo ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def format_chat_history(memory):
    messages = memory.chat_memory.messages
    if not messages:
        return "Không có lịch sử hội thoại trước."
    formatted = []
    for m in messages:
        role = getattr(m, "type", None)
        if role is None:
            role = m.get("role", "User")
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
    anle_results = query_service.query(question, k=5, doc_type="anle", strategy="hybrid")

    top_banan_docs = [
        {"source": r.metadata["source"], "text": r.text, "distance": r.distance, **r.__dict__}
        for r in banan_sum_results if r.distance is not None and r.distance != 0
    ]
    top_anle_docs = [
        {"source": r.metadata["source"], "text": r.text, "distance": r.distance, **r.__dict__}
        for r in anle_results if r.distance is not None and r.distance != 0
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

**Thông tin tham khảo (án lệ tương đồng):**  
{top_anle_docs if top_anle_docs else "Không tìm thấy án lệ phù hợp. Phân tích dựa trên các quy định pháp luật hiện hành và nguyên tắc pháp lý chung."}

**Hướng dẫn trả lời chi tiết:**
...

"""
    answer = gemini_service.generate_content(prompt)

    # Sửa: truyền đúng dict có đúng 1 key theo yêu cầu save_context
    memory.save_context({"question": question}, {"answer": answer})

    return jsonify({
        "final_response": answer,
        "top_banan_documents": top_banan_docs,
        "top_anle_documents": top_anle_docs,
        "chat_history": chat_history_str
    })


@api_bp.route("/draft_judgment", methods=["POST"])
def draft_judgment():
    data = request.get_json(silent=True) or {}
    case_details = data.get("case_details", "").strip()
    if not case_details:
        return jsonify({"error": "Invalid case details!"}), 400

    banan_results = query_service.query(case_details, k=2, doc_type="banan", strategy="faiss")
    anle_results = query_service.query(case_details, k=2, doc_type="anle", strategy="faiss")

    top_banan_docs = [{"source": r.metadata["source"], **r.__dict__} for r in banan_results]
    top_anle_docs = [{"source": r.metadata["source"], **r.__dict__} for r in anle_results]

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
        "top_anle_documents": top_anle_docs,
        "chat_history": chat_history_str
    })
