"""
Lab #4 — Tool Layer (Milestone 2)

Lớp tool là "bàn tay" của Agent: mọi dữ liệu Agent nói ra đều phải đi qua đây.
Nguyên tắc thiết kế:
  1. Tool KHÔNG BAO GIỜ ném exception ra ngoài — lỗi được trả về dưới dạng
     payload có khoá "error" để Agent đọc được, suy luận và tự phục hồi.
  2. Mỗi lỗi gắn thêm cờ "recoverable" để Agent biết nên thử lại (I/O chập chờn)
     hay dừng hẳn (tham số sai).
  3. Ghi file theo kiểu atomic (ghi tạm rồi thay thế) để dữ liệu không hỏng
     giữa chừng nếu tiến trình bị ngắt.
"""

import json
import os
import re
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

VALID_CATEGORIES = ("xe_dien", "du_lich")

# Người dùng gõ "xe điện", LLM có thể sinh ra "xe dien" hoặc "EV".
# Chuẩn hoá ngay tại biên giúp tool không vỡ chỉ vì khác dấu hay gạch dưới.
_CATEGORY_ALIASES = {
    "xe_dien": "xe_dien", "xe dien": "xe_dien", "xe điện": "xe_dien",
    "xedien": "xe_dien", "ev": "xe_dien", "oto_dien": "xe_dien",
    "du_lich": "du_lich", "du lich": "du_lich", "du lịch": "du_lich",
    "dulich": "du_lich", "travel": "du_lich", "resort": "du_lich",
}

_PRIORITY_LEVELS = ("low", "medium", "high")

# Phân loại ticket theo chủ đề, khớp với trường "category" đã có sẵn trong
# support_tickets.json. Thứ tự kiểm tra có ý nghĩa: "bảo hành" đặc thù hơn
# "lỗi", nên phải đứng trước.
_TICKET_CATEGORY_RULES = [
    ("warranty", ("bảo hành", "bao hanh", "warranty", "đổi trả", "hoàn tiền")),
    ("booking", ("đặt phòng", "đặt chỗ", "booking", "huỷ phòng", "hủy phòng",
                 "phòng", "resort", "vinpearl", "khách sạn", "tour")),
    ("technical", ("lỗi", "hỏng", "sự cố", "trục trặc", "không hoạt động",
                   "adas", "phần mềm", "pin", "sạc", "chết máy")),
]


def format_vnd(amount: Any) -> str:
    """Định dạng tiền theo chuẩn Việt Nam: 548000000 -> '548.000.000 VNĐ'."""
    try:
        return f"{int(amount):,}".replace(",", ".") + " VNĐ"
    except (TypeError, ValueError):
        return "Liên hệ"


def _normalize_category(category: str) -> Optional[str]:
    """Đưa mọi biến thể cách viết về đúng một trong hai giá trị hợp lệ."""
    if not isinstance(category, str):
        return None
    return _CATEGORY_ALIASES.get(category.strip().lower())


def _read_json_list(path: str) -> Any:
    """
    Đọc một file JSON dạng array.

    Trả về list nếu thành công, hoặc dict lỗi nếu thất bại — nơi gọi dùng
    isinstance() để phân nhánh thay vì phải bắt exception.
    """
    if not os.path.exists(path):
        return {"error": f"Không tìm thấy file dữ liệu: {os.path.basename(path)}",
                "recoverable": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return {"error": f"File {os.path.basename(path)} sai định dạng JSON: {exc}",
                "recoverable": False}
    except OSError as exc:
        # Lỗi I/O thường chỉ tạm thời (file đang bị khoá) -> cho phép thử lại.
        return {"error": f"Không đọc được {os.path.basename(path)}: {exc}",
                "recoverable": True}

    if not isinstance(data, list):
        return {"error": f"File {os.path.basename(path)} phải chứa một JSON array.",
                "recoverable": False}
    return data


def _atomic_write_json(path: str, payload: Any) -> Optional[Dict[str, Any]]:
    """
    Ghi JSON an toàn: ghi ra file tạm cùng thư mục rồi os.replace().

    Nhờ vậy support_tickets.json không bao giờ ở trạng thái ghi dở — hoặc là
    nội dung cũ, hoặc là nội dung mới hoàn chỉnh. Trả về None nếu thành công.
    """
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
        return None
    except OSError as exc:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return {"error": f"Không ghi được {os.path.basename(path)}: {exc}",
                "recoverable": True}


def _next_ticket_id(existing_tickets: List[Dict[str, Any]]) -> str:
    """
    Sinh ticket_id dạng TK-YYYYMMDD-NNN.

    Số thứ tự lấy từ (số lớn nhất đang có + 1) chứ không phải len(list).
    Lý do: nếu một ticket cũ bị xoá, len() tụt xuống và sinh ra ID trùng với
    ticket đã tồn tại — một lỗi im lặng rất khó truy vết về sau.
    """
    max_seq = 0
    for ticket in existing_tickets:
        if not isinstance(ticket, dict):
            continue
        match = re.search(r"-(\d+)$", str(ticket.get("ticket_id", "")))
        if match:
            max_seq = max(max_seq, int(match.group(1)))
    today = datetime.now().strftime("%Y%m%d")
    return f"TK-{today}-{max_seq + 1:03d}"


def _classify_ticket(issue_description: str) -> str:
    """Gán nhãn chủ đề để đội hỗ trợ định tuyến ticket về đúng bộ phận."""
    text = (issue_description or "").lower()
    for label, keywords in _TICKET_CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return label
    return "general"


# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp, sắp xếp theo giá tăng dần.
        Khi có lỗi: list một phần tử chứa khoá "error".
    """
    normalized = _normalize_category(category)
    if normalized is None:
        return [{
            "error": f"Danh mục '{category}' không hợp lệ. "
                     f"Chỉ chấp nhận: {', '.join(VALID_CATEGORIES)}.",
            "recoverable": False,
        }]

    if max_price is None:
        max_price = 999999999999
    try:
        max_price = int(max_price)
    except (TypeError, ValueError):
        return [{"error": f"max_price phải là số nguyên, nhận được: {max_price!r}",
                 "recoverable": False}]
    if max_price < 0:
        return [{"error": "max_price không được là số âm.", "recoverable": False}]

    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    products = _read_json_list(catalog_file)
    if isinstance(products, dict):          # _read_json_list đã trả về lỗi
        return [products]

    results = [
        p for p in products
        if isinstance(p, dict)
        and _normalize_category(p.get("category", "")) == normalized
        and isinstance(p.get("price_vnd"), (int, float))
        and p["price_vnd"] <= max_price
    ]
    # Sắp xếp theo giá tăng dần: khách hàng gần như luôn muốn xem lựa chọn
    # vừa túi tiền trước khi cân nhắc bản cao cấp hơn.
    results.sort(key=lambda p: p["price_vnd"])
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo (ticket_id, status, ...), hoặc dict chứa khoá
        "error" nếu không ghi được.
    """
    customer_name = (customer_name or "").strip()
    issue_description = (issue_description or "").strip()
    if not customer_name:
        return {"error": "Thiếu customer_name — không thể tạo ticket.",
                "recoverable": False}
    if not issue_description:
        return {"error": "Thiếu issue_description — không thể tạo ticket.",
                "recoverable": False}

    priority = str(priority or "medium").strip().lower()
    if priority not in _PRIORITY_LEVELS:
        priority = "medium"     # Ưu tiên lạ thì hạ về mặc định, không chặn khách.

    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # BẮT BUỘC đọc trước khi ghi. Mở thẳng mode "w" sẽ xoá sạch lịch sử ticket.
    existing_tickets = _read_json_list(tickets_file)
    if isinstance(existing_tickets, dict):
        if existing_tickets.get("recoverable"):
            return existing_tickets                 # I/O lỗi -> để Agent thử lại
        existing_tickets = []                       # File chưa có hoặc hỏng -> tạo mới

    ticket_id = _next_ticket_id(existing_tickets)
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": _classify_ticket(issue_description),
    }

    existing_tickets.append(new_ticket)
    write_error = _atomic_write_json(tickets_file, existing_tickets)
    if write_error:
        return write_error

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "category": new_ticket["category"],
        "created_at": new_ticket["created_at"],
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả tool cho LLM
#
# Đây là toàn bộ những gì model "nhìn thấy" về tool. Mô tả càng rõ thì model
# càng ít gọi sai tham số, nên phần description được viết như hướng dẫn sử dụng
# cho người dùng, không phải chú thích cho lập trình viên.
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu danh mục sản phẩm và dịch vụ chính hãng của Vingroup "
            "(xe điện VinFast, kỳ nghỉ Vinpearl) theo danh mục và mức giá tối đa. "
            "PHẢI gọi tool này mỗi khi khách hỏi về mẫu xe, gói nghỉ dưỡng, giá bán "
            "hoặc tính năng — tuyệt đối không trả lời bằng trí nhớ."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Danh mục cần tra cứu. 'xe_dien' cho ô tô điện VinFast, "
                        "'du_lich' cho gói nghỉ dưỡng Vinpearl."
                    ),
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": (
                        "Giá tối đa tính bằng VNĐ (đơn vị đồng, không phải triệu). "
                        "Ví dụ: '600 triệu' -> 600000000. Bỏ trống nếu khách không "
                        "nêu ràng buộc về giá."
                    ),
                    "minimum": 0,
                },
            },
            "required": ["category"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo phiếu hỗ trợ (ticket) trong hệ thống CSKH Vingroup khi khách báo "
            "lỗi, khiếu nại, phản ánh chất lượng hoặc yêu cầu bộ phận kỹ thuật xử lý. "
            "Chỉ gọi khi đã có tên khách hàng và mô tả vấn đề cụ thể."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng, trích từ lời khách nói.",
                    "minLength": 1,
                },
                "issue_description": {
                    "type": "string",
                    "description": (
                        "Mô tả vấn đề bằng một câu rõ ràng, giữ nguyên chi tiết kỹ "
                        "thuật khách cung cấp (tên xe, mã lỗi, địa điểm...)."
                    ),
                    "minLength": 1,
                },
                "priority": {
                    "type": "string",
                    "description": (
                        "Mức độ ưu tiên. 'high' khi có từ khoá khẩn/nghiêm trọng hoặc "
                        "ảnh hưởng an toàn; 'low' khi khách nói không gấp; "
                        "còn lại dùng 'medium'."
                    ),
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["customer_name", "issue_description"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
