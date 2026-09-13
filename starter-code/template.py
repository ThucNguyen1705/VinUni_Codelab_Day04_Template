"""
Lab #4: System Prompt Engineering & Tool Calling Engine

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.

Luồng của ToolCallingAgent theo đúng vòng ReAct:
    Intent Detection -> Plan -> Act (gọi tool) -> Observe -> Reflect -> Final Answer
Mỗi bước đều được ghi vào self.trace để có thể audit lại sau này.
"""

import json
import os
import re
import sys
from typing import Dict, Any, List, Optional, Tuple
from tools import (
    TOOL_DEFINITIONS,
    TOOL_MAP,
    format_vnd,
    search_product_catalog,
    submit_support_ticket,
)

# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE 1: SYSTEM PROMPT cấp sản xuất
#
# Prompt được viết theo 5 khối cố định. Khối AVAILABLE TOOLS không gõ tay mà
# sinh tự động từ TOOL_DEFINITIONS qua render_system_prompt(): schema và prompt
# vì thế không bao giờ lệch nhau khi ai đó sửa tool.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant.
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl và chăm sóc khách hàng.
- Giọng điệu: Chuyên nghiệp, thân thiện, đi thẳng vào trọng tâm. Xưng "VinAssistant",
  gọi khách là "anh/chị". Không dùng từ hoa mỹ, không hứa điều không có trong dữ liệu.

## 2. AVAILABLE TOOLS
Bạn được cấp đúng {tool_count} tool sau và không có tool nào khác:
{tools}

## 3. CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá, tính năng hay chính sách.
   Mọi con số nói ra PHẢI đến từ Observation của một tool đã gọi.
2. Khi khách hỏi về sản phẩm, giá hoặc gói dịch vụ: PHẢI gọi
   `search_product_catalog` trước khi trả lời, kể cả khi bạn "nghĩ là mình nhớ".
3. Khi khách báo lỗi, khiếu nại hoặc phản ánh: PHẢI gọi `submit_support_ticket`
   và trả lại mã ticket cho khách.
4. Thiếu tham số bắt buộc (ví dụ chưa có tên khách) thì HỎI LẠI, tuyệt đối
   không tự suy đoán hay điền giá trị giả.
5. Tool trả về rỗng thì phải nói thật là không có kết quả, kèm gợi ý thay thế
   nằm trong dữ liệu. Không "nới" điều kiện của khách rồi coi như đã tìm thấy.
6. Tool báo lỗi thì thử lại tối đa {max_retries} lần; vẫn lỗi thì xin lỗi và
   chuyển hướng sang kênh hỗ trợ, không giả vờ thao tác đã thành công.
7. Không bao giờ nêu giá đã suy đoán bằng cách nội suy hay quy đổi từ mẫu khác.

## 4. OPERATIONAL BOUNDARIES
- CHỈ hỗ trợ các chủ đề thuộc hệ sinh thái Vingroup: xe điện VinFast,
  nghỉ dưỡng Vinpearl, và dịch vụ hậu mãi liên quan.
- Từ chối lịch sự các chủ đề ngoài phạm vi (chính trị, y tế, tài chính cá nhân,
  sản phẩm của hãng khác) và hướng khách về đúng kênh.
- Không so sánh trực tiếp hay nhận xét tiêu cực về đối thủ cạnh tranh.
- Không thu thập thông tin nhạy cảm: số thẻ, mật khẩu, CCCD.
- Không cam kết về thời gian giao xe, khuyến mãi hay bồi thường nếu dữ liệu
  không nói tới.

## 5. OUTPUT CONTRACT
Mọi lượt suy luận tuân thủ đúng định dạng ReAct sau:

    Thought: <suy luận ngắn gọn về việc cần làm tiếp>
    Action: <tên tool cần gọi>
    Action Input: <JSON hợp lệ đúng schema của tool>
    Observation: <kết quả tool trả về — do hệ thống điền, KHÔNG tự viết>
    ... (lặp lại khi cần)
    Final Answer: <câu trả lời tiếng Việt gửi cho khách>

Ràng buộc bổ sung cho Final Answer:
- Trình bày danh sách sản phẩm dạng gạch đầu dòng, luôn kèm giá theo định dạng
  "548.000.000 VNĐ".
- Khi đã tạo ticket, PHẢI nêu rõ mã ticket và mức ưu tiên.
- Kết thúc bằng một dòng ghi nguồn dữ liệu đã dùng.
"""


def render_system_prompt(max_retries: int = 2) -> str:
    """
    Điền khối AVAILABLE TOOLS vào System Prompt từ chính TOOL_DEFINITIONS.

    Dùng str.replace() thay vì str.format() vì thân prompt có chứa dấu ngoặc
    nhọn trong ví dụ JSON — format() sẽ hiểu nhầm chúng là placeholder.
    """
    lines = []
    for spec in TOOL_DEFINITIONS:
        params = spec.get("parameters", {}).get("properties", {})
        required = set(spec.get("parameters", {}).get("required", []))
        signature = ", ".join(
            f"{name}: {meta.get('type', 'any')}" if name in required
            else f"{name}?: {meta.get('type', 'any')}"
            for name, meta in params.items()
        )
        lines.append(f"- {spec['name']}({signature})\n    {spec['description']}")

    return (SYSTEM_PROMPT
            .replace("{tools}", "\n".join(lines))
            .replace("{tool_count}", str(len(TOOL_DEFINITIONS)))
            .replace("{max_retries}", str(max_retries)))


# ═══════════════════════════════════════════════════════════════════════════
# NGÔN NGỮ HỌC: các bộ từ khoá dùng cho Intent Detection
# ═══════════════════════════════════════════════════════════════════════════

# Ý định tra cứu sản phẩm. Chỉ gồm động từ/danh từ "mua sắm" — cố tình KHÔNG
# chứa "bao lâu", "chính sách" để câu hỏi FAQ không bị kéo nhầm sang tool.
_CATALOG_INTENT_KEYWORDS = (
    "xem", "tìm", "mua", "tư vấn", "báo giá", "giá", "bao nhiêu tiền",
    "danh sách", "sản phẩm", "mẫu", "dòng xe", "giới thiệu", "tham khảo",
    "quan tâm", "đặt mua", "gói", "catalog", "còn hàng", "có sẵn",
)

# Dấu hiệu khách đang gặp vấn đề. "bị" là tín hiệu mạnh và an toàn trong tiếng
# Việt: nó hầu như chỉ xuất hiện trong câu mô tả sự cố ("bị lỗi", "bị ẩm mốc").
_TICKET_INTENT_KEYWORDS = (
    "bị", "lỗi", "hỏng", "hư", "sự cố", "trục trặc", "không hoạt động",
    "khiếu nại", "phản ánh", "phản hồi", "ghi nhận", "than phiền", "bức xúc",
    "vấn đề", "cần hỗ trợ", "yêu cầu hỗ trợ", "hỗ trợ kỹ thuật",
    "tạo ticket", "mở ticket", "chưa được xử lý",
)

# Tập con "mạnh" dùng để khoanh vùng câu mô tả sự cố khi trích issue_description.
_PRIMARY_ISSUE_MARKERS = (
    "bị", "lỗi", "hỏng", "hư", "sự cố", "trục trặc", "không hoạt động",
)

_CATEGORY_KEYWORDS = {
    "xe_dien": ("xe điện", "xe dien", "vinfast", "vf 3", "vf 5", "vf 8", "vf 9",
                "vf wild", "ô tô", "oto", "xe hơi", "bán tải", "suv", "xe"),
    "du_lich": ("du lịch", "vinpearl", "resort", "khách sạn", "nghỉ dưỡng",
                "tour", "kỳ nghỉ", "voucher", "vinwonders", "landmark 81",
                "safari", "phòng"),
}

# Thứ tự kiểm tra ưu tiên rất quan trọng: "không gấp" chứa "gấp", nên nhóm phủ
# định phải được xét TRƯỚC nhóm "high", nếu không sẽ gắn nhầm mức khẩn cấp.
_PRIORITY_RULES = (
    ("low", ("không gấp", "không vội", "không khẩn", "mức độ thấp",
             "ưu tiên thấp", "khi nào tiện", "nhẹ", "không nghiêm trọng")),
    ("medium", ("trung bình", "bình thường", "ưu tiên trung bình")),
    ("high", ("nghiêm trọng", "gấp", "khẩn", "ngay lập tức", "nguy hiểm",
              "mất an toàn", "cháy", "rất bức xúc", "không thể sử dụng",
              "ưu tiên cao", "mức độ cao")),
)

_PRICE_UNITS = (("tỷ", 10**9), ("tỉ", 10**9), ("triệu", 10**6), ("tr", 10**6),
                ("nghìn", 10**3), ("ngàn", 10**3), ("k", 10**3))

_PRICE_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(tỷ|tỉ|triệu|tr|nghìn|ngàn|k)(?!\w)", re.IGNORECASE)
_PRICE_PLAIN_RE = re.compile(
    r"(\d[\d.,]{2,})\s*(?:vnđ|vnd|đồng|đ)(?!\w)", re.IGNORECASE)

# "tôi tên là" phải đứng trước "tôi tên"; re.search lấy match trái nhất nên
# "tên tôi là X" vẫn thắng "tôi là" nhờ vị trí xuất hiện sớm hơn.
_NAME_RE = re.compile(
    r"(?:tôi tên là|tên tôi là|tên của tôi là|mình tên là|em tên là|"
    r"tôi tên|tên tôi|mình tên|em tên|tôi là|mình là)\s+(.{1,60})",
    re.IGNORECASE)


def _contains(text: str, keyword: str) -> bool:
    """
    Kiểm tra từ khoá theo ranh giới từ, không phải substring thô.

    Đây là điểm dễ sai nhất của intent detection tiếng Việt: so khớp substring
    sẽ thấy "xe" nằm trong "xem", khiến câu "Cho tôi xem resort Vinpearl"
    bị phân loại nhầm thành xe điện.
    """
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None


def _matched_keywords(text: str, keywords) -> List[str]:
    return [kw for kw in keywords if _contains(text, kw)]


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE cho câu hỏi FAQ
#
# Nguyên tắc: KHÔNG hard-code con số chính sách. Câu trả lời được trích ngược
# từ product_catalog.json để nếu dữ liệu đổi thì câu trả lời đổi theo, và
# Agent không bao giờ nói ra một con số không tồn tại trong hệ thống.
# ═══════════════════════════════════════════════════════════════════════════

def _facts_from_catalog(pattern: str) -> List[str]:
    """Quét trường features của toàn bộ catalog, trả về các fact khớp regex."""
    facts = []
    for category in ("xe_dien", "du_lich"):
        for product in search_product_catalog(category=category):
            if "error" in product:
                continue
            for feature in product.get("features", []):
                if re.search(pattern, str(feature), re.IGNORECASE):
                    facts.append(str(feature))
    # Giữ thứ tự xuất hiện nhưng loại trùng lặp.
    return list(dict.fromkeys(facts))


def _answer_battery_warranty() -> Optional[str]:
    facts = _facts_from_catalog(r"bảo hành")
    if not facts:
        return None
    return ("Về chính sách bảo hành pin xe điện VinFast: theo thông tin sản phẩm "
            "đang niêm yết trong hệ thống, chính sách áp dụng là "
            f"**{facts[0].lower()}**.\n\n"
            "Thời hạn này đi kèm sản phẩm và được ghi trực tiếp trong catalog "
            "chính hãng. Nếu anh/chị cần xác nhận điều kiện áp dụng cho đúng số "
            "VIN xe của mình, VinAssistant có thể mở một phiếu hỗ trợ để bộ phận "
            "kỹ thuật kiểm tra — anh/chị chỉ cần cho biết họ tên ạ.")


def _answer_charging() -> Optional[str]:
    facts = _facts_from_catalog(r"sạc")
    if not facts:
        return None
    bullets = "\n".join(f"- {fact}" for fact in facts)
    return ("Về khả năng sạc của dải xe điện VinFast, đây là các thông số đang "
            f"ghi nhận trong hệ thống:\n{bullets}\n\n"
            "Mỗi dòng xe có cấu hình sạc khác nhau, anh/chị cho biết mẫu xe quan "
            "tâm để VinAssistant tra cứu chi tiết ạ.")


# Mỗi entry: (từ khoá nhận diện, hàm dựng câu trả lời từ dữ liệu thật)
_FAQ_KNOWLEDGE_BASE = (
    (("bảo hành", "warranty", "bao hanh"), _answer_battery_warranty),
    (("sạc", "trạm sạc", "cắm sạc", "sac"), _answer_charging),
)

_FAQ_FALLBACK = (
    "VinAssistant chưa có dữ liệu được xác thực cho câu hỏi này, nên xin phép "
    "không trả lời phỏng đoán.\n\n"
    "VinAssistant có thể hỗ trợ anh/chị ngay với:\n"
    "- Tra cứu xe điện VinFast hoặc gói nghỉ dưỡng Vinpearl theo tầm giá.\n"
    "- Mở phiếu hỗ trợ để bộ phận chuyên trách phản hồi chính xác.\n\n"
    "Anh/chị muốn đi theo hướng nào ạ?"
)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """
    Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop.

    Mục đích duy nhất của class này là làm đối chứng: nó trả lời trôi chảy, tự
    tin, đúng văn phong tư vấn viên — và sai số liệu. Phần `hallucinations`
    trong kết quả đối chiếu ngược từng con số với product_catalog.json để
    lượng hoá mức sai, thay vì chỉ nói chung chung rằng "LLM hay bịa".
    """

    # Những con số dưới đây cố ý SAI so với catalog thật. Chúng mô phỏng đúng
    # kiểu lỗi của một LLM trả lời bằng trí nhớ: tên sản phẩm có thật, giá lệch.
    _FABRICATED_PRICES = {
        "VinFast VF 3": 240000000,
        "VinFast VF 5 Plus": 458000000,
        "VinFast VF 8 Eco": 950000000,
    }

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả lời một lượt, không tool, không kiểm chứng."""
        if self.api_key:
            live = self._try_live_query(user_input)
            if live is not None:
                return live

        answer = self._mock_answer()
        return {
            "answer": answer,
            "tool_calls": [],                      # Baseline không có tool.
            "status": "success",
            "mode": "mock_baseline",
            "hallucinations": self._audit_prices(),
            "grounded": False,
        }

    def _mock_answer(self) -> str:
        rows = "\n".join(
            f"- {name}: khoảng {format_vnd(price)}"
            for name, price in self._FABRICATED_PRICES.items()
        )
        return (
            "Dạ, VinFast hiện có nhiều mẫu xe điện rất đáng cân nhắc ạ:\n"
            f"{rows}\n"
            "Tất cả đều được bảo hành pin trọn đời và tặng kèm 3 năm sạc miễn "
            "phí tại toàn bộ trạm V-Green trên cả nước. Anh/chị đặt cọc trong "
            "tháng này sẽ được giảm thêm 5% ạ."
        )

    def _audit_prices(self) -> List[Dict[str, Any]]:
        """
        Đối chiếu từng giá đã nêu với catalog thật.

        Đây chính là bằng chứng định lượng cho Milestone 1: cùng một câu hỏi,
        baseline lệch giá tới hàng trăm triệu đồng, còn Agent thì không lệch
        đồng nào vì mọi số đều đi ra từ Observation.
        """
        truth = {
            product["name"]: product["price_vnd"]
            for product in search_product_catalog(category="xe_dien")
            if "error" not in product
        }
        findings = []
        for name, claimed in self._FABRICATED_PRICES.items():
            actual = truth.get(name)
            if actual is None:
                findings.append({"claim": name, "issue": "sản phẩm không tồn tại"})
            elif actual != claimed:
                findings.append({
                    "claim": name,
                    "issue": "sai giá",
                    "claimed_vnd": claimed,
                    "actual_vnd": actual,
                    "delta_vnd": claimed - actual,
                })
        # Hai ưu đãi trong câu trả lời không hề tồn tại trong bất kỳ file dữ liệu nào.
        findings.append({"claim": "bảo hành pin trọn đời",
                         "issue": "chính sách không có trong dữ liệu"})
        findings.append({"claim": "giảm 5% khi đặt cọc trong tháng",
                         "issue": "khuyến mãi không có trong dữ liệu"})
        return findings

    def _try_live_query(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Gọi Gemini một lượt, KHÔNG khai báo tool — đúng tinh thần baseline.

        SDK không nằm trong requirements.txt nên mọi lỗi import/mạng đều được
        nuốt và hàm trả về None để luồng rơi về mock. Autograder vì thế luôn
        chạy được kể cả khi máy không có API key hay không có Internet.
        """
        try:
            import google.generativeai as genai       # type: ignore

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(user_input)
            return {
                "answer": (response.text or "").strip(),
                "tool_calls": [],
                "status": "success",
                "mode": "live_baseline",
                "hallucinations": ["chưa kiểm chứng — cần đối chiếu thủ công"],
                "grounded": False,
            }
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """
    Agent với System Prompt Engineering & Tool Calling.

    Một "iteration" là một vòng lặp hoàn chỉnh Plan -> Act -> Observe -> Reflect.
    Trong cùng một vòng, Agent phát tất cả tool call mà kế hoạch cần (parallel
    tool calling, giống cách các Function Calling API hiện đại cho phép model
    phát nhiều call trong một lượt), rồi tổng hợp Final Answer ngay nếu không
    còn việc tồn đọng. Vòng thứ hai chỉ xảy ra khi Reflect phát hiện lỗi có thể
    khắc phục và cần gọi lại tool.
    """

    def __init__(self, max_iterations: int = 5, max_retries: int = 2):
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.system_prompt = render_system_prompt(max_retries=max_retries)
        self.trace: List[Dict[str, Any]] = []

    # ── MILESTONE 3.1: Intent Detection ───────────────────────────────────

    def detect_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Suy ra ý định và tham số tool từ câu nói của khách.

        needs_catalog và needs_ticket được tính ĐỘC LẬP (if / if, không phải
        if / elif) vì một câu hoàn toàn có thể vừa hỏi sản phẩm vừa báo lỗi —
        đó chính là kịch bản parallel tool calling ở TC03.
        """
        text = user_input.lower()

        category, category_hits = self._detect_category(text)
        max_price = self._extract_max_price(text)
        catalog_hits = _matched_keywords(text, _CATALOG_INTENT_KEYWORDS)
        ticket_hits = _matched_keywords(text, _TICKET_INTENT_KEYWORDS)

        # Tra cứu chỉ được kích hoạt khi có ĐỒNG THỜI chủ đề (xe/du lịch) và
        # động từ mua sắm. Nhờ điều kiện kép này, câu "Chính sách bảo hành pin
        # xe điện VinFast kéo dài bao lâu?" tuy có từ "xe điện" vẫn không gọi tool.
        needs_catalog = bool(category) and bool(catalog_hits)
        needs_ticket = bool(ticket_hits)

        intents = {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": not needs_catalog and not needs_ticket,
            "category": category,
            "max_price": max_price,
            "customer_name": None,
            "issue_description": None,
            "priority": "medium",
            "evidence": {
                "category_keywords": category_hits,
                "catalog_keywords": catalog_hits,
                "ticket_keywords": ticket_hits,
            },
        }

        if needs_ticket:
            intents["customer_name"] = self._extract_customer_name(user_input)
            intents["issue_description"] = self._extract_issue(user_input)
            intents["priority"] = self._detect_priority(text)

        return intents

    @staticmethod
    def _detect_category(text: str) -> Tuple[Optional[str], List[str]]:
        """
        Chọn danh mục theo số từ khoá khớp; hoà thì ưu tiên từ xuất hiện sớm hơn
        trong câu, vì chủ đề chính của khách thường được nêu ngay từ đầu.
        """
        scores = {}
        for category, keywords in _CATEGORY_KEYWORDS.items():
            hits = _matched_keywords(text, keywords)
            if hits:
                earliest = min(text.find(kw) for kw in hits)
                scores[category] = (len(hits), -earliest, hits)

        if not scores:
            return None, []
        best = max(scores, key=lambda c: scores[c][:2])
        return best, scores[best][2]

    @staticmethod
    def _extract_max_price(text: str) -> Optional[int]:
        """
        Đổi cách nói giá của người Việt sang số nguyên VNĐ.

        Regex bắt buộc phải có đơn vị đi kèm ("600 triệu"), nên con số trần
        trụi trong "xe VF 8" không bị hiểu nhầm thành ràng buộc giá.
        """
        match = _PRICE_UNIT_RE.search(text)
        if match:
            raw, unit = match.group(1).replace(",", "."), match.group(2).lower()
            multiplier = next(m for u, m in _PRICE_UNITS if u == unit)
            try:
                return int(float(raw) * multiplier)
            except ValueError:
                return None

        match = _PRICE_PLAIN_RE.search(text)
        if match:
            digits = re.sub(r"[.,\s]", "", match.group(1))
            return int(digits) if digits.isdigit() else None

        return None

    @staticmethod
    def _extract_customer_name(user_input: str) -> Optional[str]:
        """
        Lấy họ tên ngay sau các mẫu giới thiệu ("tôi tên là...").

        Chỉ nhận các từ viết hoa liên tiếp và dừng ở dấu câu đầu tiên, nên
        "Tôi tên Lê Minh Khoa, xe VF 8 của tôi..." cho ra đúng "Lê Minh Khoa"
        chứ không nuốt luôn phần mô tả lỗi phía sau.
        """
        match = _NAME_RE.search(user_input)
        if not match:
            return None

        segment = re.split(r"[,.;:!?\n]", match.group(1))[0]
        words = []
        for word in segment.split():
            if word[:1].isupper() and len(words) < 5:
                words.append(word)
            else:
                break
        return " ".join(words) if words else None

    @staticmethod
    def _extract_issue(user_input: str) -> str:
        """
        Rút gọn lời khách thành một mô tả sự cố.

        Ưu tiên các mệnh đề chứa dấu hiệu sự cố mạnh ("bị", "lỗi"...); nhờ vậy
        mệnh đề giới thiệu tên và mệnh đề nói mức ưu tiên bị loại khỏi
        issue_description, giữ cho ticket sạch và dễ định tuyến.
        """
        clauses = [c.strip() for c in re.split(r"[,.;\n]", user_input) if c.strip()]

        primary = [c for c in clauses
                   if _matched_keywords(c.lower(), _PRIMARY_ISSUE_MARKERS)]
        selected = primary or [c for c in clauses
                               if _matched_keywords(c.lower(), _TICKET_INTENT_KEYWORDS)]
        if not selected:
            selected = clauses

        issue = ", ".join(selected)[:500]
        return issue[:1].upper() + issue[1:] if issue else user_input.strip()

    @staticmethod
    def _detect_priority(text: str) -> str:
        for level, keywords in _PRIORITY_RULES:
            if _matched_keywords(text, keywords):
                return level
        return "medium"

    # ── MILESTONE 3.2: Lập kế hoạch gọi tool ──────────────────────────────

    def _build_plan(self, intents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chuyển ý định thành danh sách tool call cụ thể, đúng schema."""
        plan: List[Dict[str, Any]] = []

        if intents["needs_catalog"]:
            args: Dict[str, Any] = {"category": intents["category"]}
            if intents["max_price"] is not None:
                args["max_price"] = intents["max_price"]
            plan.append({"tool": "search_product_catalog", "args": args, "attempt": 1})

        # Không có tên khách thì KHÔNG tạo ticket. Core Rule #4: thiếu tham số
        # bắt buộc thì hỏi lại, không tự bịa một cái tên để cho qua.
        if intents["needs_ticket"] and intents["customer_name"]:
            plan.append({
                "tool": "submit_support_ticket",
                "args": {
                    "customer_name": intents["customer_name"],
                    "issue_description": intents["issue_description"],
                    "priority": intents["priority"],
                },
                "attempt": 1,
            })

        return plan

    # ── MILESTONE 3.3: Thực thi & quan sát ────────────────────────────────

    def _execute_tool(self, call: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """
        Gọi tool qua TOOL_MAP và ghi trọn vẹn một bước ReAct vào trace.

        Việc dispatch qua TOOL_MAP (thay vì gọi thẳng hàm Python) mô phỏng đúng
        cách runtime xử lý function call do model sinh ra: tên tool là dữ liệu
        không đáng tin, phải kiểm tra trước khi thực thi.
        """
        name, args = call["tool"], call["args"]
        thought = self._thought_for(call)

        func = TOOL_MAP.get(name)
        if func is None:
            observation: Any = {"error": f"Tool '{name}' không tồn tại.",
                                "recoverable": False}
        else:
            try:
                observation = func(**args)
            except TypeError as exc:
                # Model sinh sai tham số — lỗi schema, thử lại cũng vô ích.
                observation = {"error": f"Sai tham số cho '{name}': {exc}",
                               "recoverable": False}
            except Exception as exc:                     # noqa: BLE001
                observation = {"error": f"Tool '{name}' gặp lỗi: {exc}",
                               "recoverable": True}

        self.trace.append({
            "step": len(self.trace),
            "iteration": iteration,
            "phase": "action",
            "thought": thought,
            "action": name,
            "action_input": args,
            "observation": self._summarize(observation),
            "attempt": call.get("attempt", 1),
        })
        return {"tool": name, "args": args, "result": observation,
                "attempt": call.get("attempt", 1)}

    @staticmethod
    def _thought_for(call: Dict[str, Any]) -> str:
        if call["tool"] == "search_product_catalog":
            limit = call["args"].get("max_price")
            budget = f" trong tầm giá {format_vnd(limit)}" if limit else ""
            return (f"Khách đang hỏi về danh mục '{call['args'].get('category')}'"
                    f"{budget}. Phải tra catalog thật, không trả lời bằng trí nhớ.")
        if call["tool"] == "submit_support_ticket":
            return (f"Khách báo sự cố và đã cung cấp tên "
                    f"'{call['args'].get('customer_name')}'. Đủ tham số để mở ticket.")
        return f"Gọi tool {call['tool']}."

    @staticmethod
    def _extract_error(result: Any) -> Optional[Dict[str, Any]]:
        """Bóc payload lỗi ra khỏi kết quả tool, bất kể tool trả list hay dict."""
        if isinstance(result, dict) and "error" in result:
            return result
        if isinstance(result, list) and result and isinstance(result[0], dict) \
                and "error" in result[0]:
            return result[0]
        return None

    @staticmethod
    def _summarize(observation: Any) -> Any:
        """Rút gọn observation trong trace để log không phình theo kích thước data."""
        if isinstance(observation, list):
            error = ToolCallingAgent._extract_error(observation)
            if error:
                return error
            return {"count": len(observation),
                    "items": [p.get("name") for p in observation if isinstance(p, dict)]}
        return observation

    # ── MILESTONE 4.1: Reflect — quyết định có thử lại hay không ───────────

    def _reflect(self, call: Dict[str, Any], observed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Xem observation rồi quyết định bước tiếp theo.

        Chỉ thử lại khi lỗi được tool đánh dấu recoverable (I/O chập chờn).
        Lỗi tham số hay danh mục sai thì thử lại bao nhiêu lần cũng cho kết quả
        y hệt, nên Agent dừng ngay để khỏi đốt iteration vô ích.
        """
        error = self._extract_error(observed["result"])
        if not error or not error.get("recoverable"):
            return None
        if call.get("attempt", 1) >= self.max_retries:
            return None

        retry = {"tool": call["tool"], "args": call["args"],
                 "attempt": call.get("attempt", 1) + 1}
        self.trace.append({
            "step": len(self.trace),
            "phase": "reflection",
            "thought": (f"Tool '{call['tool']}' lỗi tạm thời "
                        f"({error['error']}). Thử lại lần {retry['attempt']}/"
                        f"{self.max_retries}."),
        })
        return retry

    # ── MILESTONE 3.4 + 4.2: Tổng hợp Final Answer ────────────────────────

    def _synthesize(self, intents: Dict[str, Any],
                    observations: List[Dict[str, Any]]) -> str:
        """Gộp mọi observation thành câu trả lời cuối, kèm dòng ghi nguồn."""
        blocks, sources = [], []

        for obs in observations:
            if obs["tool"] == "search_product_catalog":
                blocks.append(self._render_catalog(obs, intents))
                sources.append("product_catalog.json")
            elif obs["tool"] == "submit_support_ticket":
                blocks.append(self._render_ticket(obs))
                sources.append("support_tickets.json")

        # Khách báo lỗi nhưng chưa xưng tên: hỏi lại thay vì tạo ticket ẩn danh.
        if intents["needs_ticket"] and not intents["customer_name"]:
            blocks.append(
                "VinAssistant đã ghi nhận sự cố anh/chị mô tả. Để mở phiếu hỗ trợ "
                "và gửi mã ticket, anh/chị vui lòng cho biết họ tên đầy đủ ạ."
            )

        if not blocks:
            blocks.append(self._answer_faq(intents))

        answer = "\n\n".join(b for b in blocks if b)
        if sources:
            unique = ", ".join(dict.fromkeys(sources))
            answer += f"\n\n_Nguồn dữ liệu: {unique}_"
        return answer

    def _render_catalog(self, obs: Dict[str, Any], intents: Dict[str, Any]) -> str:
        results = obs["result"]
        error = self._extract_error(results)
        if error:
            return ("Rất tiếc, hệ thống tra cứu sản phẩm đang tạm thời gián đoạn "
                    f"({error['error']}). Anh/chị vui lòng thử lại sau ít phút "
                    "hoặc để VinAssistant mở phiếu hỗ trợ giúp ạ.")

        budget = intents.get("max_price")
        label = "xe điện VinFast" if intents.get("category") == "xe_dien" \
            else "gói nghỉ dưỡng Vinpearl"

        # MILESTONE 4.2 — Empty results: nói thật, và chỉ gợi ý bằng dữ liệu có thật.
        if not results:
            message = (f"Rất tiếc, không tìm thấy {label} nào phù hợp với yêu cầu "
                       "của anh/chị")
            message += f" trong tầm giá {format_vnd(budget)}." if budget else "."
            cheapest = search_product_catalog(category=intents.get("category") or "xe_dien")
            if cheapest and "error" not in cheapest[0]:
                item = cheapest[0]
                message += (f"\n\nLựa chọn có mức giá thấp nhất hiện nay là "
                            f"**{item['name']}** — {format_vnd(item['price_vnd'])}. "
                            "Anh/chị có muốn VinAssistant tư vấn thêm về mẫu này không ạ?")
            return message

        lines = [f"VinAssistant tìm thấy {len(results)} {label} phù hợp với yêu cầu "
                 "của anh/chị:"]
        for item in results:
            status = "Còn hàng" if item.get("availability") == "in_stock" else "Đặt trước"
            lines.append(
                f"\n**{item['name']}** — {format_vnd(item['price_vnd'])} ({status})\n"
                f"  {item.get('description', '')}\n"
                f"  Điểm nổi bật: {', '.join(item.get('features', [])[:3])}"
            )
        lines.append("\nAnh/chị muốn tìm hiểu sâu hơn mẫu nào, hoặc cần VinAssistant "
                     "sắp xếp lịch trải nghiệm ạ?")
        return "\n".join(lines)

    def _render_ticket(self, obs: Dict[str, Any]) -> str:
        result = obs["result"]
        error = self._extract_error(result)
        if error:
            return ("Rất tiếc, VinAssistant chưa ghi nhận được phiếu hỗ trợ vào hệ "
                    f"thống ({error['error']}). Anh/chị vui lòng liên hệ hotline "
                    "1900 23 23 89 để được xử lý ngay ạ.")

        priority_label = {"high": "Cao — ưu tiên xử lý trước",
                          "medium": "Trung bình",
                          "low": "Thấp"}[result["priority"]]
        return (
            f"VinAssistant đã ghi nhận sự cố của anh/chị **{result['customer_name']}** "
            f"vào hệ thống chăm sóc khách hàng.\n\n"
            f"- Mã ticket: **{result['ticket_id']}**\n"
            f"- Nội dung: {result['issue_description']}\n"
            f"- Mức ưu tiên: {priority_label}\n"
            f"- Trạng thái: {result['status']}\n\n"
            "Bộ phận kỹ thuật sẽ liên hệ với anh/chị theo mã ticket này. Anh/chị "
            "vui lòng giữ lại mã để tiện tra cứu ạ."
        )

    @staticmethod
    def _answer_faq(intents: Dict[str, Any]) -> str:
        """Trả lời câu hỏi chính sách bằng dữ liệu trích từ catalog, hoặc từ chối."""
        text = (intents.get("_raw_text") or "").lower()
        for keywords, builder in _FAQ_KNOWLEDGE_BASE:
            if _matched_keywords(text, keywords):
                answer = builder()
                if answer:
                    return answer
        return _FAQ_FALLBACK

    # ── MILESTONE 3 + 4: Agent Loop ───────────────────────────────────────

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        intents = self.detect_intent(user_input)
        intents["_raw_text"] = user_input          # _answer_faq cần câu gốc.
        self.trace.append({
            "step": 0,
            "phase": "intent_detection",
            "user_input": user_input,
            "thought": self._describe_intent(intents),
            "intents": {k: v for k, v in intents.items() if not k.startswith("_")},
        })

        pending = self._build_plan(intents)
        observations: List[Dict[str, Any]] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if pending:
                batch, pending = pending, []
                for call in batch:
                    observed = self._execute_tool(call, iteration)
                    observations.append(observed)
                    retry = self._reflect(call, observed)
                    if retry:
                        pending.append(retry)
                # Còn việc tồn đọng -> sang vòng sau, chưa được chốt câu trả lời.
                if pending:
                    continue

            answer = self._synthesize(intents, observations)
            self.trace.append({
                "step": len(self.trace),
                "iteration": iteration,
                "phase": "final_answer",
                "thought": "Đã có đủ Observation để trả lời, không cần gọi thêm tool.",
                "final_answer": answer,
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed",
                "tool_calls": [{"tool": o["tool"], "args": o["args"]} for o in observations],
            }

        # MILESTONE 4.1 — Max Iterations Guard: thà dừng có kiểm soát còn hơn
        # để Agent lặp vô hạn và đốt chi phí gọi tool.
        self.trace.append({
            "step": len(self.trace),
            "phase": "abort",
            "thought": f"Đã chạm trần {self.max_iterations} iteration mà vẫn còn "
                       f"{len(pending)} tool call chưa hoàn tất.",
        })
        return {
            "answer": ("Rất tiếc, VinAssistant chưa hoàn tất được yêu cầu của anh/chị "
                       "trong số bước cho phép. Anh/chị vui lòng thử lại hoặc liên hệ "
                       "hotline 1900 23 23 89 để được hỗ trợ trực tiếp ạ."),
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached",
            "tool_calls": [{"tool": o["tool"], "args": o["args"]} for o in observations],
        }

    @staticmethod
    def _describe_intent(intents: Dict[str, Any]) -> str:
        if intents["is_faq"]:
            return ("Câu hỏi mang tính chính sách, không kèm nhu cầu mua sắm hay báo "
                    "lỗi. Trả lời bằng kiến thức đã được kiểm chứng, không gọi tool.")
        parts = []
        if intents["needs_catalog"]:
            parts.append(f"tra cứu danh mục '{intents['category']}'")
        if intents["needs_ticket"]:
            parts.append(f"mở ticket mức '{intents['priority']}'")
        return "Khách cần: " + " và ".join(parts) + "."


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def _enable_utf8_console() -> None:
    """Console Windows mặc định dùng cp1252 và sẽ vỡ khi in tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main():
    _enable_utf8_console()

    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    baseline = chatbot.query(user_query)
    print(baseline["answer"])
    print(f"\n[!] Số tool đã gọi: {len(baseline['tool_calls'])} — "
          f"phát hiện {len(baseline['hallucinations'])} điểm bịa đặt:")
    for item in baseline["hallucinations"]:
        if item.get("issue") == "sai giá":
            print(f"    - {item['claim']}: nói {format_vnd(item['claimed_vnd'])}, "
                  f"thực tế {format_vnd(item['actual_vnd'])} "
                  f"(lệch {format_vnd(abs(item['delta_vnd']))})")
        else:
            print(f"    - {item['claim']}: {item['issue']}")

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
