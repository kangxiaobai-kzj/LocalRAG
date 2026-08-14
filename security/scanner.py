# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# security/scanner.py
# 敏感信息扫描：统计文本中的手机号、身份证、邮箱、电话、IP、银行卡、金额、费用条款、
# 护照号、车牌号、统一社会信用代码等数量
# 仅做本地统计提示，不发送任何数据。
import re

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

_PHONE_RE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
_ID_RE = re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)')
_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
_LANDLINE_RE = re.compile(r'\d{3,4}-\d{7,8}')
_IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
# 400/800 客服电话
_TOLL_FREE_RE = re.compile(r'(?:400|800)[- ]?\d{3,4}[- ]?\d{3,4}')
# 银行卡号（16-19 位连续数字；身份证 18 位在上面单独处理）
_BANKCARD_RE = re.compile(r'(?<!\d)\d{16,19}(?!\d)')
# 统一社会信用代码（18 位含字母；纯数字 18 位是身份证，已单独处理）
_CREDIT_CODE_RE = re.compile(
    r'(?<![0-9A-Z])(?=[0-9A-Z]*[A-HJ-NP-Z])[0-9A-HJ-NP-Z]{18}(?![0-9A-Z])'
)
# 护照号：1 个大写字母 + 8 位数字
_PASSPORT_RE = re.compile(r'(?<![A-Z0-9])[A-Z]\d{8}(?![A-Z0-9])')
# 车牌号
_PLATE_RE = re.compile(
    r'(?<![A-Z0-9\u4e00-\u9fa5])'
    r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]'
    r'[A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳]'
    r'(?![A-Z0-9\u4e00-\u9fa5])'
)
# 金额（人民币+外币：符号前缀、数字+单位、外币代码+数字）
_AMOUNT_RE = re.compile(
    r'[¥￥$€£]\s*\d+(?:\.\d+)?(?:万|亿)?'
    r'|(?<!\d)\d+(?:\.\d+)?\s*(?:元|万元|亿元|人民币|美元|欧元|英镑|港币|USD|EUR|GBP|HKD)'
    r'|(?<![A-Z])(?:USD|EUR|GBP|HKD)\s*\d+(?:\.\d+)?(?![A-Z0-9])'
)
# 费用/资费类关键词（语义提示：金额条款无法用正则识别，仅提醒可能含费用标准）
_FEE_RE = re.compile(r'(资费|费用标准|结算价|收费标准|单价|报价|计费|收费|价格|成本价|合同金额|薪酬|工资|奖金|补贴|费率)')

# (键, 中文名, 正则)
_SENSITIVE_RULES = [
    ("phone", "手机号", _PHONE_RE),
    ("id_card", "身份证", _ID_RE),
    ("bankcard", "银行卡", _BANKCARD_RE),
    ("credit_code", "统一社会信用代码", _CREDIT_CODE_RE),
    ("passport", "护照号", _PASSPORT_RE),
    ("plate", "车牌号", _PLATE_RE),
    ("amount", "金额", _AMOUNT_RE),
    ("email", "邮箱", _EMAIL_RE),
    ("landline", "电话", _LANDLINE_RE),
    ("toll_free", "客服电话", _TOLL_FREE_RE),
    ("ip", "IP", _IP_RE),
    ("fee", "费用条款", _FEE_RE),
]


def scan_sensitive_info(text):
    """
    统计文本中各类型敏感信息数量。
    身份证（18 位）先匹配并从待统计文本中剔除，避免其同时被银行卡（16-19 位）规则重复计数。
    返回 (result, total)：result = {key: {"name": 中文名, "count": n}}, total 为总数。
    """
    if not text:
        return {}, 0
    result = {}
    total = 0
    working = text
    id_hits = _ID_RE.findall(working)
    if id_hits:
        result["id_card"] = {"name": "身份证", "count": len(id_hits)}
        total += len(id_hits)
        working = _ID_RE.sub("", working)
    for key, name, pattern in _SENSITIVE_RULES:
        if key == "id_card":
            continue
        count = len(pattern.findall(working))
        if count:
            result[key] = {"name": name, "count": count}
            total += count
    return result, total


def scan_pdf_file(file_path):
    """
    快速提取 PDF 文本并扫描敏感信息。
    返回 (result, total, char_count)；解析失败时返回 ({}, 0, 0)。
    """
    text = ""
    if PdfReader is not None:
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                raw = page.extract_text()
                if raw:
                    text += raw + "\n"
        except Exception:
            pass
    result, total = scan_sensitive_info(text)
    return result, total, len(text)
