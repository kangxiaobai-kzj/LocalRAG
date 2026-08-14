# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# security/desensitize.py
# 敏感数据脱敏：手机号、身份证、银行卡、金额（本外币）、统一社会信用代码、护照号、
# 车牌号、邮箱、电话（含 400/800）、IP、费用条款
# 此操作仅作用于发往云端 API 的数据，不影响本地文件。
import re

# 统一社会信用代码：18 位（字母+数字，GB 32100-2015），要求至少含 1 个字母，
# 避免与 18 位纯数字身份证混淆（身份证在前面已单独处理）
_CREDIT_CODE_RE = re.compile(
    r'(?<![0-9A-Z])(?=[0-9A-Z]*[A-HJ-NP-Z])[0-9A-HJ-NP-Z]{18}(?![0-9A-Z])'
)
# 中国护照号：1 个大写字母（如 E/G）+ 8 位数字
_PASSPORT_RE = re.compile(r'(?<![A-Z0-9])[A-Z]\d{8}(?![A-Z0-9])')
# 车牌号：省份简称 + 发牌机关字母 + 5-6 位（含新能源/警学等后缀）
_PLATE_RE = re.compile(
    r'(?<![A-Z0-9\u4e00-\u9fa5])'
    r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]'
    r'[A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳]'
    r'(?![A-Z0-9\u4e00-\u9fa5])'
)
# 400/800 客服电话
_TOLL_FREE_RE = re.compile(r'(?:400|800)[- ]?\d{3,4}[- ]?\d{3,4}')


def desensitize_text(text: str) -> str:
    """对文本进行本地脱敏处理，替换手机号、身份证、金额等敏感信息。"""
    if not text:
        return text

    text = re.sub(r'(?<!\d)1[3-9]\d{9}(?!\d)', '1**********', text)
    text = re.sub(r'(?<!\d)\d{17}[\dXx](?!\d)', '******************', text)
    # 统一社会信用代码（含字母的 18 位，不含纯数字身份证）
    text = _CREDIT_CODE_RE.sub('[统一社会信用代码]', text)
    # 银行卡号（16-19 位连续数字；身份证 18 位已在上面处理，避免重复）
    text = re.sub(r'(?<!\d)\d{16,19}(?!\d)', '[银行卡]', text)
    # 金额：本外币符号前缀（¥￥$€£），或 数字+本外币单位，或 外币代码+数字
    text = re.sub(r'[¥￥$€£]\s*\d+(?:\.\d+)?(?:万|亿)?', '[金额]', text)
    text = re.sub(r'(?<!\d)\d+(?:\.\d+)?\s*(?:元|万元|亿元|人民币|美元|欧元|英镑|港币|USD|EUR|GBP|HKD)',
                  '[金额]', text)
    text = re.sub(r'(?<![A-Z])(?:USD|EUR|GBP|HKD)\s*\d+(?:\.\d+)?(?![A-Z0-9])', '[金额]', text)
    text = _PASSPORT_RE.sub('[护照]', text)
    text = _PLATE_RE.sub('[车牌]', text)
    text = _TOLL_FREE_RE.sub('[电话]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[邮箱]', text)
    text = re.sub(r'\d{3,4}-\d{7,8}', '[电话]', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', text)

    return text
