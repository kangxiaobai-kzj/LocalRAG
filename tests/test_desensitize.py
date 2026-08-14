# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：脱敏逻辑
from security.desensitize import desensitize_text


def test_desensitize_phone():
    assert desensitize_text("请联系 13812345678") == "请联系 1**********"


def test_desensitize_email():
    assert desensitize_text("邮箱 a@b.com") == "邮箱 [邮箱]"


def test_desensitize_id_card():
    assert "******************" in desensitize_text("身份证 110101199003074510")


def test_desensitize_landline_and_ip():
    text = "电话 010-12345678，内网 192.168.1.10"
    result = desensitize_text(text)
    assert "[电话]" in result and "[IP]" in result


def test_desensitize_amount():
    assert desensitize_text("费用 ¥1200，折合 300元") == "费用 [金额]，折合 [金额]"
    assert desensitize_text("单价 1.5万元/月") == "单价 [金额]/月"
    assert desensitize_text("成本约 8000 元") == "成本约 [金额]"


def test_desensitize_bankcard():
    assert "[银行卡]" in desensitize_text("收款卡号 6222021234567890123")


def test_desensitize_fee_keywords_not_masked():
    # 费用类关键词（结算价/资费）是语义信息，正则不掩码，仅金额数字被处理
    result = desensitize_text("标准结算价 500元/套")
    assert "结算价" in result and "[金额]" in result


def test_desensitize_foreign_amount():
    assert desensitize_text("报价 $1200") == "报价 [金额]"
    assert desensitize_text("成本 8000美元") == "成本 [金额]"
    assert desensitize_text("预算 EUR 50000") == "预算 [金额]"


def test_desensitize_credit_code():
    # 统一社会信用代码（18 位含字母）被掩码，身份证 18 位纯数字不被此规则掩码
    text = "企业代码 91310000MA1K35M18W，个人 110101199003074510"
    result = desensitize_text(text)
    assert "[统一社会信用代码]" in result
    assert "110101199003074510" not in result


def test_desensitize_passport_and_plate():
    result = desensitize_text("护照 E12345678，车牌 沪A12345")
    assert "[护照]" in result and "[车牌]" in result


def test_desensitize_toll_free():
    assert "[电话]" in desensitize_text("客服 400-123-4567")


def test_desensitize_empty():
    assert desensitize_text("") == ""
    assert desensitize_text(None) is None
