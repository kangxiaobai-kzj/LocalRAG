# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：敏感信息扫描
from security.scanner import scan_sensitive_info, scan_pdf_file


def test_scan_counts():
    text = "手机 13812345678 与 13912345678；邮箱 a@b.com；内网 192.168.1.10"
    result, total = scan_sensitive_info(text)
    assert result["phone"]["count"] == 2
    assert result["email"]["count"] == 1
    assert result["ip"]["count"] == 1
    assert total == 4


def test_scan_id_card_and_landline():
    text = "身份证 110101199003074510，座机 010-12345678"
    result, total = scan_sensitive_info(text)
    assert result["id_card"]["count"] == 1
    assert result["landline"]["count"] == 1
    # 身份证 18 位不应同时被银行卡规则重复计数
    assert "bankcard" not in result
    assert total == 2


def test_scan_amount_bankcard_fee():
    text = "报价 1200元，结算价 1.5万元；卡号 6222021234567890123"
    result, total = scan_sensitive_info(text)
    assert result["amount"]["count"] == 2
    assert result["bankcard"]["count"] == 1
    assert result["fee"]["count"] >= 2  # 报价/结算价 关键词命中
    assert total == 5


def test_scan_credit_code_passport_plate():
    text = "信用代码 91310000MA1K35M18W；护照 E12345678；车牌 沪A12345"
    result, total = scan_sensitive_info(text)
    assert result["credit_code"]["count"] == 1
    assert result["passport"]["count"] == 1
    assert result["plate"]["count"] == 1
    assert total == 3


def test_scan_foreign_amount_and_toll_free():
    text = "美元 8000USD 报价；客服 400-123-4567"
    result, total = scan_sensitive_info(text)
    assert result["amount"]["count"] == 1
    assert result["toll_free"]["count"] == 1
    assert result["fee"]["count"] >= 1


def test_scan_empty_or_none():
    assert scan_sensitive_info("") == ({}, 0)
    assert scan_sensitive_info(None) == ({}, 0)


def test_scan_pdf_missing_file(tmp_path):
    # 文件不存在时不应抛异常
    result, total, chars = scan_pdf_file(str(tmp_path / "not_exist.pdf"))
    assert result == {} and total == 0 and chars == 0
