# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：文件名版本解析（同主题多版本识别 / 新旧排序）
from rag.version import parse_version


def test_no_version():
    topic, label, key = parse_version("1.OnePoint高精度定位资费和订购专题.pdf")
    assert topic == "1.OnePoint高精度定位资费和订购专题"
    assert label == ""
    assert key == 0.0


def test_year_versions_same_topic():
    t1, l1, k1 = parse_version("GB/T 1234-2018.pdf")
    t2, l2, k2 = parse_version("GB/T 1234-2024.pdf")
    assert t1 == t2 == "GB/T1234"          # 同主题分组
    assert (l1, l2) == ("2018", "2024")
    assert k2 > k1                          # 2024 比 2018 新


def test_semver_versions():
    t1, l1, k1 = parse_version("产品方案v1.0.pdf")
    t2, l2, k2 = parse_version("产品方案v2.0.pdf")
    assert t1 == t2 == "产品方案"
    assert (l1, l2) == ("v1.0", "v2.0")
    assert k2 > k1


def test_edition_versions():
    t1, l1, k1 = parse_version("报告第1版.pdf")
    t2, l2, k2 = parse_version("报告第2版.pdf")
    assert t1 == t2 == "报告"
    assert (l1, l2) == ("第1版", "第2版")
    assert k2 > k1


def test_revision_version():
    topic, label, key = parse_version("制度修订版.pdf")
    assert topic == "制度"
    assert label == "修订版"
    assert key == 0.5  # 修订版略高于无版本，低于数字版本


def test_empty_filename():
    assert parse_version("") == ("", "", 0.0)


def test_date_like_number_not_treated_as_year():
    """20260703 这类日期串不应被误判为年份 2026。"""
    topic, label, key = parse_version("20260703.pdf")
    assert label == ""
    assert key == 0.0
    assert topic == "20260703"


def test_year_beats_no_version():
    """同主题下，带年份版本的文件比无版本文件更新。"""
    t1, _, k1 = parse_version("OnePoint高精度定位资费和订购专题.pdf")
    t2, _, k2 = parse_version("OnePoint高精度定位资费和订购专题2024.pdf")
    assert k2 > k1
