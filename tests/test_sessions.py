# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：会话持久化（含加密存储与旧版明文兼容）
import json

from sessions.manager import save_session, load_all_sessions, delete_session


def test_save_load_roundtrip(tmp_path):
    sid = "abc123"
    messages = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好"}]
    save_session(str(tmp_path), sid, messages)

    sessions = load_all_sessions(str(tmp_path))
    assert sid in sessions
    assert sessions[sid]["title"] == "你好"
    assert len(sessions[sid]["messages"]) == 2

    delete_session(str(tmp_path), sid)
    assert sid not in load_all_sessions(str(tmp_path))


def test_load_empty_dir(tmp_path):
    assert load_all_sessions(str(tmp_path)) == {}


def test_session_messages_encrypted_on_disk(tmp_path):
    """落盘文件不得包含明文消息内容（敏感信息加密存储）。"""
    sid = "enc1"
    secret = "绝密内容secret-xyz"
    save_session(str(tmp_path), sid, [{"role": "user", "content": secret}])
    raw = (tmp_path / f"{sid}.json").read_text(encoding="utf-8")
    assert secret not in raw, "会话明文泄露！"
    assert "messages_enc" in raw


def test_session_decrypt_roundtrip(tmp_path):
    """加密写盘后能完整解密还原。"""
    sid = "enc2"
    messages = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]
    save_session(str(tmp_path), sid, messages)
    assert load_all_sessions(str(tmp_path))[sid]["messages"] == messages


def test_legacy_plaintext_session_compatible(tmp_path):
    """旧版明文 messages 字段仍可正常读取（向后兼容）。"""
    sid = "legacy1"
    messages = [{"role": "user", "content": "旧版明文消息"}]
    (tmp_path / f"{sid}.json").write_text(
        json.dumps({"session_id": sid, "messages": messages, "title": "旧版", "updated_at": "x"},
                   ensure_ascii=False), encoding="utf-8")
    assert load_all_sessions(str(tmp_path))[sid]["messages"] == messages
