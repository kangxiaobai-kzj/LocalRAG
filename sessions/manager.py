# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# sessions/manager.py
# 会话管理：创建、保存、加载、删除，本地 JSON 持久化
# messages 与 title 均加密落盘（Windows 用 DPAPI 绑定当前用户），磁盘不落明文；
# 标题在侧栏展示时由 load_all_sessions 解密返回（解密成本极低）。
import json
import os
from datetime import datetime

from utils import encrypt_secret, decrypt_secret


def get_welcome_message(config):
    role_def = config.get("agent_role", "")
    if role_def and role_def.strip():
        first_sentence = role_def.split("。")[0].strip()
        if first_sentence.startswith("你是"):
            role_name = first_sentence[2:].strip()
        else:
            role_name = first_sentence
        if role_name:
            return f"您好！我是{role_name}，请问有什么可以帮您？"
    return "您好！我是您的专属AI专家。请问有什么可以帮您？"


def generate_session_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"][:20]
            return text + ("..." if len(msg["content"]) > 20 else "")
    return "新对话"


def _encrypt_messages(messages):
    """将消息列表整体加密为密文串；空列表返回空串。"""
    return encrypt_secret(json.dumps(messages, ensure_ascii=False))


def _decrypt_messages(blob, fallback=None):
    """解密消息密文；空串/解密失败时回退（旧版明文 messages 或空列表）。"""
    if blob:
        try:
            raw = decrypt_secret(blob)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return fallback if fallback is not None else []


def save_session(sessions_dir, session_id, messages):
    file_path = os.path.join(sessions_dir, f"{session_id}.json")
    data = {
        "session_id": session_id,
        "messages_enc": _encrypt_messages(messages),
        "title_enc": encrypt_secret(generate_session_title(messages)),
        "updated_at": datetime.now().isoformat()
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_session_title_only(sessions_dir, session_id, new_title):
    file_path = os.path.join(sessions_dir, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title_enc"] = encrypt_secret(new_title)
        data.pop("title", None)  # 清理旧版明文标题
        data["updated_at"] = datetime.now().isoformat()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_all_sessions(sessions_dir):
    sessions = {}
    if not os.path.exists(sessions_dir):
        return sessions
    for filename in os.listdir(sessions_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(sessions_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session_id = data.get("session_id", filename.replace(".json", ""))
                    messages = _decrypt_messages(data.get("messages_enc"),
                                                 fallback=data.get("messages"))
                    # 标题优先解密；旧版明文 title 或解密失败时回退
                    title = decrypt_secret(data.get("title_enc")) or data.get("title", "未命名对话")
                    sessions[session_id] = {
                        "messages": messages,
                        "title": title,
                        "updated_at": data.get("updated_at", "")
                    }
            except Exception:
                pass
    return sessions


def delete_session(sessions_dir, session_id):
    file_path = os.path.join(sessions_dir, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
