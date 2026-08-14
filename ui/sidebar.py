# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# ui/sidebar.py
# 对话页侧边栏：仅对话管理（模型配置→设置页，知识库→知识库页）
# 结构：置顶区（API 提醒 + 搜索框 + 对话管理入口，固定不滚动） + 历史会话列表（独立滚动）
import os
import uuid
from datetime import datetime

import streamlit as st

from config import SESSIONS_DIR
from llm.tokenizer import count_tokens
from sessions.manager import (
    get_welcome_message,
    save_session,
    save_session_title_only,
    delete_session,
)
from ui.nav import SETTINGS_KEY


def _relative_time(ts: str) -> str:
    """把 ISO 时间转为相对时间描述（如 3 分钟前 / 2 小时前 / 昨天 / 5 天前）。"""
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return ""
    delta = datetime.now() - dt
    if delta.days <= 0:
        if delta.seconds < 3600:
            return f"{max(1, delta.seconds // 60)} 分钟前"
        return f"{delta.seconds // 3600} 小时前"
    if delta.days == 1:
        return "昨天"
    if delta.days < 7:
        return f"{delta.days} 天前"
    return dt.strftime("%Y-%m-%d")


def render_chat_sidebar(config, collapsed=False):
    """渲染对话页左侧会话栏（对话管理）。config 用于生成欢迎语。
    collapsed=True 时仅显示展开按钮，用于窄屏/专注场景收起会话栏。"""
    # ========== 置顶区：搜索框 + 对话管理入口（固定，不随列表滚动）==========
    with st.container(key="session_head"):
        if collapsed:
            if st.button("☰", key="expand_sidebar", help="展开会话栏", use_container_width=True):
                st.session_state.sidebar_collapsed = False
                st.rerun()
            return

        # API 未配置提醒（红点 + 快捷跳转设置）
        has_key = bool(config.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""))
        if not has_key:
            st.markdown('<div class="api-alert">🔴 未配置模型 API Key</div>', unsafe_allow_html=True)
            if st.button("⚙️ 前往设置", key="api_alert_btn", use_container_width=True):
                st.session_state["app_page"] = SETTINGS_KEY
                st.rerun()

        c_search, c_toggle = st.columns([0.86, 0.14], vertical_alignment="center")
        with c_search:
            search_keyword = st.text_input(
                "搜索对话", value="", placeholder="🔍 搜索对话...",
                label_visibility="collapsed", key="session_search",
            )
        with c_toggle:
            if st.button("⏴", key="collapse_sidebar", help="收起会话栏",
                         use_container_width=True):
                st.session_state.sidebar_collapsed = True
                st.rerun()

        st.markdown('<div class="session-title">💬 对话管理</div>', unsafe_allow_html=True)
        if st.button("➕ 新建对话", key="new_session", use_container_width=True):
            new_id = str(uuid.uuid4())[:8]
            st.session_state.current_session_id = new_id
            welcome = get_welcome_message(config)
            st.session_state.all_sessions[new_id] = {
                "messages": [{"role": "assistant", "content": welcome}],
                "title": "新对话",
                "updated_at": datetime.now().isoformat(),
            }
            st.session_state.messages = st.session_state.all_sessions[new_id]["messages"].copy()
            st.session_state.messages_token_counts = [count_tokens(welcome, "assistant")]
            save_session(SESSIONS_DIR, new_id, st.session_state.messages)
            st.rerun()

    # ========== 历史会话列表（可独立滚动）==========
    with st.container(key="session_list"):
        current_id = st.session_state.current_session_id
        filtered = {sid: d for sid, d in st.session_state.all_sessions.items()
                    if search_keyword.lower() in d["title"].lower()}
        if not filtered:
            st.caption("🔍 未找到匹配的会话" if search_keyword.strip() else "暂无历史对话")
        # 会话按时间分组：今天 / 昨天 / 近 7 天 / 更早
        def _group_label(ts):
            try:
                dt = datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                return "更早"
            days = (datetime.now().date() - dt.date()).days
            if days <= 0:
                return "今天"
            if days == 1:
                return "昨天"
            if days < 7:
                return "近 7 天"
            return "更早"

        _prev_group = None
        for sid, data in sorted(filtered.items(),
                                key=lambda x: x[1].get("updated_at", ""), reverse=True):
            _group = _group_label(data.get("updated_at", ""))
            if _group != _prev_group:
                st.markdown(f'<div class="session-group">{_group}</div>', unsafe_allow_html=True)
                _prev_group = _group
            if st.session_state.editing_sid == sid:
                c1, c2 = st.columns([3, 1])
                new_title = c1.text_input("新名称", value=data["title"], key=f"rn_{sid}",
                                          label_visibility="collapsed")
                if c2.button("✅", key=f"ok_{sid}", use_container_width=True) and new_title.strip():
                    data["title"] = new_title.strip()
                    st.session_state.all_sessions[sid] = data
                    save_session_title_only(SESSIONS_DIR, sid, new_title.strip())
                    st.session_state.editing_sid = None
                    st.rerun()
                if st.button("❌ 取消", key=f"cc_{sid}"):
                    st.session_state.editing_sid = None
                    st.rerun()
            else:
                # 列比：对话名 : 重命名 : 删除 ≈ 4.2 : 1.5 : 1.5
                # 实测(侧边栏≈179px)：对话名≈96px，重命名/删除≈26px，比例约 4:1:1
                c1, c2, c3 = st.columns([4.2, 1.5, 1.5])
                rel = _relative_time(data.get("updated_at", ""))
                help_text = f"{data['title']}"
                if rel:
                    help_text += f"\n{rel}"
                help_text += "\n\n点击打开该对话"
                if c1.button(data["title"], key=f"b_{sid}", use_container_width=True,
                             type="primary" if sid == current_id else "secondary",
                             help=help_text):
                    if sid != current_id:
                        save_session(SESSIONS_DIR, current_id, st.session_state.messages)
                        st.session_state.current_session_id = sid
                        st.session_state.messages = st.session_state.all_sessions[sid]["messages"].copy()
                        st.session_state.messages_token_counts = []
                        for msg in st.session_state.messages:
                            st.session_state.messages_token_counts.append(count_tokens(msg["content"], msg["role"]))
                        st.rerun()
                if c2.button("✎", key=f"e_{sid}", use_container_width=True, help="重命名"):
                    st.session_state.editing_sid = sid
                    st.rerun()
                if c3.button("✕", key=f"d_{sid}", use_container_width=True, help="删除") \
                        and len(st.session_state.all_sessions) > 1:
                    delete_session(SESSIONS_DIR, sid)
                    del st.session_state.all_sessions[sid]
                    if sid == current_id:
                        new_curr = list(st.session_state.all_sessions.keys())[0]
                        st.session_state.current_session_id = new_curr
                        st.session_state.messages = st.session_state.all_sessions[new_curr]["messages"].copy()
                        st.session_state.messages_token_counts = []
                        for msg in st.session_state.messages:
                            st.session_state.messages_token_counts.append(count_tokens(msg["content"], msg["role"]))
                    st.rerun()
