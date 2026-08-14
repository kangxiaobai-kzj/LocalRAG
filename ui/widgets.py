# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# ui/widgets.py
# 消息渲染、Token 统计、溯源卡片、思考过程、会话消息写入等 UI 组件（Streamlit 渲染层）
import html
import json
import re
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from llm.tokenizer import count_tokens
from rag.version import parse_version

# 正文中的内联来源标注（LLM 生成，如【来源：xxx.pdf】）
_INLINE_SRC_RE = re.compile(r"【来源：[^】]*】")


def style_inline_sources(text: str, sources=None) -> str:
    """
    把正文内联来源标注【来源：xxx】渲染为数字角标 [1][2]（RAGFlow 风格）。
    - 角标数字 = 参考资料区块的序号，一一对应；hover 显示「来源：xxx · P.页」。
    - user-select: none 保证手动框选复制时角标不会带入正文。
    - sources 缺失时直接移除标注（无参考资料可对应）。
    """
    if not text:
        return text
    if not sources:
        return _INLINE_SRC_RE.sub("", text)

    # 文件名 → (序号, 页码摘要)，序号与参考资料列表顺序一致
    index = {}
    for i, s in enumerate(sources, start=1):
        src = str(s.get("source", "")).strip()
        if not src:
            continue
        pages = ", ".join(str(p) for p in s.get("pages", []))
        index[src] = (i, pages)

    def _repl(m):
        name = m.group(0)[4:-1].strip()  # 去掉【来源： 】
        if name in index:
            i, pages = index[name]
            tip = html.escape(f"来源：{name} · P.{pages}")
            return f"<sup class='src-inline' data-tip='{tip}'>{i}</sup>"
        return ""

    return _INLINE_SRC_RE.sub(_repl, text)


def strip_inline_sources(text: str) -> str:
    """移除正文内联来源标注，生成纯净可粘贴正文（来源以「复制引用」保留）。"""
    if not text:
        return text
    return _INLINE_SRC_RE.sub("", text)


def add_message(role: str, content: str, token_usage: dict = None, extra: dict = None):
    """
    添加一条消息到会话，并同步更新 token 计数和会话更新时间。
    extra 内的键（如 sources / source_chunks / thinking）会并入消息，用于回显富内容。
    """
    token_count = count_tokens(content, role)
    msg = {"role": role, "content": content}
    if token_usage is not None:
        msg["token_usage"] = token_usage
    if extra:
        msg.update(extra)
    st.session_state.messages.append(msg)
    st.session_state.messages_token_counts.append(token_count)

    # 更新 all_sessions 中的消息和更新时间
    current_id = st.session_state.current_session_id
    st.session_state.all_sessions[current_id]["messages"] = st.session_state.messages.copy()
    st.session_state.all_sessions[current_id]["updated_at"] = datetime.now().isoformat()


def _write_clipboard(content: str):
    """把文本写入剪贴板（本地应用，Chrome 下受信任）。"""
    payload = json.dumps(content).replace("<", "\\u003c").replace(">", "\\u003e")
    components.html(
        f"""<script>
const t = {payload};
if (navigator.clipboard && window.isSecureContext) {{
    navigator.clipboard.writeText(t).catch(() => {{}});
}} else {{
    const ta = document.createElement('textarea');
    ta.value = t; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta);
}}
</script>""",
        height=0,
    )


def _refs_text(sources) -> str:
    """生成结构化引用文本（文件名 + 页码），供一键复制粘贴到笔记/文档。"""
    lines = ["参考资料："]
    for i, s in enumerate(sources, start=1):
        src = s.get("source", "未知")
        pages = ", ".join(str(p) for p in s.get("pages", []))
        lines.append(f"{i}. {src}（P.{pages}）")
    return "\n".join(lines)


def render_source_cards(sources, chunks=None, key_prefix=""):
    """
    参考资料区块（RAGFlow 风格）：默认折叠、按文件聚合、可复制结构化引用。
    展示在回答正文下方，不打断阅读；展开后逐文件列出来源与命中切片。
    """
    if not sources:
        return
    with st.expander(f"📚 参考资料（{len(sources)} 份）", expanded=False):
        c_copy, _ = st.columns([0.3, 0.7])
        with c_copy:
            if st.button("📋 复制引用", key=f"copy_refs_{key_prefix}", use_container_width=True,
                         help="复制文件名与页码列表，方便粘贴到文档"):
                _write_clipboard(_refs_text(sources))
                st.toast("✅ 引用信息已复制")
        for i, s in enumerate(sources, start=1):
            src = str(s.get("source", "未知"))
            pages = ", ".join(str(p) for p in s.get("pages", []))
            _, ver_label, _ = parse_version(src)
            ver_badge = (f"<span class='badge badge-version'>{html.escape(ver_label)}</span>"
                         if ver_label else "")
            items = [c for c in (chunks or []) if c.get("source") == src]
            if items:
                with st.expander(f"[{i}] 📄 {src} · P{pages}" + (f" · {ver_label}" if ver_label else ""),
                                 expanded=False):
                    # 命中切片以横向卡片网格展示（flex 换行），文本压平为横排便于阅读
                    cards = []
                    for c in items:
                        page_no = c.get("page", "?")
                        text = re.sub(r"\s+", " ", html.escape(c.get("content", "") or "")).strip()
                        cards.append(
                            f'<div class="src-chunk"><span class="src-chunk-page">P.{page_no}</span>'
                            f'<span class="src-chunk-text">{text}</span></div>')
                    st.markdown('<div class="src-chunks">' + "".join(cards) + "</div>",
                                unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='src-card'><span class='src-no'>[{i}]</span> "
                    f"📄 <span class='src-name'>{html.escape(src)}</span>{ver_badge}"
                    f"<span class='src-pages'>P.{html.escape(pages)}</span></div>",
                    unsafe_allow_html=True,
                )


def render_thinking(trace):
    """渲染思考过程（可折叠）：意图、工具调用、召回/精排、生成等阶段。"""
    if not trace:
        return
    with st.expander("🧠 思考过程"):
        for stage, kw in trace:
            if stage == "thinking":
                st.write("🤖 Agent 思考中，正在决策是否调用工具...")
            elif stage == "tool_call":
                st.write(f"🛠️ 调用工具 `{kw.get('name')}` 参数 "
                         f"`{json.dumps(kw.get('args') or {}, ensure_ascii=False)}`")
            elif stage == "tool_result":
                st.write(f"✅ 工具 `{kw.get('name')}` 执行完成")
            elif stage == "retrieved":
                st.write(f"📄 初步召回 {kw.get('count', 0)} 个文档切片")
            elif stage == "reranked":
                st.write(f"🧠 精排保留 Top-{kw.get('count', 0)}")
            elif stage == "chat":
                st.write("💬 快速通道直达闲聊，未调用工具")
            elif stage == "generating":
                st.write("✍️ 生成回答")


def _copy_button(content: str, key: str):
    """复制按钮：把回答纯净正文写入剪贴板（自动去除内联来源标注，来源可另用「复制引用」）。"""
    if st.button("📋", key=key, help="复制纯净正文（不含来源标注）"):
        _write_clipboard(strip_inline_sources(content))
        st.toast("✅ 已复制到剪贴板")


def render_messages(messages):
    """渲染消息列表：助手消息 = 思考过程（可折叠）+ 正文（来源角标化）+ 参考资料（默认折叠）+ Token + 操作按钮。"""
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_thinking(msg.get("thinking"))
            # 正文：内联来源标注渲染为数字角标（hover 查看来源，复制不带入）
            st.markdown(style_inline_sources(msg["content"], msg.get("sources")),
                        unsafe_allow_html=True)
            # 参考资料：放在正文下方（RAGFlow 风格，默认折叠，不打断阅读）
            if msg["role"] == "assistant" and msg.get("sources"):
                render_source_cards(msg["sources"], msg.get("source_chunks"), key_prefix=f"m{idx}")
            # 助手消息显示 token 统计
            if msg["role"] == "assistant" and "token_usage" in msg:
                usage = msg["token_usage"]
                total = usage.get('total_tokens', 'N/A')
                prompt_t = usage.get('prompt_tokens', 'N/A')
                comp_t = usage.get('completion_tokens', 'N/A')
                st.caption(f"⏱️ 消耗 Token: 总计 {total} (输入 {prompt_t}, 输出 {comp_t})")
            # 操作按钮：复制 / 重新生成（助手消息）
            if msg["role"] == "assistant":
                _, c1, c2 = st.columns([5.5, 0.9, 0.9])
                with c1:
                    _copy_button(msg["content"], key=f"copy_{idx}")
                with c2:
                    if st.button("🔄", key=f"regenerate_{idx}", help="重新生成"):
                        st.session_state.regenerate_pending = True
                        st.session_state.regenerate_target_idx = idx
                        st.rerun()
