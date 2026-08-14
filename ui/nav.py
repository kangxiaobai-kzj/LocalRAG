# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# ui/nav.py
# 顶栏导航：页面切换 + 设置入口（RAGFlow 风格）
# 采用按钮式导航（st.button 无状态残留，激活态用 primary 高亮），
# 避免 st.pills 内部状态与 app_page 相互覆盖的坑。
import streamlit as st

# 页面定义：(key, 标签)
PAGES = [
    ("chat", "💬 对话"),
    ("search", "🔍 检索"),
    ("kb", "📚 知识库"),
    ("tutorial", "🎓 教程"),
]
SETTINGS_KEY = "settings"


def get_current_page() -> str:
    return st.session_state.get("app_page", "chat")


def render_top_nav():
    """渲染顶部导航栏（RAGFlow 风格）：品牌区 + 页面切换按钮 + 设置入口。"""
    current = get_current_page()

    c_brand, c_pages, c_settings = st.columns([1.6, 4.0, 0.7], gap="medium", vertical_alignment="center")
    with c_brand:
        st.markdown(
            '<div class="brand">'
            '<span class="brand-icon">🧠</span>'
            '<span class="brand-name">AI 专家助手</span>'
            '<span class="brand-sub">私有知识库 · Agentic RAG</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c_pages:
        for col, (key, label) in zip(st.columns(len(PAGES)), PAGES):
            with col:
                if st.button(label, key=f"nav_{key}", use_container_width=True,
                             type="primary" if current == key else "secondary"):
                    if key != current:
                        st.session_state["app_page"] = key
                        st.rerun()
    with c_settings:
        if st.button("⚙️ 设置", key="nav_settings", use_container_width=True,
                     type="primary" if current == SETTINGS_KEY else "secondary",
                     help="打开设置页"):
            st.session_state["app_page"] = SETTINGS_KEY
            st.rerun()
