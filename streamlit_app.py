# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# streamlit_app.py - Streamlit Web 入口（薄壳路由）
# 职责：页面初始化、会话状态管理、组件装配、顶栏路由分发。
# 业务编排位于 core/，检索位于 rag/，渲染组件位于 ui/，均不依赖本文件。
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from config import SESSIONS_DIR, KNOWLEDGE_BASE_DIR, CHROMA_DB_DIR, CONFIG_FILE
from core.orchestrator import Orchestrator
from llm.tokenizer import count_tokens
from sessions.manager import get_welcome_message, save_session, load_all_sessions
from ui.nav import render_top_nav, get_current_page, SETTINGS_KEY
from ui.sidebar import render_chat_sidebar
from ui.pages import (
    render_chat_page,
    render_search_page,
    render_search_loading,
    render_kb_page,
    render_tutorial_page,
    render_settings_page,
)
from rag.retriever import load_retriever_and_reranker
from utils import load_config

load_dotenv()

# ==========================================
# 1. 页面配置与常量
# ==========================================
st.set_page_config(page_title="🧭 AI 专家助手", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")

# ==========================================
# 全局样式：RAGFlow 风格（全宽置顶顶栏 + 左侧会话栏 + 卡片化内容）
# 注意：Streamlit 不注入 --primaryColor 这类驼峰 CSS 变量，必须用自定义 :root 变量。
# ==========================================
st.markdown("""
<style>
/* ========== 主题变量：统一蓝紫系色调 ========== */
:root {
    --brand: #4B3FE3;          /* 主品牌紫 */
    --brand-deep: #3A2FC9;     /* 激活/悬停深紫 */
    --brand-soft: #EFEDFC;     /* 浅紫底（hover/选中） */
    --bg: #FBFBFD;             /* 页面背景：微冷白 */
    --panel: #F5F5FA;          /* 侧边栏面板 */
    --border: #E8E8F1;         /* 分隔线 */
    --text: #20202E;           /* 主文字 */
    --text-sub: #8B8B9E;       /* 次要文字 */
}

/* ========== 隐藏 Streamlit 原生干扰（Deploy / 工具栏 / 头部覆盖层） ========== */
/* stHeader 是覆盖在页面顶部的原生头（z-index 极高），不隐藏会遮住顶栏按钮的点击区 */
[data-testid="stToolbar"],
[data-testid="stDeployButton"],
[data-testid="stDecoration"],
[data-testid="stHeader"] { display: none !important; }

/* ========== 页面骨架 ========== */
[data-testid="stMain"] { background: linear-gradient(180deg, #FBFBFD 0%, #F6F5FC 100%); }
[data-testid="stMain"] .block-container { padding-top: 0.6rem; padding-bottom: 2rem; max-width: 1600px; }

/* ========== 顶栏：仅含品牌区的块置顶（精确匹配，避免误伤消息等其它块） ========== */
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) {
    position: sticky; top: 0; z-index: 1000;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(6px);
    border-bottom: 1px solid var(--border);
    padding: 0.45rem 0 0.35rem; margin-bottom: 0.8rem;
}

/* 品牌区 */
div.brand { display: flex; align-items: center; gap: .55rem; height: 100%; }
div.brand .brand-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 9px;
    background: linear-gradient(135deg, var(--brand), #7A6FF5);
    color: #fff; font-size: 1.1rem;
    box-shadow: 0 2px 8px rgba(75, 63, 227, .25);
}
div.brand .brand-name { font-size: 1rem; font-weight: 700; color: var(--text); }
div.brand .brand-sub { font-size: .72rem; color: var(--text-sub); }

/* 顶栏导航按钮：扁平（无边框，hover 浅紫底） */
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) button {
    border-radius: 8px; font-size: .88rem; padding: .3rem .85rem;
    transition: all .15s ease;
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) button[kind="secondary"] {
    background: transparent; border: none; color: #4a4a58;
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) button[kind="secondary"]:hover {
    background: var(--brand-soft); color: var(--brand); transform: translateY(-1px);
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) button[kind="primary"] {
    background: var(--brand); color: #fff; font-weight: 600;
    box-shadow: 0 1px 4px rgba(75, 63, 227, .3);
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) button[kind="primary"]:hover {
    background: var(--brand-deep);
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) button p {
    margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ========== 左侧会话栏：置顶区 + 独立滚动列表 ========== */
/* API 未配置提醒条（红点警示） */
div.api-alert {
    display: flex; align-items: center; gap: 6px;
    background: #FDECEC; color: #C0392B;
    border: 1px solid #F5C6C6; border-radius: 8px;
    font-size: .78rem; font-weight: 600;
    padding: .35rem .5rem; margin-bottom: .4rem;
}
.st-key-session_panel {
    background: var(--panel); border-radius: 12px; padding: .5rem .45rem;
    position: sticky; top: 62px;
    box-shadow: 0 2px 12px rgba(60, 55, 120, .07);
}
/* 置顶区：搜索框 + 对话管理（固定，不随列表滚动） */
.st-key-session_head {
    padding: 0 .15rem .4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: .4rem;
}
.st-key-session_head .session-title {
    font-size: .78rem; font-weight: 700; color: var(--text-sub);
    padding: .1rem .45rem .3rem; letter-spacing: .04em;
}
/* 新建对话：品牌描边按钮 */
.st-key-session_head button {
    border-radius: 8px; font-size: .84rem; font-weight: 600;
}
.st-key-session_head button[kind="secondary"] {
    background: #fff; border: 1px solid var(--brand); color: var(--brand);
}
.st-key-session_head button[kind="secondary"]:hover { background: var(--brand-soft); }
.st-key-session_head button p { margin: 0; }

/* 历史会话列表：独立滚动区 */
.st-key-session_list {
    max-height: calc(100vh - 265px); overflow-y: auto;
    padding-right: .1rem;
}
.st-key-session_list button {
    border-radius: 8px; font-size: .84rem; text-align: left;
    margin: .1rem 0; padding: .3rem .4rem;
}
.st-key-session_list button[kind="secondary"] {
    background: #FFFFFF; border: 1px solid var(--border); color: #3d3d50;
}
.st-key-session_list button[kind="secondary"]:hover {
    background: var(--brand-soft); border-color: var(--brand); color: var(--brand);
}
.st-key-session_list button[kind="primary"] { background: var(--brand); color: #fff; }
.st-key-session_list button[kind="primary"]:hover { background: var(--brand-deep); }
/* 选中会话：左侧白色指示条，增强当前项辨识度 */
.st-key-session_list button[kind="primary"] { box-shadow: inset 4px 0 0 rgba(255, 255, 255, .85); }
.st-key-session_list button p { margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.st-key-session_list button[kind="primary"] p { color: #fff; }
/* 重命名/删除小按钮：缩小内边距，只容纳图标，不挤占对话名 */
.st-key-session_list [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(2) button,
.st-key-session_list [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(3) button {
    padding: .28rem .15rem; font-size: .78rem;
}
/* 会话时间分组标题 */
.st-key-session_list .session-group {
    font-size: .7rem; font-weight: 700; color: var(--text-sub);
    letter-spacing: .05em; margin: .5rem .35rem .2rem;
}
.st-key-session_list .session-group:first-child { margin-top: .15rem; }
/* 列表滚动条美化 */
.st-key-session_list::-webkit-scrollbar { width: 5px; }
.st-key-session_list::-webkit-scrollbar-thumb { background: #D6D6E4; border-radius: 4px; }
.st-key-session_list::-webkit-scrollbar-track { background: transparent; }

/* ========== 内容页：无卡片框 ========== */
.st-key-page_card { padding: .2rem .4rem; }

/* ========== 消息气泡：去框去阴影 ========== */
div[data-testid="stChatMessage"] { border-radius: 12px; padding: 8px 14px; margin-bottom: 8px; }
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) { background: var(--brand-soft); }

/* 助手消息操作条（复制/重新生成）：默认淡显，悬停消息时显现 */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stHorizontalBlock"] [data-testid="stColumn"] button {
    opacity: .3; transition: opacity .15s ease;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]):hover
    [data-testid="stHorizontalBlock"] [data-testid="stColumn"] button {
    opacity: 1;
}

/* 状态面板圆角 */
div[data-testid="stStatusWidget"] { border-radius: 10px; }

/* 溯源卡片 */
div.src-card {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 12px; margin: 4px 0; font-size: 13px;
}
div.src-card .src-no {
    color: var(--brand); font-weight: 700; margin-right: 2px;
}
div.src-card .src-name { font-weight: 600; }
div.src-card .src-pages { color: var(--text-sub); font-size: 12px; }

/* 溯源卡片内的命中切片：整行横排卡片（文本横向流动，避免窄列堆成竖条） */
div.src-chunks { display: flex; flex-wrap: wrap; gap: 6px; }
div.src-chunk {
    flex: 1 1 100%; box-sizing: border-box; width: 100%;
    border: 1px solid var(--border); border-radius: 8px;
    padding: 6px 10px; background: var(--panel);
    font-size: 13px; line-height: 1.6; color: var(--text-sub);
}
/* 参考资料折叠区内容紧凑：压缩 expander 内容容器默认内边距（Streamlit 展开状态不依赖 details[open]，需直接命中） */
div[data-testid="stExpander"] details > div { padding-left: 2px; padding-right: 2px; }
div.src-chunk .src-chunk-page {
    display: inline-block; color: var(--brand); font-weight: 700;
    font-size: 11px; margin-right: 6px; background: var(--brand-soft);
    border-radius: 4px; padding: 0 5px; line-height: 1.5;
}
div.src-chunk .src-chunk-text { word-break: break-word; }

/* 正文内联来源角标（RAGFlow 风格：数字上标，hover 显示来源详情，user-select 保证复制不带入） */
sup.src-inline {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 1.15em; height: 1.15em; padding: 0 .2em;
    font-size: .66em; font-weight: 700; font-style: normal;
    color: var(--brand); background: var(--brand-soft);
    border: 1px solid #D9D4F8; border-radius: 4px;
    cursor: help; vertical-align: super; line-height: 1;
    user-select: none; -webkit-user-select: none;
    position: relative; margin: 0 2px;
}
sup.src-inline::after {
    content: attr(data-tip);
    position: absolute; bottom: 135%; left: 50%; transform: translateX(-50%);
    background: #2B2B3A; color: #fff; font-size: .7rem; font-weight: 400;
    padding: 5px 9px; border-radius: 6px; white-space: normal;
    max-width: 340px; line-height: 1.45;
    opacity: 0; pointer-events: none; transition: opacity .15s ease;
    z-index: 50; box-shadow: 0 2px 10px rgba(0, 0, 0, .25);
}
sup.src-inline:hover::after { opacity: 1; }
sup.src-inline:hover { background: #E3DFFA; }

/* 关键词高亮 */
mark { background: #fff3bf; color: #8a5a00; border-radius: 3px; padding: 0 1px; }

/* ========== 输入框 / 控件圆角 ========== */
[data-testid="stChatInput"] { border-radius: 999px; border: 1px solid var(--border); background: #fff; }
[data-testid="stChatInput"]:focus-within { border-color: var(--brand); box-shadow: 0 0 0 3px rgba(75, 63, 227, .12); }
[data-testid="stBaseInput"] > div, [data-testid="stSelectbox"] > div > div, [data-testid="stTextArea"] textarea { border-radius: 8px; }

/* ========== 对话页欢迎卡片（空会话引导） ========== */
div.welcome-card {
    text-align: center; padding: 2.2rem 1.5rem 1.6rem;
}
div.welcome-card .wc-icon { font-size: 2.4rem; }
div.welcome-card .wc-title { font-size: 1.35rem; font-weight: 700; color: var(--text); margin: .6rem 0 .3rem; }
div.welcome-card .wc-sub { color: var(--text-sub); font-size: .88rem; margin-bottom: 1.3rem; }
div.welcome-card .wc-hint { color: var(--text-sub); font-size: .8rem; }

/* ========== 检索结果 / 溯源可折叠卡片 ========== */
div[data-testid="stExpander"] {
    border: 1px solid var(--border); border-radius: 10px;
    margin: 6px 0; overflow: hidden;
    box-shadow: 0 1px 6px rgba(60, 55, 120, .05);
}
div[data-testid="stExpander"] summary { border-radius: 10px; }
div[data-testid="stExpander"] summary:hover { background: var(--brand-soft); }

/* 检索命中分数徽章 */
div.hit-badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 10px; }
span.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: .75rem; font-weight: 600; letter-spacing: .02em;
}
span.badge-fusion { background: var(--brand-soft); color: var(--brand); }
span.badge-rerank { background: #E4F5EC; color: #1E7A46; }
span.badge-version { background: #FDF3E3; color: #9A6700; }

/* ========== 统计指标卡（知识库页） ========== */
[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 14px;
    box-shadow: 0 1px 6px rgba(60, 55, 120, .05);
}

/* ========== 设置页区块小标题（模型配置内的"高级参数"分隔） ========== */
div.set-sec-title {
    font-size: .82rem; color: var(--text-sub); margin: .9rem 0 .3rem;
    padding-top: .6rem; border-top: 1px dashed var(--border);
}
div.set-sec-title b { color: var(--brand); }
/* 必填项提示条 + 红色星号 */
div.set-req-hint {
    font-size: .8rem; color: var(--text-sub); margin: .2rem 0 .6rem;
    background: var(--panel); border: 1px dashed var(--border);
    border-radius: 8px; padding: .35rem .6rem;
}
div.set-req-hint .req { color: #E5484D; font-weight: 700; }
div.set-req-hint b { color: var(--brand); }

/* ========== 教程页卡片 ========== */
div.tut-hero {
    background: linear-gradient(135deg, #4B3FE3, #7A6FF5); color: #fff;
    border-radius: 14px; padding: 1.6rem 1.8rem; margin: .4rem 0 1.2rem;
}
div.tut-hero .th-title { font-size: 1.25rem; font-weight: 700; margin-bottom: .35rem; }
div.tut-hero .th-sub { font-size: .88rem; opacity: .92; line-height: 1.6; }
div.tut-step {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
}
div.tut-step .step-no {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 8px;
    background: var(--brand); color: #fff; font-weight: 700; font-size: .85rem;
    margin-bottom: .5rem;
}
div.tut-step .step-title { font-weight: 700; color: var(--text); margin-bottom: .3rem; }
div.tut-step .step-desc { font-size: .8rem; color: var(--text-sub); line-height: 1.55; }
div.tut-feature {
    display: flex; gap: .7rem; align-items: flex-start;
    background: #fff; border: 1px solid var(--border); border-radius: 10px;
    padding: .7rem .9rem; margin: .4rem 0;
}
div.tut-feature .tf-icon { font-size: 1.15rem; }
div.tut-feature .tf-title { font-weight: 700; color: var(--text); font-size: .88rem; }
div.tut-feature .tf-desc { font-size: .8rem; color: var(--text-sub); line-height: 1.5; }

/* ========== 窄屏适配（防错位） ========== */
@media (max-width: 900px) {
    div.brand .brand-sub { display: none; }
    [data-testid="stMain"] [data-testid="stVerticalBlock"] > div:has(div.brand) button { padding: .2rem .45rem; font-size: .8rem; }
    .st-key-session_panel { position: static; }
    .st-key-session_list { max-height: none; }
}
</style>
""", unsafe_allow_html=True)

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)


# ==========================================
# 2. 状态初始化 (Session State)
# ==========================================
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = load_all_sessions(SESSIONS_DIR)

if "editing_sid" not in st.session_state:
    st.session_state.editing_sid = None

if "current_session_id" not in st.session_state:
    if st.session_state.all_sessions:
        st.session_state.current_session_id = list(st.session_state.all_sessions.keys())[0]
    else:
        new_id = str(uuid.uuid4())[:8]
        st.session_state.current_session_id = new_id
        config0 = load_config(CONFIG_FILE)
        welcome = get_welcome_message(config0)
        st.session_state.all_sessions[new_id] = {
            "messages": [{"role": "assistant", "content": welcome}],
            "title": "新对话",
            "updated_at": datetime.now().isoformat()
        }
        save_session(SESSIONS_DIR, new_id, st.session_state.all_sessions[new_id]["messages"])

current_id = st.session_state.current_session_id

# 初始化 messages 和 messages_token_counts
if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.all_sessions[current_id]["messages"].copy()
    if "messages_token_counts" not in st.session_state or len(st.session_state.messages_token_counts) != len(
            st.session_state.messages):
        st.session_state.messages_token_counts = []
        for msg in st.session_state.messages:
            st.session_state.messages_token_counts.append(count_tokens(msg["content"], msg["role"]))

# 重新生成相关标记
if "regenerate_pending" not in st.session_state:
    st.session_state.regenerate_pending = False
if "regenerate_question" not in st.session_state:
    st.session_state.regenerate_question = ""
if "regenerate_target_idx" not in st.session_state:
    st.session_state.regenerate_target_idx = -1


# ==========================================
# 3. 核心组件预热（Streamlit 缓存）
# ==========================================
@st.cache_resource
def cached_load_retriever_and_reranker():
    return load_retriever_and_reranker(CHROMA_DB_DIR)


def clear_retriever_cache():
    """清空检索器缓存并重置检索页的"模型已加载"标记（知识库变更后需重新加载模型）。"""
    st.session_state["search_model_ready"] = False
    cached_load_retriever_and_reranker.clear()


def check_kb_exists():
    if not os.path.exists(CHROMA_DB_DIR):
        return False
    if not os.listdir(CHROMA_DB_DIR):
        return False
    return True


is_kb_empty = not check_kb_exists()
config = load_config(CONFIG_FILE)
# API Key 支持环境变量回退：config.json 优先，其次 DEEPSEEK_API_KEY
has_api_config = bool(config.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""))

# 组装编排器（持有运行时配置，不感知前端）
orchestrator = Orchestrator(config)

# ==========================================
# 4. 顶栏导航 + 页面分发
# 对话页：左列会话栏 + 右列对话；其余页面全宽
# ==========================================
render_top_nav()
app_page = get_current_page()

if app_page == "chat":
    collapsed = st.session_state.get("sidebar_collapsed", False)
    if collapsed:
        c_side, c_main = st.columns([0.07, 3.93], gap="small")
    else:
        # 侧栏占比从 23.8% 压缩到 ~19%，给主对话/切片卡片区腾出更多宽度
        c_side, c_main = st.columns([0.78, 3.22], gap="medium")
    with c_side:
        with st.container(key="session_panel"):
            render_chat_sidebar(config, collapsed)
    with c_main:
        render_chat_page(orchestrator, config, has_api_config, is_kb_empty, cached_load_retriever_and_reranker)
elif app_page == "search":
    with st.container(key="page_card"):
        # 首次进入（或知识库变更后）走独立加载页：占位页先提交、模型在定时重跑中加载，
        # 避免加载期间上一个页面（如知识库）的内容残留可点击。
        if not st.session_state.get("search_model_ready", False) and os.path.exists(CHROMA_DB_DIR):
            render_search_loading(cached_load_retriever_and_reranker)
        else:
            render_search_page(cached_load_retriever_and_reranker)
elif app_page == "kb":
    with st.container(key="page_card"):
        render_kb_page(clear_retriever_cache)
elif app_page == "tutorial":
    with st.container(key="page_card"):
        render_tutorial_page()
elif app_page == SETTINGS_KEY:
    with st.container(key="page_card"):
        render_settings_page(config)
