# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# ui/pages.py
# 页面渲染层：对话页、检索测试页、知识库管理页、教程页、设置页（Streamlit 渲染层）
import gc
import html
import json
import os
import stat
import time
from datetime import datetime

import streamlit as st

from config import (
    CHROMA_DB_DIR,
    CONFIG_FILE,
    KNOWLEDGE_BASE_DIR,
    MAX_UPLOAD_MB,
    RERANK_TOP_K,
    SESSIONS_DIR,
    get_chunk_config,
    get_embedding_model,
)
from llm.client import test_llm_connection
from rag.builder import get_kb_stats, rebuild_vector_store_sync
from rag.parsers import is_supported_file, parse_document
from rag.search_debug import search_debug
from rag.version import parse_version
from security.scanner import scan_sensitive_info
from sessions.manager import generate_session_title, save_session
from ui.widgets import (
    add_message,
    render_messages,
    render_source_cards,
    style_inline_sources,
)
from utils import load_config, save_config, validate_api_config


# ==========================================
# 教程页（面向新用户：产品定位 + 上手步骤 + 模型清单 + 隐私说明 + FAQ）
# ==========================================
def render_tutorial_page():
    st.markdown("### 📖 新手指南")
    st.markdown(
        '<div class="tut-hero">'
        '<div class="th-title">🧭 欢迎使用 AI 专家助手</div>'
        '<div class="th-sub">本项目旨在帮助<b>个人用户在自己的电脑上本地化部署</b>一套私有知识库问答系统：'
        '把个人文档（PDF / TXT / MD / DOCX / XLSX / PPTX）导入本地知识库后，即可用自然语言提问，'
        '系统会从文档中精准检索相关内容并生成带来源标注的回答。'
        '文档处理与检索全程在本地完成，仅生成回答时调用云端大模型，且发送内容已自动脱敏。</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- 三步快速上手 ----
    st.markdown("#### 🚀 三步快速上手")
    c1, c2, c3 = st.columns(3)
    steps = [
        ("1", "⚙️ 配置模型", "点击顶部「设置」，选择模型服务商（DeepSeek / OpenAI / 智谱 / 百炼 / Kimi / SiliconFlow / OpenRouter / Ollama 本地等），填入 API Key 与模型名称，点击「测试连接」确认可用后保存。"),
        ("2", "📚 导入个人文档", "点击顶部「知识库」，上传个人文档（PDF / TXT / MD / DOCX / XLSX / PPTX）并点击「上传并重建知识库」，等待本地构建完成。上传时会自动检测同名文件与旧版本文件并确认处理方式。"),
        ("3", "💬 开始提问", "回到「对话」页输入问题。空会话有快捷提问可一键体验；每条回答可展开「思考过程」查看检索链路。下方为<b>各板块功能总览</b>，便于快速了解系统能做什么。"),
    ]
    for col, (no, title, desc) in zip((c1, c2, c3), steps):
        with col:
            st.markdown(
                f'<div class="tut-step"><span class="step-no">{no}</span>'
                f'<div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    # ---- 各板块功能总览（树状图）----
    st.markdown(
        """
        <style>
        .tut-tree { margin: .6rem 0 1.2rem; }
        .tree-root { display: inline-block; background: linear-gradient(135deg, #4B3FE3, #7A6FF5); color: #fff;
            border-radius: 12px; padding: .5rem 1.4rem; font-weight: 700; margin-bottom: .2rem;
            box-shadow: 0 2px 8px rgba(75,63,227,.25); }
        .tree-branch { margin: .35rem 0 0 1.3rem; border-left: 2px solid #D9D4F8; padding-left: 1.1rem; }
        .tree-node { display: inline-block; font-weight: 700; color: #4B3FE3; background: #EFEDFC;
            border: 1px solid #D9D4F8; border-radius: 10px; padding: .35rem .85rem; margin: .3rem 0; }
        .tree-leaf { position: relative; background: #fff; border: 1px solid #E8E8F1; border-radius: 8px;
            padding: .32rem .75rem; margin: .32rem 0; font-size: .82rem; color: #40404F; }
        .tree-leaf::before { content: ""; position: absolute; left: -1.1rem; top: 50%; width: 1.1rem; height: 2px;
            background: #D9D4F8; }
        @media (max-width: 900px) { .tree-branch { margin-left: .4rem; padding-left: .7rem; } }
        </style>
        <div class="tut-tree">
            <div class="tree-root">🧭 系统功能总览</div>
            <div class="tree-branch">
                <div class="tree-node">💬 对话板块</div>
                <div class="tree-leaf">闲聊快速通道（开启 Web 联网后可搜索互联网）</div>
                <div class="tree-leaf">Agent 智能编排：先查知识库 → 不足时自动联网补充</div>
                <div class="tree-leaf">逐句溯源标注【来源：文件名】+ 引用补全精修</div>
                <div class="tree-leaf">重新生成回答 / 一键复制 / 导出对话为 Markdown</div>
            </div>
            <div class="tree-branch">
                <div class="tree-node">🔍 检索板块</div>
                <div class="tree-leaf">混合检索：BM25 关键词 + 向量语义（RRF 融合）</div>
                <div class="tree-leaf">深度语义重排精排（本地 BGE-reranker）</div>
                <div class="tree-leaf">最相关条数 / 来源过滤 / 仅最新版本</div>
                <div class="tree-leaf">融合分 / 重排分展示与关键词高亮</div>
            </div>
            <div class="tree-branch">
                <div class="tree-node">📚 知识库板块</div>
                <div class="tree-leaf">多格式上传（PDF / TXT / MD / DOCX / XLSX / PPTX）</div>
                <div class="tree-leaf">同名文件与新旧版检测确认（可删除旧版或新旧共存）</div>
                <div class="tree-leaf">敏感信息扫描（仅本地提示，不外发）</div>
                <div class="tree-leaf">增量构建 / 全量重建 / 勾选删除 / 一键清空</div>
            </div>
            <div class="tree-branch">
                <div class="tree-node">⚙️ 设置板块</div>
                <div class="tree-leaf">模型配置：服务商 / API Key / 采样参数</div>
                <div class="tree-leaf">系统设置：Web 联网搜索 / 仅检索最新版本</div>
                <div class="tree-leaf">检索与切片设置：切片大小 / 策略 / 向量模型 / 最相关条数</div>
                <div class="tree-leaf">角色定义：自定义 Agent 定位</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 模型清单：本地部署 vs 需联网 ----
    st.markdown("#### 🤖 系统使用了哪些模型？")
    st.markdown('<div class="tut-feature"><span class="tf-icon">💻</span>'
                '<div><div class="tf-title">本地部署（不上云，无需网络）</div>'
                '<div class="tf-desc">文档解析（pypdf / OCR / python-docx / openpyxl / python-pptx）与文本切片、'
                '向量模型 BAAI/bge-small-zh-v1.5（本地推理）、BM25 关键词检索、'
                '语义重排模型 BAAI/bge-reranker-v2-m3（本地推理）、敏感信息扫描（本地正则）。'
                '注意：两个 BGE 模型首次运行时需联网从 HuggingFace 国内镜像下载权重，'
                '下载后即缓存到本地，之后可完全离线使用。</div></div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tut-feature"><span class="tf-icon">🌐</span>'
                '<div><div class="tf-title">需联网（云端 API）</div>'
                '<div class="tf-desc">大语言模型（LLM）：用于意图识别与回答生成，'
                '由「设置」页配置的服务商提供（DeepSeek / OpenAI / 智谱 / Ollama 本地 / 百炼 / Kimi / '
                'SiliconFlow / OpenRouter 或任意 OpenAI 兼容接口），调用其云端 API。</div></div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tut-feature"><span class="tf-icon">🔎</span>'
                '<div><div class="tf-title">Web 联网搜索（可选，默认关闭）</div>'
                '<div class="tf-desc">默认情况下 Agent 只回答本地知识库内容，不访问互联网（隐私最佳）。'
                '如需补充知识库外的最新资讯，可在「设置 → 系统设置」勾选「启用 Web 联网搜索」，'
                '之后询问行业动态/最新政策等外部信息时，Agent 会自动调用必应搜索（免 Key）。'
                '注意：开启后查询词会发送给搜索引擎，发送云端前内容仍会本地脱敏。<br><br>'
                '<b>联网后的回答策略</b>：知识库已有内容时，Agent 会<b>先检索知识库</b>，'
                '结果充足就只回答库内内容并标注【来源：文件名】，<b>不会联网</b>；'
                '仅当知识库结果<b>不足 / 过时 / 没有</b>时才调用联网搜索补充，'
                '并区分来源：知识库→【来源：文件名】，网页→【来源：网页标题】。'
                '是否联网由大模型按规则自主决策。开启后闲聊分支在需要最新外部信息时也可联网。</div></div></div>',
                unsafe_allow_html=True)

    # ---- 数据与隐私 ----
    st.markdown("#### 🔒 数据与隐私")
    privacy = [
        ("🗂️", "文档全程本地处理", "上传的文档在本地完成解析、切片、向量化，存入本地向量库，不会上传到任何云服务。"),
        ("🔎", "检索也在本地", "混合检索（BM25 + 向量）与语义重排都在本地模型上执行，不涉及联网。"),
        ("🌐", "Web 联网搜索默认关闭", "除非您在「设置 → 系统设置」中主动开启 Web 联网搜索，否则系统不会向任何搜索引擎发送查询词；开启后查询词会发送给必应，但发送云端前仍会本地脱敏。"),
        ("🛡️", "联网内容自动脱敏", "仅「生成回答」环节会把检索到的文本与您的问题发送到云端大模型。发送前自动脱敏：手机号→1**********、身份证→****、银行卡→[银行卡]、金额→[金额]、统一社会信用代码→[统一社会信用代码]、护照→[护照]、车牌→[车牌]、邮箱→[邮箱]、电话→[电话]、IP→[IP]。"),
        ("⚠️", "上传时的敏感扫描", "上传文档时会做本地敏感信息扫描（手机号/身份证/银行卡/金额/邮箱/电话/IP/费用条款等），仅提示数量，不发送任何数据。"),
        ("✏️", "本地文件不会被修改", "脱敏只作用于发往云端的数据，不会改动您电脑上的原始文件。"),
    ]
    for icon, title, desc in privacy:
        st.markdown(
            f'<div class="tut-feature"><span class="tf-icon">{icon}</span>'
            f'<div><div class="tf-title">{title}</div><div class="tf-desc">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )

    # ---- 各页面功能说明 ----
    st.markdown("#### 🧩 功能页面一览")
    features = [
        ("💬", "对话", "支持闲聊快速通道（开启 Web 联网后闲聊也可搜索互联网）与 Agent 智能编排：需要查资料时先自动检索知识库，结果不足/过时再联网补充。回答逐句标注【来源：文件名】（引用不足时自动补全精修）；可重新生成回答、一键复制、导出对话为 Markdown。"),
        ("🔍", "检索", "透明化查看检索过程：输入问题即可看到混合检索 + 语义重排后的命中切片、融合分/重排分与来源页码，支持调整最相关条数、来源过滤、仅最新版本等参数（对标 RAGFlow 检索测试）。"),
        ("📚", "知识库", "上传文档（PDF / TXT / MD / DOCX / XLSX / PPTX）、增量构建或全量重建向量库、勾选删除文件、查看文档与切片统计。上传时自动检测同名文件与旧版本文件并弹出确认（可选择删除旧版或新旧版共存）；同时自动进行敏感信息扫描，命中会提示发送云端前自动脱敏。"),
        ("⚙️", "设置", "配置模型服务商（9 种可选）/API Key/Base URL/Model Name（必填，带 * 标记）、Temperature，以及高级采样参数（Top P / Top K / 频率惩罚 / 上下文长度等，留空即用服务商默认最优配置）。API Key 可勾选「记住」以加密保存到本机（仅当前用户可解密），也可不勾选每次填写。「⚙️ 系统设置」为独立区块（默认折叠），包含「Web 联网搜索」与「仅检索最新版本」两个开关，与 API 配置分离，随时可单独切换。"),
    ]
    for icon, title, desc in features:
        st.markdown(
            f'<div class="tut-feature"><span class="tf-icon">{icon}</span>'
            f'<div><div class="tf-title">{title}</div><div class="tf-desc">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )

    # ---- 常见问题 ----
    st.markdown("#### ❓ 常见问题（FAQ）")
    faqs = [
        ("提示「尚未配置 API Key」怎么办？",
         "到顶部「设置」页选择服务商并填入 API Key，点击「测试连接」验证通过后保存。系统也支持环境变量 DEEPSEEK_API_KEY。"),
        ("提示「知识库为空」怎么办？",
         "到「知识库」页上传个人文档（PDF / TXT / MD / DOCX / XLSX / PPTX，可多选），点击「上传并重建知识库」。首次构建需要一些时间，请留意进度条与完成提示。"),
        ("回答为什么带【来源：文件名】标注？",
         "这是系统的防幻觉机制：每个基于资料的事实都会标注出处。若大模型个别句子漏标，系统会自动做一次「引用补全精修」把缺失的标注补上；您也可点击回答上方的资料卡片查看命中的原文切片核验。"),
        ("「检索」页的融合分/重排分是什么？",
         "融合分是 BM25 关键词检索与向量语义检索的加权融合得分，重排分是深度语义模型精排后的相关性得分。分数越高代表与问题越相关。"),
        ("上传时提示「检测到同名文件 / 旧版本」是什么意思？",
         "为防止误覆盖与新旧版混存：上传同名文件时会提示「将覆盖原文件」（可取消勾选跳过）；上传的文件若比知识库中已有的同主题文件更新（按文件名中的年份/v1.0/第X版识别），会完整列出旧版文件全称，您可选择「删除旧版」或「新旧版共存」后确认执行。"),
        ("知识库中同一文档存在新旧版本时，系统会选哪个？",
         "系统会自动识别文件名中的版本标识（年份如 GB/T 1234-2024、v1.0/v2.0、第X版、XXXX版 等），当同一主题存在多个版本时，检索会优先返回最新版本的内容；无法从文件名识别版本时，则以上传时间较新的文件为准。对话与检索结果中会显示版本徽章（如 v2.0、2024），回答的来源标注也会写明所用版本。也可在「设置 → 系统设置」开启「仅检索最新版本」。"),
        ("开启 Web 联网搜索后，回答会优先联网吗？",
         "不会。Agent 会先检索知识库：知识库已有且结果充足时只回答库内内容并标注【来源：文件名】；仅当知识库结果不足/过时/没有时才联网补充，并区分【来源：网页标题】。是否联网由大模型按规则自主决策。闲聊分支在需要最新外部信息时也可联网。"),
        ("上传的文档包含敏感信息会被外发吗？",
         "不会。文档解析、切片、向量化与检索全部在本地完成；仅生成回答时会把检索到的文本发送到云端大模型，且发送前会自动脱敏（手机号、身份证、银行卡、金额、统一社会信用代码、护照、车牌、邮箱、电话、IP 等）。"),
        ("PPTX 上传后为什么检索不到图片里的内容？",
         "PPTX 解析仅提取幻灯片中文字框与表格的文本。若幻灯片多为「图片 + 大字号标题」的版面，图片中的文字无法直接提取，建议将这类演示文稿转换为 PDF 后上传（PDF 会走 OCR 识别图片中的文字），查询效果更好。"),
        ("模型在哪里运行？需要安装什么？",
         "向量与重排模型（BGE 系列）随项目安装，首次运行时自动从国内镜像下载权重并缓存到本地，之后离线可用；大语言模型走您配置的云端 API。"),
        ("可以自定义助手的角色定位吗？",
         "可以。在「设置」→「角色定义」中填写 Agent 定位，例如“你是某领域的资深专家”，保存后生效。"),
    ]
    for q, a in faqs:
        with st.expander(q, expanded=False):
            st.markdown(a)


# ==========================================
# 检索页
# ==========================================
def _highlight(query: str, text: str) -> str:
    """用 jieba 分词对切片文本做关键词高亮（<mark>）。
    先替换为占位符再统一插入 <mark>，避免 token 与已插入标签相互污染。"""
    import jieba
    safe = html.escape(text)
    tokens = []
    seen = set()
    for t in jieba.cut(query):
        tok = html.escape(t.strip())
        if len(t.strip()) >= 2 and tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    for i, tok in enumerate(tokens):
        safe = safe.replace(tok, f"\uE000{i}\uE000")  # 私有区占位符
    for i, tok in enumerate(tokens):
        safe = safe.replace(f"\uE000{i}\uE000", f"<mark>{tok}</mark>")
    return safe


@st.fragment(run_every=2.0)
def render_search_loading(cached_retriever_loader):
    """🔍 检索页首次加载专用界面（run_every 定时片段）：
    首次进入时本函数被正常 run 渲染（占位页提交后，上一个页面的内容被清除），
    随后的定时重跑完成耗时的模型加载，避免加载期间旧页面残留可点击。"""
    st.markdown("### 🔍 检索")
    if st.session_state.get("search_model_ready", False):
        # 模型已就绪：立即重跑整个应用，切换到正常检索界面（并停止本定时片段）
        st.rerun()
        return
    if not st.session_state.get("search_loading", False):
        # 第一轮：只渲染纯占位页（无任何按钮），等待定时重跑执行加载
        st.session_state["search_loading"] = True
        st.markdown("### ⏳ 正在加载检索模型")
        st.info("首次加载约需 10-30 秒，加载完成后将自动进入检索界面，请稍候。")
        return
    # 后续轮次：真正执行耗时的模型加载
    with st.spinner("⏳ 正在加载检索模型（首次加载约需 10-30 秒）..."):
        retriever, reranker = cached_retriever_loader()
    st.session_state["search_model_ready"] = True
    st.session_state["search_loading"] = False
    st.rerun()


@st.fragment
def render_search_page(cached_retriever_loader):
    """🔍 检索面板：查看命中切片、分数、来源与关键词高亮（局部刷新，不重跑全页）。"""
    st.markdown("### 🔍 检索")
    st.caption("输入问题查看混合检索 + 重排的命中切片与分数，可调参即时对比。")

    with st.spinner("⏳ 正在加载检索模型（首次加载约需 10-30 秒）..."):
        retriever, reranker = cached_retriever_loader()
    if retriever is None:
        st.info("🚀 知识库为空，请先在顶部「知识库」页上传文档并构建。")
        return
    sources = retriever.list_sources() if hasattr(retriever, "list_sources") else []
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1.1])
    with c1:
        question = st.text_input("检索问题", key="dbg_query")
    with c2:
        default_topk = int(load_config(CONFIG_FILE).get("rerank_top_k", RERANK_TOP_K))
        top_k = st.slider("最相关条数", 1, 10, min(10, max(1, default_topk)), key="dbg_topk",
                          help="返回与问题最相关的前 N 个内容片段（即 Top-N，默认取自「设置 → 检索与切片设置」）。"
                               "越大上下文越全但越慢，一般 3-5 条即可覆盖大多数问题。")
    with c3:
        use_rerank = st.checkbox("使用重排", value=True, key="dbg_rerank",
                                 help="用深度语义模型（bge-reranker-v2-m3，本地推理）对初步召回的切片精排，"
                                      "相关性更准但耗时略增。关闭后仅按 BM25+向量融合分排序。")
    with c4:
        # 默认跟随设置页「仅检索最新版本」，可在此临时切换
        default_latest = load_config(CONFIG_FILE).get("latest_only", False)
        latest_only = st.checkbox("仅最新版本", value=default_latest, key="dbg_latest",
                                  help="同一文档存在新旧版本时，只保留最新版本的命中")
    source_filter = st.selectbox("来源过滤（可选）", ["全部"] + sources, key="dbg_source")
    # 显式「开始检索」按钮：降低新手使用门槛（回车同样可用）
    search_clicked = st.button("🔍 开始检索", type="primary", use_container_width=True)

    question = (question or "").strip()
    if not question:
        if search_clicked:
            st.warning("⚠️ 请先输入检索问题")
        else:
            st.info("输入问题后点击「开始检索」或按回车即可。")
        return

    # 思考过程：展示 混合召回 → 语义精排 → 完成 的检索链路（避免误以为卡住）
    with st.status("🔍 正在检索知识库...", expanded=True) as status:
        st.write("🔎 执行混合检索：BM25 关键词检索 + 向量语义检索（RRF 融合）...")

        def _on_stage(stage, **kw):
            if stage == "retrieved":
                st.write(f"📄 初步召回 {kw.get('count', 0)} 个文档切片")
            elif stage == "reranking":
                status.update(label="正在使用深度语义模型进行精排...", state="running")
            elif stage == "reranked":
                st.write(f"🧠 语义精排保留 Top-{kw.get('count', 0)}")
                status.update(label="检索完成", state="complete", expanded=False)

        # 相似度阈值跟随设置页「检索与切片设置」，默认 0（关闭）
        default_min_score = float(load_config(CONFIG_FILE).get("min_score", 0.0) or 0.0)
        hits = search_debug(
            question, retriever, reranker, top_k=top_k, use_rerank=use_rerank,
            source=None if source_filter == "全部" else source_filter,
            latest_only=latest_only,
            min_score=default_min_score,
            on_stage=_on_stage,
        )
        if not hits:
            status.update(label="未找到匹配内容", state="error", expanded=False)
        else:
            status.update(label=f"检索完成，共 {len(hits)} 条命中", state="complete", expanded=False)

    # 命中按来源文件聚合：一个文件一个卡片，内部列出全部命中切片（同文件切片合并）
    from collections import OrderedDict
    grouped = OrderedDict()
    for h in hits:
        grouped.setdefault(h.source, []).append(h)

    st.markdown(f"**共 {len(hits)} 条命中**，来自 {len(grouped)} 份文档")
    for src, items in grouped.items():
        _, ver_label, _ = parse_version(src)
        pages = "、".join(sorted({h.page for h in items}))
        title = f"📄 {src} · P{pages} · {len(items)} 处命中"
        if ver_label:
            title += f" · 📌 {ver_label}"
        with st.expander(title, expanded=False):
            for h in items:
                badges = [f'<span class="badge badge-fusion">融合分 {h.score:.4f}</span>']
                if h.rerank_score is not None:
                    badges.append(f'<span class="badge badge-rerank">重排分 {h.rerank_score:.4f}</span>')
                st.markdown('<div class="hit-badges">' + "".join(badges) + "</div>",
                            unsafe_allow_html=True)
                st.caption(f"#{h.rank} · 页码 P{h.page}")
                st.markdown(_highlight(question, h.content), unsafe_allow_html=True)
                st.divider()


# ==========================================
# 知识库管理页
# ==========================================
def _force_remove(path: str) -> None:
    """删除文件：先清除只读属性（部分来源文件带 ReadOnly，os.remove 会拒绝访问），再删除。"""
    if os.path.exists(path):
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            pass
        os.remove(path)


def render_kb_page(clear_retriever_cache):
    st.markdown("### 📚 知识库管理")
    doc_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if is_supported_file(f)]
    stats = get_kb_stats(CHROMA_DB_DIR)
    total_chunks = sum(stats.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("📄 文档数", len(doc_files))
    c2.metric("🧩 切片数", total_chunks)
    c3.metric("构建状态", "✅ 已构建" if total_chunks else "⏳ 未构建")

    def _run_rebuild(force_full: bool = False):
        # 先释放检索器持有的向量库文件句柄，避免 Windows 下重命名/清空目录时报"程序正在使用"
        clear_retriever_cache()
        gc.collect()
        t0 = time.time()
        overlay = st.empty()
        with overlay.container():
            st.warning("⏳ 知识库重建中，请稍候...")
            progress_bar = st.progress(0, text="开始处理...")
        chunk_count, failed_files, file_chunks = rebuild_vector_store_sync(
            KNOWLEDGE_BASE_DIR, CHROMA_DB_DIR, progress_bar, force_full=force_full)
        overlay.empty()
        elapsed = time.time() - t0
        if chunk_count > 0:
            st.success(f"✅ 成功构建 {chunk_count} 个文本块（耗时 {elapsed:.0f} 秒）")
            with st.expander("📋 各文件处理明细", expanded=True):
                for fname, (parser, n) in sorted(file_chunks.items()):
                    st.markdown(f"- 📄 **{fname}**：解析引擎 `{parser}`，生成 {n} 个切片")
                if failed_files:
                    st.markdown("**❌ 处理失败：**")
                    for fname, err in failed_files:
                        st.markdown(f"- ⚠️ {fname}：{err}")
            clear_retriever_cache()
            st.toast("✅ 检索已切换到最新知识库，可直接开始提问", icon="✅")
            time.sleep(1)
            st.rerun()
        else:
            # 知识库目录已为空（如删除全部文件后重建）→ 视为成功清空，而非失败
            if not [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if is_supported_file(f)]:
                clear_retriever_cache()
                st.success("✅ 知识库已清空（当前无任何文档），可上传新文件构建")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ 更新失败{('：' + '；'.join(f'{f}:{e}' for f, e in failed_files)) if failed_files else ''}")

    st.markdown("#### 上传与构建")
    uploaded_files = st.file_uploader(
        "上传文档（支持 PDF / TXT / MD / DOCX / XLSX / PPTX）",
        type=["pdf", "txt", "md", "docx", "xlsx", "pptx"],
        accept_multiple_files=True,
        help="上传时会检测同名文件与旧版本文件，确认后再执行；上传后仅重建变更文件（增量构建），速度更快。",
    )
    st.caption(f"💡 提示：单文件建议不超过 {MAX_UPLOAD_MB} MB（超大文件解析耗时、内存占用高，"
               f"超过 {MAX_UPLOAD_MB} MB 的文件会被拦截）。"
               "PPTX 仅提取文字框与表格内容；若幻灯片多为「图片 + 大字号标题」的版面（文字密度低），"
               "图片中的文字无法提取，建议转换为 PDF 后上传，查询效果更好。")

    # ---- 上传前检测：同名文件 / 旧版本文件（文件名级） ----
    def _detect_upload_conflicts(files):
        """检测上传文件与知识库已有文件的冲突。
        返回 (同名文件列表, 版本升级列表)。版本升级：新文件主题与库内文件相同且版本号更新，
        每项含新文件及对应的全部旧版文件全称（供用户选择删除或保留）。"""
        existing = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if is_supported_file(f)]
        same_name = [uf for uf in files if os.path.exists(os.path.join(KNOWLEDGE_BASE_DIR, uf.name))]
        same_set = {uf.name for uf in same_name}
        upgrades = []
        for uf in files:
            if uf.name in same_set:
                continue  # 同名文件走"覆盖/跳过"逻辑，不算版本升级
            new_topic, new_label, new_key = parse_version(uf.name)
            old_files = []
            for old_name in existing:
                if old_name == uf.name:
                    continue
                old_topic, old_label, old_key = parse_version(old_name)
                if old_topic == new_topic and new_key > old_key:
                    old_files.append({"name": old_name, "label": old_label or "无版本号"})
            if old_files:
                upgrades.append({"new_file": uf, "new_label": new_label or "新版", "old_files": old_files})
        return same_name, upgrades

    def _save_uploads(files, skip_names=(), delete_old_names=()):
        """按用户决策保存上传文件：跳过列表内的文件、删除指定的旧版文件。返回失败列表。"""
        failed = []
        for uf in files:
            if uf.name in skip_names:
                continue
            target = os.path.join(KNOWLEDGE_BASE_DIR, uf.name)
            try:
                # 覆盖同名文件：先清除只读属性，避免 open("wb") 拒绝访问
                if os.path.exists(target):
                    os.chmod(target, stat.S_IWRITE)
                with open(target, "wb") as f:
                    f.write(uf.getbuffer())
            except OSError as e:
                failed.append((uf.name, str(e)))
        for old in delete_old_names:
            try:
                _force_remove(os.path.join(KNOWLEDGE_BASE_DIR, old))
            except OSError as e:
                failed.append((old, f"旧版删除失败：{e}"))
        return failed

    def _scan_and_rebuild(saved_files):
        """敏感信息扫描提示（仅本地统计，不发送任何数据）后重建知识库。"""
        hit_files = 0
        for uf in saved_files:
            path = os.path.join(KNOWLEDGE_BASE_DIR, uf.name)
            page_texts, _, _ = parse_document(path)
            raw_text = "\n".join(t for texts in page_texts.values() for t in texts)
            result, total = scan_sensitive_info(raw_text)
            if total:
                hit_files += 1
                detail = "、".join(f"{v['name']}×{v['count']}" for v in result.values())
                st.warning(f"⚠️ {uf.name}：检测到敏感信息（{detail}），发送云端前将自动脱敏")
        if hit_files == 0:
            st.info("✅ 本次上传的文件未检测到明显敏感信息")
        _run_rebuild()

    c1, c2 = st.columns(2)
    with c1:
        if uploaded_files and st.button("🔄 上传并重建知识库", type="primary", use_container_width=True):
            # 上传前尺寸校验：超大文件解析耗时/内存高，直接拦截并给出明确提示
            size_limit = MAX_UPLOAD_MB * 1024 * 1024
            oversized = [uf for uf in uploaded_files if uf.size and uf.size > size_limit]
            if oversized:
                detail = "\n".join(f"- {uf.name}（{uf.size / 1024 / 1024:.1f} MB）" for uf in oversized)
                st.error(f"❌ 以下文件超过单文件上限 {MAX_UPLOAD_MB} MB，未进入上传流程：\n"
                         f"{detail}\n\n请压缩或拆分后重试（建议不超过 {MAX_UPLOAD_MB} MB）。")
                st.stop()
            same_name, upgrades = _detect_upload_conflicts(uploaded_files)
            if same_name or upgrades:
                # 检测到同名/旧版 → 先进入确认面板，不立即保存
                st.session_state.kb_pending_upload = {
                    "files": list(uploaded_files), "same_name": same_name, "upgrades": upgrades}
                st.rerun()
            else:
                failed = _save_uploads(uploaded_files)
                if failed:
                    st.error("❌ 部分文件保存失败（文件可能被其他程序占用）：\n"
                             + "\n".join(f"- {f}：{e}" for f, e in failed)
                             + "\n\n请关闭占用该文件的程序后重试。")
                    st.stop()
                _scan_and_rebuild(uploaded_files)
    with c2:
        if doc_files and st.button("🔄 重建知识库（使用现有文件）", use_container_width=True,
                                   help="忽略增量缓存，对全部文件重新解析、切片并向量化（速度较慢）"):
            _run_rebuild(force_full=True)

    # ---- 上传确认面板（检测到同名/旧版时展示，等待用户决策） ----
    pending = st.session_state.get("kb_pending_upload")
    if pending:
        with st.container(border=True):
            st.markdown("#### ⚠️ 上传前确认")
            if pending["same_name"]:
                st.warning("检测到**同名文件**（上传将覆盖原文件）：\n"
                           + "\n".join(f"- {uf.name}" for uf in pending["same_name"]))
                overwrite = st.checkbox("覆盖同名文件（取消勾选则跳过这些文件，不覆盖）",
                                        value=True, key="kb_dup_overwrite")
            else:
                overwrite = True
            if pending["upgrades"]:
                for g in pending["upgrades"]:
                    old_names = "、".join(f"{o['name']}（{o['label']}）" for o in g["old_files"])
                    st.warning(f"📌 检测到**旧版本**：新上传 **{g['new_file'].name}**（{g['new_label']}）\n\n"
                               f"→ 知识库已有旧版：**{old_names}**")
                del_old = st.checkbox("删除以上旧版文件（取消勾选则新旧版共存入库）", value=True, key="kb_del_old")
            else:
                del_old = False
            c1, c2 = st.columns(2)
            if c1.button("✅ 确认上传并重建", type="primary", use_container_width=True):
                skip_names = [] if overwrite else [uf.name for uf in pending["same_name"]]
                delete_old = [o["name"] for g in pending["upgrades"] for o in g["old_files"]] if del_old else []
                failed = _save_uploads(pending["files"], skip_names, delete_old)
                st.session_state.kb_pending_upload = None
                if failed:
                    st.error("❌ 部分文件保存/删除失败（文件可能被其他程序占用）：\n"
                             + "\n".join(f"- {f}：{e}" for f, e in failed)
                             + "\n\n请关闭占用该程序的文件后重试。")
                    st.stop()
                saved = [uf for uf in pending["files"] if uf.name not in skip_names]
                if not saved:
                    st.warning("⚠️ 所有文件均被跳过，未执行上传与重建。")
                    st.rerun()
                _scan_and_rebuild(saved)
            if c2.button("✖ 取消", use_container_width=True):
                st.session_state.kb_pending_upload = None
                st.rerun()

    st.markdown("#### 当前知识库文件")
    if doc_files:
        with st.expander(f"📋 共 {len(doc_files)} 份文档（勾选可删除）", expanded=True):

            def _on_sel_all_change():
                """勾选/取消全选时，联动设置所有文件的勾选状态。"""
                v = st.session_state["kb_sel_all"]
                for f in doc_files:
                    st.session_state[f"kb_del_{f}"] = v

            def _on_file_change():
                """任一文件被取消勾选时，自动取消全选，保持状态一致。"""
                if st.session_state.get("kb_sel_all") and any(
                        not st.session_state.get(f"kb_del_{f}", False) for f in doc_files):
                    st.session_state["kb_sel_all"] = False

            # 全选：勾选后删除全部文件并清空知识库
            select_all = st.checkbox("☑ 全选（删除全部文件并清空知识库）", key="kb_sel_all",
                                     help="勾选全部文件，可一键删除知识库中所有文档并清空向量索引",
                                     on_change=_on_sel_all_change)
            sel = []
            for f in sorted(doc_files):
                mtime = datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(KNOWLEDGE_BASE_DIR, f))
                ).strftime("%Y-%m-%d %H:%M")
                checked = st.checkbox(
                    f"📄 **{f}**（{stats.get(f, 0)} 个切片 · 上传于 {mtime}）",
                    key=f"kb_del_{f}",
                    value=False,
                    on_change=_on_file_change,
                )
                if checked:
                    sel.append(f)

            c1, c2 = st.columns(2)
            with c1:
                del_btn = st.button(f"🗑️ 删除选中（{len(sel)} 份）并重建", use_container_width=True,
                                    disabled=not sel)
            with c2:
                clear_btn = st.button("🧹 删除全部并清空", use_container_width=True)

            if del_btn:
                st.session_state.kb_pending_del = sorted(sel)
            if clear_btn:
                st.session_state.kb_pending_clear = True

            # 二次确认：删除选中文件
            if st.session_state.get("kb_pending_del"):
                pending = st.session_state.kb_pending_del
                st.warning("⚠️ 即将删除以下文件并重建知识库：\n" + "\n".join(f"- {f}" for f in pending))
                c1, c2 = st.columns(2)
                if c1.button("✅ 确认删除并重建", type="primary", use_container_width=True):
                    failed = []
                    for f in pending:
                        try:
                            _force_remove(os.path.join(KNOWLEDGE_BASE_DIR, f))
                        except OSError as e:
                            failed.append((f, str(e)))
                    st.session_state.kb_pending_del = None
                    if failed:
                        st.error("❌ 部分文件删除失败（文件可能被其他程序占用）：\n"
                                 + "\n".join(f"- {f}：{e}" for f, e in failed)
                                 + "\n\n请关闭占用该文件的程序后重试。")
                        st.stop()
                    _run_rebuild()
                if c2.button("✖ 取消", use_container_width=True):
                    st.session_state.kb_pending_del = None
                    st.rerun()

            # 二次确认：清空全部
            if st.session_state.get("kb_pending_clear"):
                st.warning("⚠️ 将删除知识库**全部文件**并清空向量索引，此操作不可恢复！")
                c1, c2 = st.columns(2)
                if c1.button("⚠️ 确认清空", type="primary", use_container_width=True):
                    import shutil
                    # 先释放检索器持有的向量库文件句柄，确保目录可被完整删除
                    # （避免 Windows 下句柄占用导致部分删除残留 → 空库被误判为非空 → 查询异常）
                    clear_retriever_cache()
                    gc.collect()
                    failed = []
                    for f in doc_files:
                        try:
                            _force_remove(os.path.join(KNOWLEDGE_BASE_DIR, f))
                        except OSError as e:
                            failed.append((f, str(e)))
                    if failed:
                        st.error("❌ 部分文件删除失败（文件可能被其他程序占用）：\n"
                                 + "\n".join(f"- {f}：{e}" for f, e in failed)
                                 + "\n\n请关闭占用该文件的程序后重试。")
                        st.stop()
                    shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)  # 清空向量库与 BM25 缓存
                    st.session_state.kb_pending_clear = None
                    clear_retriever_cache()
                    st.toast("🧹 知识库已清空，可重新上传文件", icon="🧹")
                    time.sleep(1)
                    st.rerun()
                if c2.button("✖ 取消清空", use_container_width=True):
                    st.session_state.kb_pending_clear = None
                    st.rerun()
    else:
        st.info("知识库为空，请上传文档（PDF / TXT / MD / DOCX / XLSX / PPTX）并构建。")


# ==========================================
# 设置页
# ==========================================
_PROVIDER_DEFAULTS = {
    "DeepSeek": {"base": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "OpenAI": {"base": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "智谱AI (Zhipu)": {"base": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    "Ollama (本地)": {"base": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
    "阿里云百炼 (Qwen)": {"base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "Moonshot (Kimi)": {"base": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "SiliconFlow (硅基流动)": {"base": "https://api.siliconflow.cn/v1", "model": "deepseek-ai/DeepSeek-V3"},
    "OpenRouter": {"base": "https://openrouter.ai/api/v1", "model": "anthropic/claude-3.5-sonnet"},
}


def _parse_optional_number(raw, label: str, vmin=None, vmax=None, is_int: bool = False):
    """解析可选的数字输入：空串/None → None（= 使用服务商默认）；非法或越界 → 抛 ValueError。"""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        val = int(raw) if is_int else float(raw)
    except ValueError:
        raise ValueError(f"{label} 必须是数字")
    if vmin is not None and val < vmin:
        raise ValueError(f"{label} 不能小于 {vmin}")
    if vmax is not None and val > vmax:
        raise ValueError(f"{label} 不能大于 {vmax}")
    return val


def render_settings_page(config):
    st.markdown("### ⚙️ 设置")
    st.caption("模型配置、系统设置与 Agent 角色定义。")

    # ---- 模型配置 ----
    with st.expander("🤖 模型配置", expanded=True):
        st.markdown('<div class="set-req-hint">⚠️ 带 <span class="req">*</span> 的为<b>必填项</b>；'
                    '未标 * 的项（高级参数）<b>留空即用服务商默认最优配置</b>，一般无需修改。</div>',
                    unsafe_allow_html=True)
        provider_options = ["DeepSeek", "OpenAI", "智谱AI (Zhipu)", "Ollama (本地)",
                            "阿里云百炼 (Qwen)", "Moonshot (Kimi)", "SiliconFlow (硅基流动)",
                            "OpenRouter", "自定义"]
        default_provider = config.get("provider", "DeepSeek")
        if default_provider not in provider_options:
            default_provider = "自定义"
        provider = st.selectbox("服务商 *", options=provider_options, key="set_provider",
                                index=provider_options.index(default_provider),
                                help="选择大模型服务商。Ollama 为本地部署，无需联网；其余为云端 API。")

        if provider == "自定义":
            base_url = st.text_input("自定义 Base URL *", value=config.get("custom_base", ""), key="set_custom_base",
                                     help="任意 OpenAI 兼容 API 的地址（以 http:// 或 https:// 开头）。")
            model_name = st.text_input("自定义 Model Name *", value=config.get("custom_model", ""), key="set_custom_model",
                                       help="服务商支持的模型名，如 gpt-4o。")
        else:
            base_url = st.text_input("Base URL *", value=config.get("base_url", _PROVIDER_DEFAULTS[provider]["base"]),
                                     key="set_base_url")
            model_name = st.text_input("Model Name *",
                                       value=config.get("model_name", _PROVIDER_DEFAULTS[provider]["model"]),
                                       key="set_model_name")

        default_api_key = config.get("api_key", "") if config.get("remember_api_key", False) else ""
        is_local = provider == "Ollama (本地)"
        api_key = st.text_input("API Key *", type="password", value=default_api_key, key="set_api_key",
                                help=("Ollama 本地服务通常无需鉴权，可填写任意占位符（如 ollama）；"
                                      "云端服务商请填写真实的 API Key，Key 仅保存在本机。"
                                      if is_local else
                                      "必填。填写服务商提供的 API Key；勾选下方「记住」后加密保存在本机（仅当前用户可解密），"
                                      "不勾选则每次需重新填写。"))
        remember_api_key = st.checkbox("记住 API Key (本地保存)", value=config.get("remember_api_key", False),
                                       key="set_remember_api_key")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0,
                                value=config.get("temperature", 0.3), step=0.05, key="set_temperature")

        # ---- 高级采样参数（RAGFlow 风格：必填项在上方，此处留空即用服务商默认） ----
        st.markdown('<div class="set-sec-title">🔧 高级参数（<b>留空即用服务商默认最优配置</b>）</div>',
                    unsafe_allow_html=True)
        max_tokens = st.number_input("最大输出 Token（Max Tokens）", min_value=64, max_value=32768,
                                     step=128, value=int(config.get("max_tokens") or 2048), key="set_max_tokens",
                                     help="单次回答最大输出 Token 数。")
        c1, c2 = st.columns(2)
        with c1:
            top_p = st.text_input("Top P（核采样）", value=str(config.get("top_p") or ""), key="set_top_p",
                                  help="留空 = 服务商默认。只保留累积概率达到该值的候选词（0~1）。")
            frequency_penalty = st.text_input("Frequency Penalty（频率惩罚）",
                                              value=str(config.get("frequency_penalty") or ""),
                                              key="set_freq_pen",
                                              help="留空 = 服务商默认。范围 -2~2，越大越抑制重复用词。")
        with c2:
            top_k = st.text_input("Top K（候选采样）", value=str(config.get("top_k") or ""), key="set_top_k",
                                  help="留空 = 服务商默认。仅部分服务商支持（如 DeepSeek / Qwen）。")
            presence_penalty = st.text_input("Presence Penalty（存在惩罚）",
                                             value=str(config.get("presence_penalty") or ""),
                                             key="set_pres_pen",
                                             help="留空 = 服务商默认。范围 -2~2，越大越鼓励引入新话题。")
        max_context = st.text_input("最大上下文长度（Context Length）",
                                    value=str(config.get("max_context_length") or ""), key="set_max_ctx",
                                    help="留空 = 服务商默认。用于限制发送给模型的历史对话 Token 数，"
                                         "防止超出模型上下文窗口（历史截断上限）。")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 保存配置", use_container_width=True):
                valid, err = validate_api_config(provider, api_key, base_url, model_name)
                if not valid:
                    st.error(f"❌ {err}")
                else:
                    # 高级参数：留空 → None（使用服务商默认）；非法/越界 → 报错阻止保存
                    try:
                        opt_vals = {
                            "max_tokens": int(max_tokens),
                            "top_p": _parse_optional_number(top_p, "Top P", 0.0, 1.0),
                            "top_k": _parse_optional_number(top_k, "Top K", 1, 200, is_int=True),
                            "frequency_penalty": _parse_optional_number(frequency_penalty,
                                                                        "Frequency Penalty", -2.0, 2.0),
                            "presence_penalty": _parse_optional_number(presence_penalty,
                                                                       "Presence Penalty", -2.0, 2.0),
                            "max_context_length": _parse_optional_number(max_context,
                                                                         "最大上下文长度", 1000, 1000000, is_int=True),
                        }
                    except ValueError as e:
                        st.error(f"❌ {e}")
                        st.stop()
                    cfg_to_save = {
                        "provider": provider, "temperature": temperature, "remember_api_key": remember_api_key,
                        "base_url": base_url if provider != "自定义" else "",
                        "model_name": model_name if provider != "自定义" else "",
                        "custom_base": base_url if provider == "自定义" else "",
                        "custom_model": model_name if provider == "自定义" else "",
                        "api_key": api_key if remember_api_key else "",
                        **opt_vals,
                    }
                    save_config(CONFIG_FILE, cfg_to_save)
                    st.success("✅ 配置已保存")
                    time.sleep(0.5)
                    st.rerun()
        with c2:
            if st.button("🔌 测试连接", use_container_width=True):
                valid, err = validate_api_config(provider, api_key, base_url, model_name)
                if not valid:
                    st.error(f"❌ {err}")
                else:
                    with st.spinner("测试中..."):
                        res = test_llm_connection(api_key, base_url, model_name, temperature)
                        if res["success"]:
                            st.success(f"✅ {res['message']}")
                        else:
                            st.error(f"❌ {res['error']}")

    # ---- 系统设置（Web 联网搜索 + 检索策略，独立区块，与 API 配置分离） ----
    with st.expander("⚙️ 系统设置", expanded=False):
        def _on_web_toggle():
            """开关变更时立即持久化，避免与模型配置保存按钮耦合。"""
            cfg = load_config(CONFIG_FILE)
            cfg["web_search_enabled"] = bool(st.session_state.get("set_web_search", False))
            save_config(CONFIG_FILE, cfg)

        def _on_latest_toggle():
            cfg = load_config(CONFIG_FILE)
            cfg["latest_only"] = bool(st.session_state.get("set_latest_only", False))
            save_config(CONFIG_FILE, cfg)

        web_enabled = st.checkbox(
            "🌐 启用 Web 联网搜索（Agent 可搜索互联网补充回答）",
            value=config.get("web_search_enabled", False), key="set_web_search",
            on_change=_on_web_toggle,
            help="开启后，Agent 遇到知识库中没有的最新资讯/公开资料时会自动调用必应搜索（免 Key）；"
                 "闲聊分支在需要最新外部信息时也可联网。\n"
                 "注意：联网搜索会把您的查询词发送给搜索引擎，请按需开启；发送前内容仍会本地脱敏。")
        if web_enabled:
            st.caption("💡 已开启：对话页会显示联网提示；询问知识库外的新资讯（如行业动态、最新政策）时，"
                       "Agent 将自动搜索互联网。关闭此开关后立即恢复纯本地问答。")
        else:
            st.caption("💡 默认关闭：Agent 仅回答本地知识库内容，不访问互联网，隐私最佳。")

        st.markdown('<div class="set-sec-title">🔎 检索策略</div>', unsafe_allow_html=True)
        st.checkbox("仅检索最新版本（同主题多版本时只用最新版文件）",
                    value=config.get("latest_only", False), key="set_latest_only",
                    on_change=_on_latest_toggle,
                    help="知识库同时存在新旧版本（如执行标准更新）时，对话与 Agent 检索只采用最新版本文件的内容")
        st.caption("💡 开启后，同主题多版本的文档只以最新版为准参与检索；关闭则新旧版本可同时参与。")

    # ---- 检索与切片设置 ----
    with st.expander("🧩 检索与切片设置", expanded=False):
        st.caption("「最相关条数」即时生效无需重建；切片参数与向量模型保存后需到「知识库」页重建生效（参数变更会自动转为全量重建）。")
        chunk_cfg = get_chunk_config()
        emb_model = get_embedding_model()
        rerank_top_k = st.number_input("最相关条数（对话 / 检索 / Agent 共用）", min_value=1, max_value=20,
                                       value=int(config.get("rerank_top_k", RERANK_TOP_K)),
                                       key="set_rerank_top_k",
                                       help="生成回答时送入大模型的与问题最相关的内容片段数量（即 Top-N）。"
                                            "越大上下文越全但越慢、噪音越多，一般 3-5 条即可。即时生效，无需重建。")
        min_score = st.number_input("最低相关分（重排后过滤，0=关闭）", min_value=0.0, max_value=1.0,
                                    step=0.05, value=float(config.get("min_score", 0.0) or 0.0),
                                    key="set_min_score",
                                    help="重排后仅保留相关分 ≥ 该值的片段，过滤低质量内容（低于阈值的片段不进回答）。"
                                         "0 表示关闭（不过滤）。不同模型的分数范围不同，建议从 0.3-0.5 起步试调。"
                                         "即时生效，无需重建。")
        chunk_size = st.number_input("切片大小（字符）", min_value=100, max_value=2000,
                                     step=50, value=chunk_cfg["chunk_size"], key="set_chunk_size",
                                     help="每个文本块的目标长度。越大上下文越完整，越小检索越精准。")
        chunk_strategy = st.selectbox("切片策略", options=["sentence", "heading", "fixed"],
                                      index=["sentence", "heading", "fixed"].index(chunk_cfg["chunk_strategy"]),
                                      key="set_chunk_strategy",
                                      format_func=lambda s: {"sentence": "句级感知（推荐，句子不截断）",
                                                             "heading": "标题感知（按 Markdown 标题切分）",
                                                             "fixed": "固定长度（递归切分）"}[s],
                                      help="sentence：按句号/问号等断句，句子永不截断；"
                                           "heading：先按标题分块再补切；fixed：纯按字符数递归切分。")
        _EMB_OPTIONS = {
            "BAAI/bge-small-zh-v1.5": "bge-small-zh-v1.5（默认 · 快）",
            "BAAI/bge-base-zh-v1.5": "bge-base-zh-v1.5（更准 · 较慢）",
            "BAAI/bge-large-zh-v1.5": "bge-large-zh-v1.5（最准 · 最慢）",
        }
        emb_options = list(_EMB_OPTIONS)
        emb_default = emb_model if emb_model in emb_options else emb_options[0]
        embedding_model = st.selectbox("向量模型（Embedding）", options=emb_options,
                                       format_func=lambda m: _EMB_OPTIONS[m],
                                       index=emb_options.index(emb_default), key="set_emb_model",
                                       help="用于把文档切片转成向量的本地模型。切换后需全量重建知识库，"
                                            "首次使用新模型会联网下载权重并缓存到本地。")
        if st.button("💾 保存检索设置", use_container_width=True):
            cfg_to_save = load_config(CONFIG_FILE)
            cfg_to_save["chunk_size"] = int(chunk_size)
            cfg_to_save["chunk_strategy"] = chunk_strategy
            cfg_to_save["embedding_model"] = embedding_model
            cfg_to_save["rerank_top_k"] = int(rerank_top_k)
            cfg_to_save["min_score"] = float(min_score)
            save_config(CONFIG_FILE, cfg_to_save)
            st.success("✅ 检索设置已保存：最相关条数即时生效；切片/向量模型到「知识库」页重建后生效")
            time.sleep(0.5)
            st.rerun()

    # ---- 角色定义 ----
    with st.expander("🧠 角色定义", expanded=False):
        agent_role = st.text_area("Agent 定位", value=config.get("agent_role", ""), height=120, key="set_agent_role")
        if st.button("💾 保存角色"):
            config["agent_role"] = agent_role
            save_config(CONFIG_FILE, config)
            st.success("✅ 已保存")
            time.sleep(0.5)
            st.rerun()


# ==========================================
# 对话页（含编排状态面板与富内容回显）
# ==========================================
def _paint_stage(status, stage, **kw):
    """把编排器上报的阶段事件映射到状态面板展示。"""
    if stage == "thinking":
        status.update(label="🤖 Agent 思考中，正在决策是否调用工具...", state="running")
    elif stage == "tool_call":
        name = kw.get("name", "")
        args = kw.get("args") or {}
        st.write(f"🛠️ 调用工具：`{name}` 参数 `{json.dumps(args, ensure_ascii=False)}`")
        status.update(label=f"正在执行工具 {name}...", state="running")
    elif stage == "tool_result":
        st.write(f"✅ 工具 `{kw.get('name')}` 执行完成")
    elif stage == "retrieved":
        st.write(f"📄 初步召回了 {kw.get('count', 0)} 个文档切片...")
        status.update(label="正在匹配相关文档...", state="running")
    elif stage == "reranked":
        status.update(label="正在使用深度语义模型进行精排...", state="running")
        st.write(f"🧠 精排完成，保留 Top-{kw.get('count', 0)} 篇...")
    elif stage == "generating":
        status.update(label="正在生成回答...", state="complete", expanded=False)


def _messages_to_markdown(messages) -> str:
    """把消息列表转为 Markdown 文本（供会话导出下载）。"""
    lines = ["# 对话记录\n"]
    for m in messages:
        role = "👤 用户" if m.get("role") == "user" else "🤖 助手"
        lines.append(f"## {role}\n\n{m.get('content', '')}\n")
    return "\n".join(lines)


def render_chat_page(orchestrator, config, has_api_config, is_kb_empty, cached_retriever_loader):
    """💬 对话页：意图路由 → 检索 → 生成 的完整交互流程。"""
    if not has_api_config:
        st.warning("⚠️ 尚未配置 API Key，请先到顶部「设置」页配置。")
    if is_kb_empty:
        st.info("🚀 知识库为空，请先到顶部「知识库」页上传文档并构建。")
        st.stop()

    # Web 联网搜索提示：仅在设置页开启后展示，提醒用户当前为"本地 + 联网"混合模式
    if config.get("web_search_enabled", False):
        st.caption("🌐 <b>Web 联网搜索已开启</b>：询问知识库外的最新资讯/公开资料时，"
                   "Agent 会自动搜索互联网补充；关闭入口在「设置 → 系统设置」。（发送云端内容仍会本地脱敏）",
                   unsafe_allow_html=True)

    # ==============================
    # 处理重新生成请求（在显示消息之前）
    # ==============================
    if st.session_state.regenerate_pending:
        idx = st.session_state.regenerate_target_idx
        if 0 <= idx < len(st.session_state.messages) and st.session_state.messages[idx]["role"] == "assistant":
            # 删除该助手消息及其 token count（用户消息保留在 idx-1）
            del st.session_state.messages[idx]
            del st.session_state.messages_token_counts[idx]
            user_idx = idx - 1
            if user_idx >= 0 and st.session_state.messages[user_idx]["role"] == "user":
                st.session_state.regenerate_question = st.session_state.messages[user_idx]["content"]
                st.session_state.regenerate_pending = False

    # ==============================
    # 对话展示容器（顶部带当前会话导出）
    # ==============================
    if st.session_state.messages:
        top_c1, top_c2 = st.columns([6, 1], vertical_alignment="center")
        with top_c2:
            st.download_button(
                "📤 导出对话",
                data=_messages_to_markdown(st.session_state.messages),
                file_name=f"对话导出_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
                help="把当前会话导出为 Markdown 文件（含用户与助手内容）。",
            )

    chat_container = st.container()
    with chat_container:
        render_messages(st.session_state.messages)

        # 空会话欢迎卡片 + 快捷提问（仅初始欢迎语时显示）
        if len(st.session_state.messages) <= 1:
            st.markdown(
                '<div class="welcome-card">'
                '<div class="wc-icon">🧠</div>'
                '<div class="wc-title">AI 专家助手</div>'
                '<div class="wc-sub">私有知识库 · Agentic RAG —— 输入问题，或直接点击下方快捷提问</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            s1, s2, s3 = st.columns(3)
            for col, q in zip((s1, s2, s3), (
                    "知识库里包含哪些内容？",
                    "帮我检索相关文档并总结要点",
                    "介绍一下 Agentic RAG 的工作方式")):
                with col:
                    if st.button(q, key=f"sugg_{q}", use_container_width=True):
                        st.session_state.auto_question = q
                        st.rerun()

    # ==============================
    # 捕获用户输入（或重新生成的问题 / 快捷提问）
    # ==============================
    auto_q = st.session_state.pop("auto_question", None)
    pending_question = st.session_state.get("regenerate_question", None)
    if auto_q:
        prompt_user = auto_q
        add_user_msg = True
    elif pending_question:
        prompt_user = pending_question
        st.session_state.regenerate_question = None
        add_user_msg = False
    else:
        prompt_user = st.chat_input("请输入您的问题...")
        add_user_msg = True

    if not prompt_user:
        return

    if add_user_msg:
        add_message("user", prompt_user)

    role_def = st.session_state.get("agent_role", config.get("agent_role", ""))
    history_msg = st.session_state.messages[:-1]
    history_token_counts = st.session_state.messages_token_counts[:-1]

    with chat_container:
        with st.chat_message("assistant"):
            try:
                # ==============================
                # Agent 编排：快速通道闲聊 / function calling 工具决策
                # 思考轨迹 trace 同时用于状态面板展示与持久化回显
                # ==============================
                trace = []

                def _on_stage(stage, **kw):
                    trace.append((stage, kw))
                    _paint_stage(status, stage, **kw)

                with st.status("🕵️ 正在分析提问意图...", expanded=True) as status:
                    st.write("⚡ 本地快速通道识别意图...")
                    intent = orchestrator.detect(prompt_user)
                    st.write(f"🎯 意图初判：**{intent}**（CHAT=闲聊直达，AGENTIC=进入 Agent 工具编排）")

                    if intent == "CHAT":
                        status.update(label="命中 [快速通道] 闲聊，无需调用工具", state="complete", expanded=False)
                        st.write("💬 准备进行日常交互...")
                        result = orchestrator.answer(
                            prompt_user, intent, history_msg, role_def, None, None, history_token_counts,
                            on_stage=_on_stage,
                        )
                    else:
                        status.update(label="🤖 正在唤醒 Agent 编排（工具决策）...", state="running")
                        with st.spinner("⏳ 正在加载检索模型（首次加载约需 10-30 秒）..."):
                            retriever, reranker = cached_retriever_loader()
                        if retriever is None:
                            st.error("无法加载知识库组件，请检查后台日志。")
                            st.stop()
                        result = orchestrator.answer(
                            prompt_user, intent, history_msg, role_def,
                            retriever, reranker, history_token_counts,
                            on_stage=_on_stage,
                        )
                        status.update(label="回答已生成", state="complete", expanded=False)

                # ==============================
                # 输出展示（RAG/AGENTIC/CHAT 逐字渲染，GLOBAL 直接入库）
                # ==============================
                if result.intent in ("RAG", "AGENTIC", "CHAT"):
                    # 先逐字渲染正文（来源角标化），再折叠展示参考资料（RAGFlow 风格）
                    placeholder = st.empty()
                    displayed_text = ""
                    for ch in result.answer:
                        displayed_text += ch
                        placeholder.markdown(
                            style_inline_sources(displayed_text, result.sources) + "▌",
                            unsafe_allow_html=True)
                    placeholder.markdown(
                        style_inline_sources(displayed_text, result.sources),
                        unsafe_allow_html=True)
                    render_source_cards(result.sources, result.source_chunks, key_prefix="live")

                add_message("assistant", result.response, result.token_usage,
                            extra={"sources": result.sources,
                                   "source_chunks": result.source_chunks,
                                   "thinking": trace})

                st.session_state.all_sessions[st.session_state.current_session_id].update({
                    "messages": st.session_state.messages.copy(),
                    "title": generate_session_title(st.session_state.messages),
                    "updated_at": datetime.now().isoformat(),
                })
                save_session(SESSIONS_DIR, st.session_state.current_session_id, st.session_state.messages)
                st.rerun()

            except Exception as e:
                st.error(f"❌ 出现异常：{e}")
