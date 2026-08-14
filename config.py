# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# config.py
# 集中管理所有程序配置（路径、模型参数、切分参数等）

import json
import os

# 国内网络镜像：必须在 huggingface_hub 被导入【之前】设置，
# 否则其 ENDPOINT 常量已在导入时固化，此处设置将失效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ==================== 目录路径 ====================
SESSIONS_DIR = "./chat_sessions"
KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_DB_DIR = "./chroma_db"
CONFIG_FILE = "./config.json"

# ==================== 文本分块参数 ====================
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
# 切片策略：sentence=句级感知（推荐，句子永不截断）/ heading=标题感知 / fixed=固定长度
CHUNK_STRATEGY = "sentence"
TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# ==================== 向量模型 ====================
# 本地 embedding 模型（FastEmbed，支持 BAAI/bge-{small,base,large}-zh-v1.5）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def get_embedding_model() -> str:
    """从 config.json 读取用户配置的 embedding 模型，未配置时用默认值。"""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("embedding_model", EMBEDDING_MODEL)
    except Exception:
        return EMBEDDING_MODEL


def get_chunk_config() -> dict:
    """从 config.json 读取切片参数（size/strategy），未配置时用默认值。"""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "chunk_size": int(cfg.get("chunk_size", CHUNK_SIZE)),
            "chunk_strategy": cfg.get("chunk_strategy", CHUNK_STRATEGY),
        }
    except Exception:
        return {"chunk_size": CHUNK_SIZE, "chunk_strategy": CHUNK_STRATEGY}

# ==================== 解析引擎 ====================
# 可选 MinerU 解析（复杂版面/表格/公式），需单独安装 mineru 并置 True
PARSER_ENABLE_MINERU = False

# ==================== 上传限制 ====================
# 单文件上传上限（MB）。超大文件解析耗时、内存占用高，且受 Streamlit 默认 200MB 限制。
# 超限文件在上传界面直接拦截并给出明确提示（不进入解析/构建流程）。
MAX_UPLOAD_MB = 200

# ==================== 检索与重排序参数 ====================
RETRIEVER_K = 20           # 混合检索初始召回数量
# 重排序后最终保留文档数。评测显示：本知识库原始 Top-3 已足够，
# 重排 Top-3 偏激进，故放宽到 5，兼顾重排价值与上下文覆盖。
RERANK_TOP_K = 5
RERANK_MIN_DOCS = 3        # 触发重排序的最低文档数（文档数大于此值才执行重排）

# ==================== 混合检索融合与缓存 ====================
RRF_K = 60                 # RRF 融合的位次平滑常数（k 越大，位次差异影响越小）
RETRIEVER_CACHE_SIZE = 128 # 检索结果 LRU 缓存条数（重建知识库后自动清空）

# ==================== 历史对话截断 ====================
MAX_HISTORY_TOKENS = 2000

# ==================== 其他默认值 ====================
DEFAULT_TEMPERATURE = 0.3   # 仅在未从 config.json 读取时使用

# ==================== Web 联网搜索（可选能力，默认关闭） ====================
# 关闭时 Agent 仅使用本地知识库工具；开启后额外暴露 web_search 工具（必应，免 Key）。
# 联网搜索会把查询词发送给搜索引擎，请注意隐私权衡（本地仍做脱敏）。
WEB_SEARCH_ENABLED = False
WEB_SEARCH_MAX_RESULTS = 5   # 默认返回结果条数（1-10）

# ==================== LLM 高级采样参数 ====================
# None = 未配置，使用服务商默认最优值（与 RAGFlow 的"留空用默认"语义一致）
LLM_OPTIONAL_DEFAULTS = {
    "max_tokens": None,          # 最大输出 Token（None 时由调用方传 MAX_COMPLETION_TOKENS）
    "top_p": None,               # 核采样（0-1）
    "top_k": None,               # 候选采样（部分服务商支持，如 DeepSeek/Qwen）
    "frequency_penalty": None,   # 频率惩罚（-2 ~ 2）
    "presence_penalty": None,    # 存在惩罚（-2 ~ 2）
    "max_context_length": None,  # 最大上下文长度（用于历史截断，None = 服务商默认窗口）
}


def get_llm_optional_config(cfg: dict) -> dict:
    """从配置 dict 读取 LLM 可选参数，缺失时用 None（= 服务商默认）。"""
    out = dict(LLM_OPTIONAL_DEFAULTS)
    for key in out:
        val = cfg.get(key)
        if val not in (None, "", 0):  # 0/空 视为未配置，走服务商默认
            out[key] = val
    return out

# ==================== BM25 缓存 ====================
BM25_CACHE_FILENAME = "bm25_cache.pkl"   # 缓存文件名，存放在 CHROMA_DB_DIR 下

# ==================== LLM 调用 Token 限制 ====================
# 主对话（RAG / CHAT）最大输出 Token 数
# 设置为 2048 可覆盖绝大多数长回答（约 1500-2000 汉字），兼顾成本与长度
MAX_COMPLETION_TOKENS = 2048

# 意图识别（轻量分类）最大输出 Token 数
# 只需输出 A/B/C 单字母，2-5 个 token 足够，保持 5 以兼容多余空格
INTENT_MAX_TOKENS = 5

# ==================== MCP 工具执行后端 ====================
# direct: 进程内直调检索器（默认，轻量稳定，复用已加载模型）
# mcp:    通过 MCP 协议（stdio）调用 mcp_server.py（演示真实 MCP 链路）
MCP_TOOL_BACKEND = "direct"