# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# core/orchestrator.py
# 对话编排层：快速通道闲聊 → Agent 模式（LLM 通过 function calling 自主决策调用工具）→ 生成
# 本模块禁止依赖 Streamlit，可被 Web、API、CLI 复用。
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from config import (
    RERANK_TOP_K,
    RERANK_MIN_DOCS,
    DEFAULT_TEMPERATURE,
    MAX_HISTORY_TOKENS,
    MAX_COMPLETION_TOKENS,
    get_llm_optional_config,
)
from core.history import truncate_history
from core.intents import detect_fastlane_intent
from core.mcp.executor import ToolExecutor
from core.mcp.registry import get_tool_schemas
from core.prompts import (
    get_dynamic_prompt,
    build_chat_prompt,
    build_chat_web_prompt,
    get_agentic_system_prompt,
    GLOBAL_RESPONSE,
)
from llm.client import build_llm
from security.desensitize import desensitize_text
from rag.reranker import rerank_documents
from utils.logger import get_logger

logger = get_logger("orchestrator")


@dataclass
class OrchestrationResult:
    """一次对话编排的输出。"""

    intent: str                                   # CHAT / AGENTIC / RAG / GLOBAL
    response: str                                 # 完整回复（含来源头，用于持久化）
    answer: str                                   # 纯回答正文（不含来源头，用于逐字展示）
    source_header: str                            # 来源头 Markdown（无来源时为空串）
    token_usage: Optional[Dict[str, int]]
    sources: List[Dict[str, Any]]                 # [{source, pages}]
    source_chunks: List[Dict[str, Any]] = field(default_factory=list)  # [{source, page, content}] 供可展开卡片
    retrieved_count: int = 0                      # RAG 分支的初步召回数 / 工具调用数


class Orchestrator:
    """RAG 对话编排器，不感知任何前端框架。"""

    def __init__(self, runtime_config: dict):
        self.config = runtime_config

    # ================= 模型配置 =================
    def _llm_config(self) -> dict:
        c = self.config
        is_custom = c.get("provider") == "自定义"
        cfg = {
            # 优先级：运行时配置 > 环境变量（DEEPSEEK_API_KEY / BASE_URL）
            "api_key": c.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
            "base_url": (c.get("custom_base") if is_custom else c.get("base_url"))
                        or os.getenv("BASE_URL", ""),
            "model_name": c.get("custom_model") if is_custom else c.get("model_name"),
            "temperature": c.get("temperature", DEFAULT_TEMPERATURE),
        }
        # 高级采样参数（top_p/top_k/惩罚/上下文长度）：未配置时为 None → 服务商默认
        cfg.update(get_llm_optional_config(c))
        return cfg

    def _build_llm(self, cfg: dict, max_tokens: int = MAX_COMPLETION_TOKENS,
                   timeout: Optional[float] = None):
        """按运行时配置构建 LLM；可选参数未配置时透传 None（= 服务商默认最优）。
        max_tokens 优先使用用户配置（cfg['max_tokens']，设置页或 config.json），未配置回退默认值。"""
        user_max = cfg.get("max_tokens")
        if user_max:
            max_tokens = int(user_max)
        return build_llm(
            cfg["api_key"], cfg["base_url"], cfg["model_name"], cfg["temperature"],
            max_tokens=max_tokens, timeout=timeout,
            top_p=cfg.get("top_p"), top_k=cfg.get("top_k"),
            frequency_penalty=cfg.get("frequency_penalty"),
            presence_penalty=cfg.get("presence_penalty"),
        )

    def _history_limit(self, cfg: dict) -> int:
        """历史截断上限：用户配置了最大上下文长度则用之，否则用默认 2000 token。"""
        ctx = cfg.get("max_context_length")
        return int(ctx) if ctx else MAX_HISTORY_TOKENS

    # ================= 对外接口 =================
    def detect(self, question: str) -> str:
        """意图初判：本地快速通道返回 CHAT，其余进入 Agent 模式（AGENTIC）。"""
        return detect_fastlane_intent(question)

    def run(self, question: str, history: List[dict], role_def: str,
            retriever, reranker, history_token_counts: Optional[List[int]] = None,
            on_stage: Optional[Callable] = None) -> OrchestrationResult:
        """完整流程：先初判意图，再按意图回答。"""
        intent = self.detect(question)
        return self.answer(question, intent, history, role_def, retriever, reranker,
                           history_token_counts, on_stage)

    def answer(self, question: str, intent: str, history: List[dict], role_def: str,
               retriever, reranker, history_token_counts: Optional[List[int]] = None,
               on_stage: Optional[Callable] = None) -> OrchestrationResult:
        """按意图执行回答流程。AGENTIC 异常时自动降级为经典 RAG 路径。"""
        if intent == "CHAT":
            return self._answer_chat(question, role_def, on_stage)
        if intent == "AGENTIC":
            try:
                return self._answer_agentic(question, history, role_def, retriever, reranker,
                                            history_token_counts, on_stage)
            except Exception as e:
                logger.warning("Agent 编排异常，降级为经典 RAG 路径: %s", e)
                return self._answer_rag(question, history, role_def, retriever, reranker,
                                        history_token_counts, on_stage)
        if intent == "GLOBAL":
            return OrchestrationResult(
                intent="GLOBAL", response=GLOBAL_RESPONSE, answer=GLOBAL_RESPONSE,
                source_header="",
                token_usage={"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
                sources=[], retrieved_count=0,
            )
        # 兼容旧意图值 RAG
        return self._answer_rag(question, history, role_def, retriever, reranker,
                                history_token_counts, on_stage)

    # ================= 闲聊分支 =================
    def _answer_chat(self, question: str, role_def: str, on_stage=None) -> OrchestrationResult:
        def _stage(event: str, **kw):
            if on_stage:
                on_stage(event, **kw)

        cfg = self._llm_config()
        _stage("chat")
        llm = self._build_llm(cfg)
        # 闲聊问题同样需要脱敏后再发往云端
        question_clean = desensitize_text(question)

        final_response = None
        if self.config.get("web_search_enabled", False):
            # 闲聊分支联网支持：绑定 web_search 工具，需要最新外部资讯时按需搜索后再回答；
            # 纯闲聊（问候/寒暄等）模型会直接回答，不调用工具。
            from core.mcp.registry import TOOLS
            web_schema = [t for t in TOOLS if t["function"]["name"] == "web_search"]
            executor = ToolExecutor()  # 仅用 web_search，不加载知识库检索模型
            llm_with_tools = llm.bind_tools(web_schema)
            sys_tpl, hum_tpl = build_chat_web_prompt(role_def)
            messages = [
                SystemMessage(content=sys_tpl),
                HumanMessage(content=hum_tpl.format(question=question_clean)),
            ]
            for _ in range(2):  # 最多两轮工具调用
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    final_response = response
                    break
                for tc in tool_calls:
                    _stage("tool_call", name=tc.get("name", ""), args=tc.get("args", {}) or {})
                    try:
                        result_text = executor.call(tc["name"], tc.get("args", {}) or {})
                    except Exception as e:
                        result_text = f"工具执行失败：{e}"
                    messages.append(ToolMessage(content=result_text, tool_call_id=tc.get("id", "")))
                    _stage("tool_result", name=tc.get("name", ""))
            if final_response is None:
                final_response = llm.invoke(messages)
        else:
            sys_tpl, hum_tpl = build_chat_prompt(role_def)
            prompt = [
                SystemMessage(content=sys_tpl),
                HumanMessage(content=hum_tpl.format(question=question_clean)),
            ]
            final_response = llm.invoke(prompt)

        content = final_response.content or ""
        return OrchestrationResult(
            intent="CHAT", response=content, answer=content,
            source_header="",
            token_usage=self._extract_usage(final_response), sources=[], retrieved_count=0,
        )

    # ================= Agent 模式（function calling + MCP 工具契约） =================
    def _answer_agentic(self, question: str, history: List[dict], role_def: str,
                        retriever, reranker, history_token_counts: Optional[List[int]],
                        on_stage: Optional[Callable]) -> OrchestrationResult:
        def _stage(event: str, **kw):
            if on_stage:
                on_stage(event, **kw)

        cfg = self._llm_config()
        _stage("thinking")

        # 工具执行器：优先复用调用方传入的检索器，避免重复加载模型
        executor = ToolExecutor(retriever=retriever, reranker=reranker,
                                latest_only=self.config.get("latest_only", False),
                                top_k=int(self.config.get("rerank_top_k", RERANK_TOP_K)),
                                min_score=float(self.config.get("min_score", 0.0) or 0.0))

        truncated_hist, _ = truncate_history(history, self._history_limit(cfg), history_token_counts)
        history_text = "\n".join(
            [f"{'用户' if m['role'] == 'user' else '助手'}：{desensitize_text(m['content'])}"
             for m in truncated_hist]
        )

        messages = [
            SystemMessage(content=get_agentic_system_prompt(
                role_def, enable_web_search=self.config.get("web_search_enabled", False))),
            HumanMessage(content=f"{history_text}\n\n用户最新问题：{desensitize_text(question)}"),
        ]

        llm = self._build_llm(cfg)
        # 工具列表按配置开关动态组装：默认仅本地知识库工具；开启 Web 联网搜索后才暴露 web_search
        llm_with_tools = llm.bind_tools(
            get_tool_schemas(enable_web_search=self.config.get("web_search_enabled", False)))

        # 多轮工具调用循环：模型可能需要"先检索 → 再补充检索/联网 → 最后作答"多轮决策。
        # 仅在模型不再要求调用工具（返回纯文本）时视为最终回答；否则把工具结果回填后继续。
        # 达到轮数上限仍无最终回答时，去掉工具绑定强制文本作答，避免把 <tool_calls> 原文当答案输出。
        tool_result_texts = []
        max_tool_rounds = 4
        final_response = None
        for _ in range(max_tool_rounds):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                final_response = response
                break
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {}) or {}
                _stage("tool_call", name=tool_name, args=tool_args)
                try:
                    result_text = executor.call(tool_name, tool_args)
                except Exception as e:
                    result_text = f"工具执行失败：{e}"
                tool_result_texts.append(result_text)
                messages.append(ToolMessage(content=result_text, tool_call_id=tc.get("id", "")))
                _stage("tool_result", name=tool_name)
        if final_response is None:
            # 超出轮数仍执着于调用工具 → 撤销工具绑定，强制基于现有资料直接回答
            _stage("generating")
            final_response = llm.invoke(messages)

        _stage("generating")
        answer_content = final_response.content or "根据现有资料无法回答。"

        # 从工具结果还原溯源信息（来源 + 页码 + 切片内容）
        sources = self._parse_sources(tool_result_texts)
        source_chunks = self._parse_source_chunks(tool_result_texts)
        source_header = self._build_source_header(sources)

        # 溯源补全精修：大模型标注行为不稳定，回答有资料依据但引用标注明显不足时，
        # 做一次"补全引用"精修，保证每个基于资料的事实句末都有【来源：文件名】。
        if sources and answer_content.count("【来源：") < max(3, len(answer_content) // 60):
            refined = self._refine_citations(question, answer_content, tool_result_texts, cfg)
            if refined:
                answer_content = refined

        return OrchestrationResult(
            intent="AGENTIC", response=answer_content, answer=answer_content,
            source_header=source_header,
            token_usage=self._extract_usage(final_response), sources=sources,
            source_chunks=source_chunks,
            retrieved_count=len(tool_result_texts),
        )

    @staticmethod
    def _refine_citations(question: str, draft: str, tool_result_texts: List[str], cfg: dict):
        """基于检索资料对回答草稿做"补全引用"精修：仅补充/校正缺失的引用标注，不改动事实内容。
        资料片段中的'来源:xxx'即为文件名；精修失败返回 None（保留原回答）。"""
        try:
            sys_msg = (
                "你是知识库问答助手。下面是针对用户问题已生成的回答草稿，以及检索到的资料片段。\n"
                "请重写回答，使【每一个基于资料的事实】句末都标注【来源：文件名】"
                "（资料片段中每条的'来源:xxx'即为文件名，可附页码）。\n"
                "规则：1) 只补充/校正引用标注，不得改动事实内容、语气与结构；"
                "2) 资料中没有的信息不得添加引用，也不得新增内容；"
                "3) 已正确标注的保持原样；4) 只输出修订后的回答正文，不要任何解释。")
            context = "\n\n".join(tool_result_texts)
            llm = build_llm(
                cfg["api_key"], cfg["base_url"], cfg["model_name"], cfg["temperature"],
                max_tokens=cfg.get("max_tokens") or 2048,
                top_p=cfg.get("top_p"), top_k=cfg.get("top_k"),
                frequency_penalty=cfg.get("frequency_penalty"),
                presence_penalty=cfg.get("presence_penalty"),
            )
            resp = llm.invoke([
                SystemMessage(content=sys_msg),
                HumanMessage(content=f"用户问题：{question}\n\n回答草稿：\n{draft}\n\n检索资料片段：\n{context}"),
            ])
            return resp.content
        except Exception as e:
            logger.warning("引用精修失败，保留原回答: %s", e)
            return None

    # ================= 经典 RAG 分支（降级兜底） =================
    def _answer_rag(self, question: str, history: List[dict], role_def: str,
                    retriever, reranker, history_token_counts: Optional[List[int]],
                    on_stage: Optional[Callable]) -> OrchestrationResult:
        def _stage(event: str, **kw):
            if on_stage:
                on_stage(event, **kw)

        cfg = self._llm_config()
        if retriever is None:
            raise RuntimeError("知识库组件未加载，无法进行检索。")

        _stage("loading")
        docs = retriever.invoke(question, latest_only=self.config.get("latest_only", False))
        initial_count = len(docs)
        _stage("retrieved", count=initial_count)

        docs = rerank_documents(question, docs, reranker,
                                int(self.config.get("rerank_top_k", RERANK_TOP_K)),
                                RERANK_MIN_DOCS,
                                float(self.config.get("min_score", 0.0) or 0.0))
        _stage("reranked", count=len(docs))

        sources = self._build_sources(docs)
        source_header = self._build_source_header(sources)
        source_chunks = [
            {"source": d.metadata.get("source", "未知"),
             "page": str(d.metadata.get("page", "?")),
             "content": d.page_content}
            for d in docs
        ]

        # 端侧脱敏：仅作用于发往云端 LLM 的数据
        context_text = desensitize_text("\n".join(d.page_content for d in docs))
        question_clean = desensitize_text(question)

        truncated_hist, _ = truncate_history(history, self._history_limit(cfg), history_token_counts)
        history_text = "\n".join(
            [f"{'用户' if m['role'] == 'user' else '助手'}：{desensitize_text(m['content'])}"
             for m in truncated_hist]
        )

        sys_tpl, hum_tpl = get_dynamic_prompt(role_def)
        prompt = [
            SystemMessage(content=sys_tpl.format(history=history_text, context=context_text)),
            HumanMessage(content=hum_tpl.format(question=question_clean)),
        ]
        _stage("generating")
        llm = self._build_llm(cfg)
        response = llm.invoke(prompt)
        answer_content = response.content

        # 溯源补全精修（与 Agentic 分支一致）：回答有资料依据但引用标注明显不足时补全【来源：文件名】。
        if sources and answer_content.count("【来源：") < max(3, len(answer_content) // 60):
            # RAG 分支的上下文不带"来源:xxx"前缀，需重建带来源的片段供精修对照
            context_items = "\n\n".join(
                f"[{i + 1}] 来源:{d.metadata.get('source', '未知')} "
                f"页码:P{d.metadata.get('page', '?')}\n{d.page_content}"
                for i, d in enumerate(docs))
            refined = self._refine_citations(question, answer_content, [context_items], cfg)
            if refined:
                answer_content = refined

        return OrchestrationResult(
            intent="RAG", response=answer_content, answer=answer_content,
            source_header=source_header,
            token_usage=self._extract_usage(response), sources=sources,
            source_chunks=source_chunks,
            retrieved_count=initial_count,
        )

    # ================= 内部工具 =================
    @staticmethod
    def _extract_usage(llm_response) -> Optional[Dict[str, int]]:
        usage = getattr(llm_response, "usage_metadata", None)
        if not usage:
            return None
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens) or 0
        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    @staticmethod
    def _build_sources(docs) -> List[Dict[str, Any]]:
        source_map: Dict[str, set] = {}
        for doc in docs:
            source_map.setdefault(doc.metadata.get("source", "未知"), set()).add(
                doc.metadata.get("page", "?")
            )
        return [{"source": f, "pages": sorted(str(p) for p in ps)} for f, ps in source_map.items()]

    @staticmethod
    def _parse_sources(tool_result_texts: List[str]) -> List[Dict[str, Any]]:
        """从工具返回文本中解析 来源:文件名 页码:Pn，用于还原溯源信息。"""
        source_map: Dict[str, set] = {}
        # 文件名可能包含空格，因此用非贪婪 .+? 匹配，并以"页码:P"作为截止锚点
        pattern = re.compile(r"来源:(.+?)\s+页码:P([\d?]+)")
        for text in tool_result_texts:
            for m in pattern.finditer(text or ""):
                source_map.setdefault(m.group(1).strip(), set()).add(m.group(2))
        return [{"source": f, "pages": sorted(str(p) for p in ps)} for f, ps in source_map.items()]

    @staticmethod
    def _parse_source_chunks(tool_result_texts: List[str]) -> List[Dict[str, Any]]:
        """从工具返回文本解析 [序号] 来源:xxx 页码:Pn\\ncontent 形式的切片内容。"""
        chunks: List[Dict[str, Any]] = []
        for text in tool_result_texts:
            blocks = re.split(r"\n\n(?=\[\d+\] 来源:)", text or "")
            for block in blocks:
                m = re.match(r"\[\d+\] 来源:(.+?)\s+页码:P([\d?]+)\s*\n(.*)", block, re.DOTALL)
                if m:
                    chunks.append({
                        "source": m.group(1).strip(),
                        "page": m.group(2),
                        "content": m.group(3).strip(),
                    })
        return chunks

    @staticmethod
    def _build_source_header(sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return ""
        lines = [f"- `{s['source']}` (P.{', '.join(s['pages'])})" for s in sources]
        return "📚 **参考资料**：\n" + "\n".join(lines) + "\n\n---\n\n"
