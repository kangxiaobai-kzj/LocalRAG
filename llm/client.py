# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# llm/client.py
# LLM 连接工厂与连通性测试（兼容 OpenAI 协议的服务商）
# 注意：langchain_openai 导入耗时约 13s（连带 huggingface_hub 等），
# 因此延迟到 build_llm 内导入，避免拖慢应用冷启动首屏。


def build_llm(api_key, base_url, model_name, temperature,
              max_tokens=None, streaming=False, timeout=None,
              top_p=None, top_k=None, frequency_penalty=None, presence_penalty=None):
    """构建 ChatOpenAI 实例。可选高级参数（top_p/top_k/frequency/presence）为 None 时使用服务商默认。"""
    from langchain_openai import ChatOpenAI  # 延迟导入：加速首屏加载
    kwargs = {"streaming": streaming}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    # 高级采样参数：仅显式配置时透传，未配置则由服务商采用默认最优值
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
        **kwargs,
    )


def test_llm_connection(api_key, base_url, model, temperature):
    """测试 API 连接。"""
    try:
        llm = build_llm(api_key, base_url, model, temperature, timeout=10)
        response = llm.invoke("请回复：连接成功")
        return {"success": True, "message": response.content[:50]}
    except Exception as e:
        return {"success": False, "error": str(e)}
