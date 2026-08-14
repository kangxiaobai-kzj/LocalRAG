# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# utils/__init__.py
# 集中管理运行时配置的读写与校验（脱敏已移至 security/desensitize.py）
# 说明：utils 为包，含 logger 子模块；本文件承担原 utils.py 的配置与加解密能力。
import base64
import json
import os

# API Key 加密存储：Windows 用 DPAPI（绑定当前 Windows 用户），其余平台退化为 base64 混淆
_ENC_DPAPI = "enc:dpapi:"
_ENC_B64 = "enc:b64:"


def encrypt_secret(text: str) -> str:
    """加密敏感串（API Key）。Windows 优先 DPAPI（仅当前用户可解密），失败退化为 base64 混淆。"""
    if not text:
        return ""
    try:
        import win32crypt
        blob = win32crypt.CryptProtectData(text.encode("utf-8"), "AI_Agent_Project",
                                           None, None, None, 0)
        return _ENC_DPAPI + base64.b64encode(blob).decode("ascii")
    except Exception:
        return _ENC_B64 + base64.b64encode(text.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> str:
    """解密 encrypt_secret 的产物；格式异常或（DPAPI 跨用户/跨机器）无法解密时返回空串。"""
    if not token:
        return ""
    try:
        if token.startswith(_ENC_DPAPI):
            import win32crypt
            blob = base64.b64decode(token[len(_ENC_DPAPI):])
            _, data = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
            return data.decode("utf-8")
        if token.startswith(_ENC_B64):
            return base64.b64decode(token[len(_ENC_B64):]).decode("utf-8")
    except Exception:
        pass
    return ""


def load_config(config_file):
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # 新版加密存于 api_key_enc，加载时自动解密；旧版明文 api_key 直接兼容
            enc = cfg.get("api_key_enc")
            if enc:
                cfg["api_key"] = decrypt_secret(enc)
            return cfg
        except Exception:
            return {}
    return {}


def save_config(config_file, config):
    """保存配置：api_key 不再明文落盘，改为加密后写入 api_key_enc。"""
    data = dict(config)
    api_key = data.get("api_key", "")
    if api_key:
        data["api_key_enc"] = encrypt_secret(api_key)
    else:
        data.pop("api_key_enc", None)
    data["api_key"] = ""  # 仅保留内存中的明文，磁盘一律加密
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_api_config(provider, api_key, base_url, model_name):
    """
    校验 API 配置的基本格式，避免因明显错误导致运行时异常。
    放宽了校验规则，仅做必要检查。
    Ollama 本地服务通常无需鉴权，API Key 可留空或填写任意占位符。
    返回 (is_valid, error_message)
    """
    is_local = provider == "Ollama (本地)"
    if not is_local:
        if not api_key or not api_key.strip():
            return False, "API Key 不能为空"
        # 仅做长度下限检查，不再限制字符类型（允许各种合法 API Key 格式）
        if len(api_key.strip()) < 10:
            return False, "API Key 长度过短（至少 10 个字符）"

    if not base_url or not base_url.strip():
        return False, "Base URL 不能为空"
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        return False, "Base URL 必须以 http:// 或 https:// 开头"

    if not model_name or not model_name.strip():
        return False, "Model Name 不能为空"
    if ' ' in model_name.strip():
        return False, "Model Name 不能包含空格"

    return True, ""
