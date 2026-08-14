# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# scripts/download_models.py
# 预下载本地模型权重（Embedding + 重排），下载后缓存本地，离线可用。
# 缓存位置由环境变量控制（未设置时用默认路径）：
#   FASTEMBED_CACHE_PATH → Embedding（fastembed）
#   HF_HOME              → 重排（HuggingFace）
# 用法：python scripts/download_models.py [--embed BAAI/bge-base-zh-v1.5]
import argparse
import os
import sys

# 允许以脚本方式从项目根目录导入（config.py 会设置 HF 镜像）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_embed(model_name: str) -> None:
    print(f"== 下载 Embedding 模型 {model_name}（约 90MB，走 hf-mirror 镜像）==")
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    emb = FastEmbedEmbeddings(model_name=model_name)
    emb.embed_documents(["LocalRAG 模型下载测试"])  # 触发下载与 ONNX 构建
    print("✅ Embedding 模型下载完成\n")


def download_rerank(model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
    print(f"== 下载重排模型 {model_name}（约 2.2GB，耗时较长）==")
    from sentence_transformers import SentenceTransformer
    _ = SentenceTransformer(model_name)
    print("✅ 重排模型下载完成\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="预下载 LocalRAG 本地模型")
    parser.add_argument("--embed", default="BAAI/bge-small-zh-v1.5",
                        help="Embedding 模型名（默认 bge-small-zh-v1.5）")
    parser.add_argument("--skip-rerank", action="store_true", help="跳过重排模型（体积大）")
    args = parser.parse_args()

    try:
        download_embed(args.embed)
        if not args.skip_rerank:
            download_rerank()
    except Exception as e:
        print(f"❌ 下载失败：{e}\n请检查网络后重试。")
        return 1

    # 校验
    import subprocess
    return subprocess.call([sys.executable,
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_models.py")])


if __name__ == "__main__":
    sys.exit(main())
