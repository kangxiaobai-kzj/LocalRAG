# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# eval/evaluate.py
# 黄金 QA 检索基线评测：验证 Top-k 检索结果是否命中问题的关键词
# 输出两类指标：
#   1) 精排 Top-k 全命中率（当前生产口径）
#   2) recall@1/3/5/10（在混合检索原始排序上统计，用于诊断检索深度）
# 用法（在项目根目录、虚拟环境内）：
#   python eval/evaluate.py                        # 默认：含重排 Top-3 + recall@k
#   python eval/evaluate.py --no-rerank --topk 10  # 关闭重排 / 调整 Top-k
import argparse
import json
import os
import sys

# 允许以脚本方式从项目根目录导入包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_DB_DIR, RERANK_TOP_K, RERANK_MIN_DOCS
from rag.retriever import load_retriever_and_reranker
from rag.reranker import rerank_documents

# recall@k 诊断深度
RECALL_KS = [1, 3, 5, 10]


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def matched_keywords(docs, keywords):
    """返回 docs 中命中的关键词列表。"""
    return [kw for kw in keywords if any(kw in d.page_content for d in docs)]


def main():
    parser = argparse.ArgumentParser(description="黄金 QA 检索基线评测")
    parser.add_argument("--questions", default="eval/questions.json", help="问题集 JSON 路径")
    parser.add_argument("--topk", type=int, default=RERANK_TOP_K, help="精排后保留的文档数")
    parser.add_argument("--no-rerank", action="store_true", help="跳过重排，仅用混合检索")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("❌ 问题集为空，请检查 questions.json。")
        return

    retriever, reranker = load_retriever_and_reranker(CHROMA_DB_DIR)
    if retriever is None:
        print("❌ 知识库未构建，无法评测。请先在 Web 端上传 PDF 并重建知识库。")
        return

    mode = "含重排" if not args.no_rerank else "仅混合检索"
    print(f"评测模式：{mode}，精排 Top-{args.topk}；recall 深度 {RECALL_KS}\n")

    headline_hits = 0
    recall_hits = {k: 0 for k in RECALL_KS}
    rows = []

    for q in questions:
        ranked = retriever.invoke(q["question"])  # 混合检索原始排序（Top-RETRIEVER_K）

        # 精排 Top-k（生产口径）
        if not args.no_rerank:
            top = rerank_documents(q["question"], ranked, reranker, args.topk, RERANK_MIN_DOCS)
        else:
            top = ranked[: args.topk]
        matched = matched_keywords(top, q["keywords"])
        headline_hit = len(matched) == len(q["keywords"])
        headline_hits += 1 if headline_hit else 0

        # recall@k（在原始召回序上，不依赖重排）
        per_k = {}
        for k in RECALL_KS:
            per_k[k] = len(matched_keywords(ranked[:k], q["keywords"])) == len(q["keywords"])
            recall_hits[k] += 1 if per_k[k] else 0

        # Top-10 仍未全命中时，记录缺失关键词（用于区分排序问题 vs 覆盖问题）
        miss_at10 = [kw for kw in q["keywords"]
                     if kw not in matched_keywords(ranked[:RECALL_KS[-1]], q["keywords"])]
        rows.append((q, headline_hit, per_k, miss_at10))

    # 明细表
    header = f"{'ID':<6}{'精排Top' + str(args.topk):<10}" + "".join(f"{'hit@' + str(k):<8}" for k in RECALL_KS)
    print(header)
    print("-" * len(header))
    for q, headline_hit, per_k, miss_at10 in rows:
        cells = "".join("✅     " if per_k[k] else "❌     " for k in RECALL_KS)
        print(f"{q['id']:<6}{'✅' if headline_hit else '❌':<10}{cells}")

    # 汇总
    total = len(rows)
    print("-" * len(header))
    print(f"🎯 精排 Top-{args.topk} 全命中：{headline_hits}/{total} = {headline_hits / total:.0%}")
    for k in RECALL_KS:
        print(f"   recall@{k:<2}：{recall_hits[k]}/{total} = {recall_hits[k] / total:.0%}")

    # 诊断结论
    print("\n📋 诊断：")
    for q, headline_hit, per_k, miss_at10 in rows:
        if not per_k[RECALL_KS[-1]]:
            print(f"   ⚠️ {q['id']}：Top-{RECALL_KS[-1]} 仍未全命中，缺失关键词 {miss_at10}"
                  f" → 若文档确有该内容，属覆盖/切片问题（P4 解析层）；若 Top-3 缺而更深层有，属排序问题（P2 重排）")
        elif not headline_hit:
            print(f"   ℹ️ {q['id']}：更深层可命中但未进精排 Top-{args.topk} → 排序问题（重排可优化）")
    print("\n💡 提示：关键词若与文档用词不符，请核对 PDF 正文后调整 eval/questions.json。")


if __name__ == "__main__":
    main()
