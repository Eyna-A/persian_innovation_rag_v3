"""Evaluate FAQ retrieval with Precision@K, Recall@K, and MRR."""
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from rag import load_faq, load_embedder, connect_weaviate, retrieve, COLLECTION_NAME, TOP_K

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
EVAL_PATH = ROOT / "evaluation" / "evaluation_dataset.csv"


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"فایل ارزیابی پیدا نشد: {EVAL_PATH}")
    tests = pd.read_csv(EVAL_PATH, encoding="utf-8-sig").fillna("")
    required = {"query", "expected_source_question"}
    if not required.issubset(tests.columns):
        raise ValueError(f"ستون‌های لازم: {sorted(required)}")
    tests["query"] = tests["query"].astype(str).str.strip()
    tests["expected_source_question"] = tests["expected_source_question"].astype(str).str.strip()
    tests = tests[(tests.query != "") & (tests.expected_source_question != "")]
    if tests.empty:
        print("هیچ نمونه‌ای برای ارزیابی برچسب‌گذاری نشده. ستون expected_source_question را با سؤال دقیق از دیتاست پر کن.")
        return
    client = connect_weaviate()
    try:
        if not client.collections.exists(COLLECTION_NAME):
            raise RuntimeError("Collection پیدا نشد؛ ابتدا python rag.py ingest را اجرا کن.")
        collection = client.collections.get(COLLECTION_NAME)
        embedder = load_embedder()
        k_values = [int(x.strip()) for x in os.getenv("EVAL_K_VALUES", "1,3,5,8").split(",") if x.strip()]
        summaries = []
        for k in k_values:
            precisions, recalls, reciprocal_ranks = [], [], []
            for _, row in tests.iterrows():
                docs = retrieve(collection, embedder, row["query"], top_k=k)
                found_questions = [d["question"] for d in docs]
                expected = row["expected_source_question"]
                rank = found_questions.index(expected) + 1 if expected in found_questions else None
                relevant_count = int(rank is not None)
                # One expected relevant FAQ is labelled per evaluation query.
                precisions.append(relevant_count / k)
                recalls.append(float(relevant_count))
                reciprocal_ranks.append(1 / rank if rank else 0.0)
            summary = {"K": k, "Precision@K": sum(precisions) / len(precisions),
                       "Recall@K": sum(recalls) / len(recalls), "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks)}
            summaries.append(summary)
        print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nتعداد پرسش‌های ارزیابی‌شده: {len(tests)}")
        print("توجه: Recall در این پیاده‌سازی بر اساس یک منبع مرجع برای هر query است؛ این معیار کیفیت پاسخ نهایی LLM را نمی‌سنجد.")
    finally:
        client.close()

if __name__ == "__main__":
    main()
