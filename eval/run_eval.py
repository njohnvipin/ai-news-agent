"""
run_eval.py — Measure the RAG pipeline on a fixed, synthetic test set.

Run from the project folder:
    python -m eval.run_eval              retrieval metrics only (no LLM calls, free)
    python -m eval.run_eval --with-llm   also checks answers, citations and refusals (uses Groq)

Uses its own ChromaDB collection ("eval_fixture"), deleted afterwards, so your
real knowledge base is never touched. Live news changes every day, so a fixed
test set is what makes results comparable between versions.
"""
import argparse
import json
import sys
from pathlib import Path

from services.rag_service import (
    NO_ANSWER,
    ask_news_rag,
    group_by_article,
    ingest_articles,
    retrieve_relevant_news,
)
from utils.chroma_utils import delete_collection

EVAL_COLLECTION = "eval_fixture"
FIXTURES = Path(__file__).with_name("fixtures.json")


def _link(article_id: str) -> str:
    return f"https://synthetic.example/{article_id}"


def load_fixtures() -> dict:
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    for article in data["articles"]:
        article["link"] = _link(article["id"])
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--with-llm", action="store_true", help="also evaluate generated answers (calls Groq)")
    args = parser.parse_args()

    data = load_fixtures()
    delete_collection(EVAL_COLLECTION)
    try:
        ingest_articles(data["articles"], collection_name=EVAL_COLLECTION, download_full_text=False)

        # ── Retrieval: is the right article among the sources, and how high? ──
        print("\nRETRIEVAL")
        hits, reciprocal_ranks = 0, []
        for case in data["questions"]:
            groups = group_by_article(retrieve_relevant_news(case["question"], collection_name=EVAL_COLLECTION))
            links = [g["metadata"]["link"] for g in groups]
            expected = _link(case["expected"])
            rank = links.index(expected) + 1 if expected in links else None
            hits += rank is not None
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            print(f"  {'PASS' if rank else 'FAIL'}  rank={rank or '-':<2} {case['question']}")

        refused = 0
        for q in data["off_topic"]:
            nothing_found = not retrieve_relevant_news(q, collection_name=EVAL_COLLECTION)
            refused += nothing_found
            print(f"  {'PASS' if nothing_found else 'FAIL'}  off-topic  {q}")

        n = len(data["questions"])
        print(f"\n  Hit rate:            {hits}/{n} ({hits / n:.0%})  right article among the sources")
        print(f"  Mean reciprocal rank: {sum(reciprocal_ranks) / n:.2f}  1.00 = always ranked first")
        print(f"  Off-topic filtered:  {refused}/{len(data['off_topic'])}  nothing passed the relevance threshold")

        # ── Generation: valid citations, honest refusals ─────────────────────
        if args.with_llm:
            print("\nANSWERS (Groq)")
            clean = 0
            for case in data["questions"]:
                result = ask_news_rag(case["question"], collection_name=EVAL_COLLECTION)
                ok = result["ok"] and not result["warnings"] and NO_ANSWER not in result["answer"]
                clean += ok
                print(f"  {'PASS' if ok else 'FAIL'}  {case['question']}")
                for w in result.get("warnings", []):
                    print(f"        ! {w}")
            honest = 0
            for q in data["off_topic"]:
                result = ask_news_rag(q, collection_name=EVAL_COLLECTION)
                said_no = NO_ANSWER in result["answer"]
                honest += said_no
                print(f"  {'PASS' if said_no else 'FAIL'}  off-topic  {q}")
            print(f"\n  Answered with valid citations: {clean}/{n}")
            print(f"  Said \"I don't know\" off-topic:  {honest}/{len(data['off_topic'])}")
    finally:
        delete_collection(EVAL_COLLECTION, recreate=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
