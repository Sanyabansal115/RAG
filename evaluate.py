"""
RAG Evaluation Script for ReqMind
==================================
Runs a test set of queries against the RAG pipeline and computes:

  String-based  (no extra API calls, require reference_answer in testset):
    - ROUGE-L F1          — longest-common-subsequence overlap vs reference
    - Token F1            — bag-of-words precision/recall/F1 vs reference

  Retrieval-based  (require relevant_sources in testset):
    - Hit Rate @ K        — ≥1 relevant source in top-K retrieved chunks
    - Mean Reciprocal Rank (MRR) — rank of first relevant source

  LLM-as-Judge  (Mistral API, always computed):
    - Faithfulness        — answer is grounded in retrieved context (1-5)
    - Answer Relevance    — answer addresses the query (1-5)
    - Context Relevance   — retrieved context is useful for the query (1-5)

  System:
    - Latency (s)         — wall-clock time per query
    - Answer Found Rate   — fraction of queries with a non-fallback answer

Usage
-----
  # Full evaluation (LLM-judge enabled):
  python evaluate.py

  # Skip LLM-judge to save API quota:
  python evaluate.py --skip-llm-judge

  # Point to a custom testset or output path:
  python evaluate.py --testset my_tests.json --output my_results.json

  # First run: auto-populate reference answers from the current system,
  # then use them as the silver-standard baseline for future runs:
  python evaluate.py --build-testset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project imports (reuse existing modules)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from src.loader import load_pdfs_to_documents, scan_pdfs
from src.chunker import split_documents
from src.rag_chain import create_rag_chain
from src.vectorstore_build import DEFAULT_STORE_FILENAME, build_vectorstore, load_vectorstore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FALLBACK_PHRASES = [
    "could not find",
    "not find relevant",
    "no relevant information",
    "not in the context",
    "cannot find",
    "don't have information",
]
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


# ===========================================================================
# 1. String-based metrics  (pure Python, no extra deps)
# ===========================================================================

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute the length of the Longest Common Subsequence of two token lists."""
    m, n = len(x), len(y)
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-L precision, recall, and F1 via LCS."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """Bag-of-words token precision, recall, and F1."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum((pred_counts & ref_counts).values())
    if common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ===========================================================================
# 2. Retrieval metrics
# ===========================================================================

def _source_basename(path: str) -> str:
    return os.path.basename(path).strip()


def retrieval_metrics(source_documents, relevant_sources: List[str]) -> Dict[str, float]:
    """
    Compute Hit Rate @ K and Mean Reciprocal Rank (MRR).
    relevant_sources: list of expected source filenames (basenames).
    """
    if not relevant_sources:
        return {}

    relevant_set = {s.strip().lower() for s in relevant_sources}
    retrieved_basenames = [
        _source_basename(doc.metadata.get("source", "")).lower()
        for doc in source_documents
    ]

    # Hit Rate: 1 if any retrieved doc matches a relevant source
    hit = any(b in relevant_set for b in retrieved_basenames)

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for rank, b in enumerate(retrieved_basenames, start=1):
        if b in relevant_set:
            mrr = 1.0 / rank
            break

    return {"hit_rate": float(hit), "mrr": round(mrr, 4)}


# ===========================================================================
# 3. LLM-as-Judge metrics  (Mistral API)
# ===========================================================================

_LLM_JUDGE_SYSTEM = (
    "You are a strict evaluator of question-answering systems. "
    "Score the given aspect on a scale from 1 to 5 where:\n"
    "  1 = very poor  2 = poor  3 = acceptable  4 = good  5 = excellent\n"
    "Reply with ONLY a JSON object: {\"score\": <integer 1-5>, \"reason\": \"<one sentence>\"}. "
    "Do not include any other text."
)

_FAITHFULNESS_PROMPT = (
    "Context retrieved:\n{context}\n\n"
    "Question: {query}\n"
    "Answer: {answer}\n\n"
    "Score FAITHFULNESS: Is every claim in the answer directly supported by the retrieved context? "
    "Penalise heavily for any statement not found in the context."
)

_ANSWER_RELEVANCE_PROMPT = (
    "Question: {query}\n"
    "Answer: {answer}\n\n"
    "Score ANSWER RELEVANCE: Does the answer directly and completely address the question? "
    "Penalise for irrelevant content, excessive vagueness, or topic drift."
)

_CONTEXT_RELEVANCE_PROMPT = (
    "Question: {query}\n"
    "Retrieved context:\n{context}\n\n"
    "Score CONTEXT RELEVANCE: How well does the retrieved context contain the information needed "
    "to answer the question? Penalise for off-topic or unhelpful chunks."
)


def _call_judge(
    api_key: str,
    model: str,
    prompt: str,
    retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Call Mistral and parse the JSON judge response. Returns None on failure."""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                MISTRAL_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 120,
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Tolerate markdown fences around the JSON
            raw = raw.strip("` \n")
            if raw.startswith("json"):
                raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                print(f"    [WARN] LLM judge call failed: {exc}")
                return None
    return None


def llm_judge_metrics(
    api_key: str,
    model: str,
    query: str,
    answer: str,
    context: str,
) -> Dict[str, Any]:
    """Run all three LLM-judge evaluations and return scores + reasons."""
    results: Dict[str, Any] = {}

    for metric, prompt_template in [
        ("faithfulness", _FAITHFULNESS_PROMPT),
        ("answer_relevance", _ANSWER_RELEVANCE_PROMPT),
        ("context_relevance", _CONTEXT_RELEVANCE_PROMPT),
    ]:
        prompt = prompt_template.format(query=query, answer=answer, context=context)
        parsed = _call_judge(api_key, model, prompt)
        if parsed and "score" in parsed:
            results[f"{metric}_score"] = int(parsed["score"])
            results[f"{metric}_reason"] = parsed.get("reason", "")
        else:
            results[f"{metric}_score"] = None
            results[f"{metric}_reason"] = "evaluation failed"

    return results


# ===========================================================================
# 4. Helper: format context string from source docs
# ===========================================================================

def _docs_to_context(source_documents) -> str:
    parts = []
    for doc in source_documents:
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {os.path.basename(src)}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _is_fallback_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in FALLBACK_PHRASES)


# ===========================================================================
# 5. Core evaluation loop
# ===========================================================================

def evaluate(
    qa_chain,
    testset: List[Dict],
    api_key: str,
    model: str,
    skip_llm_judge: bool = False,
) -> List[Dict]:
    results = []
    n = len(testset)

    for i, case in enumerate(testset, start=1):
        qid = case.get("id", i)
        query = case["query"]
        reference = case.get("reference_answer")
        relevant_sources = case.get("relevant_sources", [])
        tags = case.get("tags", [])

        print(f"  [{i}/{n}] id={qid}  {query[:70]}{'...' if len(query) > 70 else ''}")

        # --- Retrieval (always run, independent of LLM) ---
        source_docs = qa_chain.vectordb.similarity_search(query, k=qa_chain.top_k)

        # --- LLM generation (timed) ---
        t0 = time.perf_counter()
        rag_error: Optional[str] = None
        answer: str = ""
        if source_docs:
            context_for_llm = _docs_to_context(source_docs)
            try:
                rag_output = qa_chain.invoke({"query": query})
                answer = rag_output.get("result", "")
                # sync back any docs the chain itself resolved
                source_docs = rag_output.get("source_documents", source_docs)
            except requests.exceptions.HTTPError as exc:
                rag_error = f"HTTP {exc.response.status_code}: {exc.response.reason}"
                print(f"    [WARN] LLM call failed — {rag_error}")
            except Exception as exc:  # noqa: BLE001
                rag_error = str(exc)
                print(f"    [WARN] LLM call failed — {rag_error}")
        else:
            answer = "I could not find relevant information in the provided documents."
        latency = round(time.perf_counter() - t0, 3)
        context_str = _docs_to_context(source_docs)
        record: Dict[str, Any] = {
            "id": qid,
            "query": query,
            "tags": tags,
            "answer": answer,
            "latency_s": latency,
            "error": rag_error,
            "answer_found": not _is_fallback_answer(answer) if not rag_error else None,
            "num_retrieved_chunks": len(source_docs),
        }

        # --- String-based metrics ---
        if reference is not None:
            record["reference_answer"] = reference
            rl = rouge_l(answer, reference)
            tf = token_f1(answer, reference)
            record["rouge_l_f1"] = rl["f1"]
            record["rouge_l_precision"] = rl["precision"]
            record["rouge_l_recall"] = rl["recall"]
            record["token_f1"] = tf["f1"]
            record["token_precision"] = tf["precision"]
            record["token_recall"] = tf["recall"]
        else:
            record["reference_answer"] = None
            record["rouge_l_f1"] = None
            record["token_f1"] = None

        # --- Retrieval metrics ---
        if relevant_sources:
            ret = retrieval_metrics(source_docs, relevant_sources)
            record["hit_rate"] = ret.get("hit_rate")
            record["mrr"] = ret.get("mrr")
        else:
            record["hit_rate"] = None
            record["mrr"] = None

        # --- LLM-judge metrics ---
        if not skip_llm_judge and not rag_error:
            judge = llm_judge_metrics(api_key, model, query, answer, context_str)
            record.update(judge)
        else:
            for k in ("faithfulness_score", "answer_relevance_score", "context_relevance_score"):
                record[k] = None

        results.append(record)

    return results


# ===========================================================================
# 6. Aggregate + pretty-print summary
# ===========================================================================

def _safe_mean(values: List) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def aggregate(results: List[Dict]) -> Dict[str, Any]:
    keys = [
        "latency_s",
        "rouge_l_f1",
        "token_f1",
        "hit_rate",
        "mrr",
        "faithfulness_score",
        "answer_relevance_score",
        "context_relevance_score",
    ]
    summary: Dict[str, Any] = {
        "total_queries": len(results),
        "answer_found_rate": round(
            sum(1 for r in results if r.get("answer_found")) / len(results), 4
        )
        if results
        else 0.0,
    }
    for k in keys:
        summary[f"mean_{k}"] = _safe_mean([r.get(k) for r in results])
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    labels = {
        "total_queries":                  "Total queries evaluated",
        "answer_found_rate":              "Answer Found Rate        (↑ higher better)",
        "mean_latency_s":                 "Mean Latency (s)         (↓ lower better)",
        "mean_rouge_l_f1":                "Mean ROUGE-L F1          (↑ higher better)",
        "mean_token_f1":                  "Mean Token F1            (↑ higher better)",
        "mean_hit_rate":                  "Mean Hit Rate @ K        (↑ higher better)",
        "mean_mrr":                       "Mean MRR                 (↑ higher better)",
        "mean_faithfulness_score":        "Mean Faithfulness /5     (↑ higher better)",
        "mean_answer_relevance_score":    "Mean Answer Relevance /5 (↑ higher better)",
        "mean_context_relevance_score":   "Mean Context Relevance /5(↑ higher better)",
    }
    for key, label in labels.items():
        value = summary.get(key)
        display = f"{value}" if value is not None else "N/A"
        print(f"  {label:<44} {display}")
    print("=" * 60)


# ===========================================================================
# 7. --build-testset mode: auto-generate silver reference answers
# ===========================================================================

def build_testset_with_references(
    qa_chain,
    testset: List[Dict],
    output_path: str,
) -> None:
    """
    Run each query through the current RAG system and store the answers as
    silver-standard reference_answers.  Only overwrites entries where
    reference_answer is currently null.
    """
    updated = deepcopy(testset)
    n = len(updated)
    print(f"\nGenerating reference answers for {n} queries …")
    for i, case in enumerate(updated, start=1):
        if case.get("reference_answer") is not None:
            print(f"  [{i}/{n}] id={case['id']} — skipped (already has reference)")
            continue
        print(f"  [{i}/{n}] id={case['id']}  {case['query'][:60]}")
        output = qa_chain.invoke({"query": case["query"]})
        case["reference_answer"] = output.get("result", "").strip()

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(updated, fh, indent=2, ensure_ascii=False)
    print(f"\nSilver testset saved to: {output_path}")
    print("Review the reference answers, edit as needed, then run evaluate.py normally.")


# ===========================================================================
# 8. Entry point
# ===========================================================================

def _load_or_build_vectorstore(chunks, persist_directory: str):
    store_path = os.path.join(persist_directory, DEFAULT_STORE_FILENAME)
    if os.path.exists(store_path):
        try:
            vectordb = load_vectorstore(persist_directory=persist_directory)
            if vectordb.count() > 0:
                return vectordb
        except Exception:
            pass
    return build_vectorstore(chunks=chunks, persist_directory=persist_directory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ReqMind RAG pipeline.")
    parser.add_argument(
        "--testset",
        default="eval_testset.json",
        help="Path to the JSON test set (default: eval_testset.json)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Path for the detailed JSON output (default: eval_results.json)",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge metrics (saves API calls)",
    )
    parser.add_argument(
        "--build-testset",
        action="store_true",
        help=(
            "Auto-populate null reference_answers using the current RAG system "
            "and save the result alongside the original testset file "
            "(adds '_with_refs' suffix).  Does not run evaluation."
        ),
    )
    args = parser.parse_args()

    # --- Environment ---
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY", "")
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    if not api_key:
        sys.exit("[ERROR] MISTRAL_API_KEY not set. Add it to your .env file.")

    # --- Load test set ---
    testset_path = args.testset
    if not os.path.exists(testset_path):
        sys.exit(f"[ERROR] Test set file not found: {testset_path}")
    with open(testset_path, "r", encoding="utf-8") as fh:
        testset: List[Dict] = json.load(fh)
    print(f"Loaded {len(testset)} test cases from {testset_path}")

    # --- Build RAG pipeline ---
    data_dir = os.path.join(os.getcwd(), "data")
    persist_dir = os.path.join(os.getcwd(), "chroma_db")

    pdf_paths = scan_pdfs(data_dir)
    if not pdf_paths:
        sys.exit(f"[ERROR] No PDFs found in {data_dir}.")

    documents = load_pdfs_to_documents(pdf_paths)
    chunks = split_documents(documents)
    vectordb = _load_or_build_vectorstore(chunks, persist_dir)
    qa_chain = create_rag_chain(api_key, model, vectordb)

    # --- --build-testset mode ---
    if args.build_testset:
        base, ext = os.path.splitext(testset_path)
        out_path = f"{base}_with_refs{ext}"
        build_testset_with_references(qa_chain, testset, out_path)
        return

    # --- Evaluation ---
    print(f"\nRunning evaluation on {len(testset)} queries …")
    if args.skip_llm_judge:
        print("(LLM-judge metrics are disabled)\n")
    else:
        print("(LLM-judge metrics are enabled — this will use Mistral API quota)\n")

    results = evaluate(
        qa_chain=qa_chain,
        testset=testset,
        api_key=api_key,
        model=model,
        skip_llm_judge=args.skip_llm_judge,
    )

    # --- Aggregate + display ---
    summary = aggregate(results)
    print_summary(summary)

    # --- Save detailed results ---
    output = {"summary": summary, "results": results}
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
