"""Run Ragas evaluation against this project's Agentic RAG pipeline.

This script:
1. loads the 10-row CSV dataset from data/test/labor_law_ragas_dataset.csv
2. calls backend.agent.chat_with_agent() for each sample
3. extracts retrieved contexts from rag_trace["retrieved_chunks"]
4. evaluates faithfulness, context_recall, and context_precision with Ragas
5. saves detailed samples and aggregated metrics under evaluate/output/
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "test" / "labor_law_ragas_dataset.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evaluate" / "output"

if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")
os.environ["LANGSMITH_TRACING"] = "false"


def _lazy_import_runtime_dependencies():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pandas. Install it with `uv add pandas` or `pip install pandas`."
        ) from exc

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Install it with `uv add datasets` or `pip install datasets`."
        ) from exc

    try:
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ragas. Install it with `uv add ragas` or `pip install ragas`."
        ) from exc

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: langchain-openai. Install it with `uv sync` or `pip install langchain-openai`."
        ) from exc

    return pd, Dataset, evaluate, LangchainLLMWrapper, ChatOpenAI


def _load_metrics(evaluator_llm: Any) -> list[Any]:
    """Load metrics with compatibility across common Ragas versions."""
    try:
        from ragas.metrics import Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference

        return [
            Faithfulness(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
        ]
    except Exception:
        pass

    try:
        from ragas.metrics.collections import Faithfulness, ContextRecall, ContextPrecision

        return [
            Faithfulness(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
        ]
    except Exception:
        pass

    try:
        from ragas.metrics import faithfulness, context_recall, context_precision

        return [faithfulness, context_recall, context_precision]
    except Exception as exc:
        raise RuntimeError(
            "Unable to import Ragas metrics for faithfulness/context_recall/context_precision. "
            "Please check the installed ragas version."
        ) from exc


@dataclass
class EvalConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    user_id: str = "ragas_eval_user"
    answer_model: str = os.getenv("MODEL") or "gpt-4o-mini"
    model: str = os.getenv("GRADE_MODEL") or os.getenv("MODEL") or "gpt-4o-mini"
    api_key: str = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ARK_API_KEY")
        or ""
    )
    base_url: str | None = (
        os.getenv("OPENAI_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("BASE_URL")
        or None
    )


def _load_reference_samples(dataset_path: Path) -> list[dict[str, str]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            user_input = (row.get("user_input") or "").strip()
            reference = (row.get("reference") or "").strip()
            if not user_input or not reference:
                continue
            rows.append({"user_input": user_input, "reference": reference})

    if not rows:
        raise ValueError(f"No valid rows found in dataset: {dataset_path}")

    return rows


def _extract_retrieved_contexts(rag_trace: dict[str, Any] | None) -> list[str]:
    if not isinstance(rag_trace, dict):
        return []

    chunks = rag_trace.get("retrieved_chunks")
    if not isinstance(chunks, list):
        return []

    contexts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "").strip()
        if text:
            contexts.append(text)
    return contexts


def _build_evaluator_llm(ChatOpenAI: Any, LangchainLLMWrapper: Any, config: EvalConfig) -> Any:
    if not config.api_key:
        raise RuntimeError(
            "No evaluator API key found in .env. Set OPENAI_API_KEY or ARK_API_KEY before running."
        )

    llm = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def _build_answer_llm(ChatOpenAI: Any, config: EvalConfig) -> Any:
    if not config.api_key:
        raise RuntimeError(
            "No answer model API key found in .env. Set OPENAI_API_KEY or ARK_API_KEY before running."
        )

    return ChatOpenAI(
        model=config.answer_model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0.3,
    )


def _answer_with_project_retrieval(ChatOpenAI: Any, config: EvalConfig, question: str) -> tuple[str, dict[str, Any] | None, str]:
    from backend.tools import get_last_rag_context, reset_tool_call_guards, search_knowledge_base

    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    retrieved_text = search_knowledge_base.invoke({"query": question})
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if isinstance(rag_context, dict) else None

    answer_llm = _build_answer_llm(ChatOpenAI, config)
    system_prompt = (
        "You are a helpful legal assistant. "
        "Answer strictly based on the retrieved context. "
        "If the retrieved context is insufficient, say you don't know. "
        "Do not fabricate facts."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{retrieved_text}\n\n"
        "Please provide a concise answer grounded only in the retrieved context."
    )
    response = answer_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return str(getattr(response, "content", response) or "").strip(), rag_trace, "retrieval_fallback"


def _run_agent_samples(samples: list[dict[str, str]], user_id: str, ChatOpenAI: Any, config: EvalConfig) -> list[dict[str, Any]]:
    chat_with_agent = None
    run_mode = "agent"
    agent_import_error = None
    try:
        chat_with_agent = importlib.import_module("agent").chat_with_agent
    except Exception as exc:
        run_mode = "retrieval_fallback"
        agent_import_error = repr(exc)
        print(f"Agent import failed, falling back to project retrieval pipeline: {agent_import_error}")

    results: list[dict[str, Any]] = []
    total = len(samples)
    for idx, sample in enumerate(samples, 1):
        question = sample["user_input"]
        session_id = f"ragas_eval_{uuid4().hex}"
        print(f"[{idx}/{total}] Running agent for question: {question}")

        if chat_with_agent is not None:
            result = chat_with_agent(
                user_text=question,
                user_id=user_id,
                session_id=session_id,
            )
            response = str(result.get("response", "") or "") if isinstance(result, dict) else str(result)
            rag_trace = result.get("rag_trace") if isinstance(result, dict) else None
            sample_mode = run_mode
        else:
            response, rag_trace, sample_mode = _answer_with_project_retrieval(ChatOpenAI, config, question)

        retrieved_contexts = _extract_retrieved_contexts(rag_trace)

        results.append(
            {
                "user_input": question,
                "reference": sample["reference"],
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "retrieved_context_count": len(retrieved_contexts),
                "retrieval_stage": (rag_trace or {}).get("retrieval_stage") if isinstance(rag_trace, dict) else None,
                "retrieval_mode": (rag_trace or {}).get("retrieval_mode") if isinstance(rag_trace, dict) else None,
                "rewrite_needed": (rag_trace or {}).get("rewrite_needed") if isinstance(rag_trace, dict) else None,
                "rewrite_strategy": (rag_trace or {}).get("rewrite_strategy") if isinstance(rag_trace, dict) else None,
                "run_mode": sample_mode,
                "agent_import_error": agent_import_error,
                "rag_trace": rag_trace or {},
            }
        )
    return results


def _save_detailed_outputs(pd: Any, rows: list[dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{stem}_samples.jsonl"
    csv_path = output_dir / f"{stem}_samples.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "user_input": row["user_input"],
                "reference": row["reference"],
                "response": row["response"],
                "retrieved_context_count": row["retrieved_context_count"],
                "retrieved_contexts": json.dumps(row["retrieved_contexts"], ensure_ascii=False),
                "retrieval_stage": row.get("retrieval_stage"),
                "retrieval_mode": row.get("retrieval_mode"),
                "rewrite_needed": row.get("rewrite_needed"),
                "rewrite_strategy": row.get("rewrite_strategy"),
                "run_mode": row.get("run_mode"),
            }
        )

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return jsonl_path, csv_path


def _save_metric_outputs(result: Any, output_dir: Path, stem: str) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{stem}_metrics.json"
    detail_path: Path | None = None

    summary_payload = _extract_metric_summary(result)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    if hasattr(result, "to_pandas"):
        detail_path = output_dir / f"{stem}_metrics_detail.csv"
        result.to_pandas().to_csv(detail_path, index=False, encoding="utf-8-sig")

    return summary_path, detail_path


def _extract_metric_summary(result: Any) -> dict[str, Any]:
    if hasattr(result, "_repr_dict") and isinstance(result._repr_dict, dict):
        return dict(result._repr_dict)

    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict):
            return payload

    if hasattr(result, "_scores_dict") and isinstance(result._scores_dict, dict):
        summary: dict[str, Any] = {}
        for key, values in result._scores_dict.items():
            clean_values = [v for v in values if isinstance(v, (int, float))]
            if clean_values:
                summary[key] = sum(clean_values) / len(clean_values)
            else:
                summary[key] = None
        return summary

    if hasattr(result, "scores") and isinstance(result.scores, list) and result.scores:
        keys = result.scores[0].keys()
        summary = {}
        for key in keys:
            clean_values = [row.get(key) for row in result.scores if isinstance(row.get(key), (int, float))]
            summary[key] = sum(clean_values) / len(clean_values) if clean_values else None
        return summary

    raise RuntimeError("Unable to extract summary metrics from Ragas evaluation result.")


def main() -> None:
    pd, Dataset, evaluate, LangchainLLMWrapper, ChatOpenAI = _lazy_import_runtime_dependencies()
    config = EvalConfig()

    print(f"Loading reference dataset from: {config.dataset_path}")
    reference_samples = _load_reference_samples(config.dataset_path)
    print(f"Loaded {len(reference_samples)} reference samples")

    eval_rows = _run_agent_samples(reference_samples, config.user_id, ChatOpenAI, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ragas_eval_{timestamp}"

    jsonl_path, csv_path = _save_detailed_outputs(pd, eval_rows, config.output_dir, stem)
    print(f"Saved generated evaluation samples to: {jsonl_path}")
    print(f"Saved generated evaluation samples to: {csv_path}")

    ragas_dataset = Dataset.from_list(
        [
            {
                "user_input": row["user_input"],
                "reference": row["reference"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
            }
            for row in eval_rows
        ]
    )

    evaluator_llm = _build_evaluator_llm(ChatOpenAI, LangchainLLMWrapper, config)
    metrics = _load_metrics(evaluator_llm)

    print("Running Ragas evaluation for: faithfulness, context_recall, context_precision")
    result = evaluate(ragas_dataset, metrics=metrics)

    summary_path, detail_path = _save_metric_outputs(result, config.output_dir, stem)
    print(f"Saved aggregated metrics to: {summary_path}")
    if detail_path:
        print(f"Saved per-sample metric details to: {detail_path}")

    summary = _extract_metric_summary(result)

    print("\n=== Aggregated Metrics ===")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
