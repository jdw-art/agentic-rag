"""Run Ragas answer correctness evaluation from pre-generated sample files.

This script:
1. loads a samples CSV/JSONL file that already contains user_input, reference, response
2. skips agent / retrieval execution entirely
3. evaluates answer correctness with Ragas
4. saves aggregated metrics and per-sample details under evaluate/output/
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_PATH = ROOT_DIR / "evaluate" / "output" / "correctness" / "ragas_correctness_eval_20260501_112320_samples.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evaluate" / "output" / "correctness"

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
        from ragas.run_config import RunConfig
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

    return pd, Dataset, evaluate, LangchainLLMWrapper, RunConfig, ChatOpenAI


def _load_metrics(evaluator_llm: Any) -> list[Any]:
    """Load answer correctness metric with compatibility across Ragas versions."""
    try:
        from ragas.metrics.collections import AnswerCorrectness

        return [AnswerCorrectness(llm=evaluator_llm)]
    except Exception:
        pass

    try:
        from ragas.metrics import AnswerCorrectness

        return [AnswerCorrectness(llm=evaluator_llm)]
    except Exception:
        pass

    try:
        from ragas.metrics.collections import answer_correctness

        metric = answer_correctness
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        return [metric]
    except Exception:
        pass

    try:
        from ragas.metrics import answer_correctness

        metric = answer_correctness
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        return [metric]
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the Ragas answer correctness metric. "
            "Please check the installed ragas version."
        ) from exc


@dataclass
class EvalConfig:
    samples_path: Path = DEFAULT_SAMPLES_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
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
    ragas_timeout: int = int(os.getenv("RAGAS_TIMEOUT", "600"))
    ragas_max_retries: int = int(os.getenv("RAGAS_MAX_RETRIES", "2"))
    ragas_max_wait: int = int(os.getenv("RAGAS_MAX_WAIT", "30"))
    ragas_max_workers: int = int(os.getenv("RAGAS_MAX_WORKERS", "4"))
    ragas_batch_size: int = int(os.getenv("RAGAS_BATCH_SIZE", "4"))


def _detect_api_key_source() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY"
    if os.getenv("ARK_API_KEY"):
        return "ARK_API_KEY"
    return "NONE"


def _mask_secret(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 10:
        return "*" * len(secret)
    return f"{secret[:6]}...{secret[-4:]}"


def _load_samples(samples_path: Path) -> list[dict[str, str]]:
    if not samples_path.is_file():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    suffix = samples_path.suffix.lower()
    rows: list[dict[str, str]] = []

    if suffix == ".csv":
        with samples_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_input = (row.get("user_input") or "").strip()
                reference = (row.get("reference") or "").strip()
                response = (row.get("response") or "").strip()
                if user_input and reference and response:
                    rows.append(
                        {
                            "user_input": user_input,
                            "reference": reference,
                            "response": response,
                        }
                    )
    elif suffix == ".jsonl":
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                user_input = str(obj.get("user_input") or "").strip()
                reference = str(obj.get("reference") or "").strip()
                response = str(obj.get("response") or "").strip()
                if user_input and reference and response:
                    rows.append(
                        {
                            "user_input": user_input,
                            "reference": reference,
                            "response": response,
                        }
                    )
    else:
        raise ValueError(f"Unsupported samples file format: {samples_path.suffix}")

    if not rows:
        raise ValueError(f"No valid samples found in: {samples_path}")

    return rows


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
        timeout=config.ragas_timeout,
        max_retries=config.ragas_max_retries,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_run_config(RunConfig: Any, config: EvalConfig) -> Any:
    return RunConfig(
        timeout=config.ragas_timeout,
        max_retries=config.ragas_max_retries,
        max_wait=config.ragas_max_wait,
        max_workers=config.ragas_max_workers,
    )


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
            summary[key] = sum(clean_values) / len(clean_values) if clean_values else None
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
    pd, Dataset, evaluate, LangchainLLMWrapper, RunConfig, ChatOpenAI = _lazy_import_runtime_dependencies()
    config = EvalConfig()
    key_source = _detect_api_key_source()

    print(f"Loading generated samples from: {config.samples_path}")
    rows = _load_samples(config.samples_path)
    print(f"Loaded {len(rows)} samples with user_input/reference/response")
    print(
        "Evaluator config: "
        f"model={config.model}, "
        f"api_key_source={key_source}, "
        f"api_key={_mask_secret(config.api_key)}, "
        f"base_url={config.base_url or '<default>'}"
    )

    ragas_dataset = Dataset.from_list(rows)

    evaluator_llm = _build_evaluator_llm(ChatOpenAI, LangchainLLMWrapper, config)
    metrics = _load_metrics(evaluator_llm)
    run_config = _build_ragas_run_config(RunConfig, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ragas_correctness_from_samples_{timestamp}"

    print("Running Ragas evaluation for: answer_correctness")
    print(
        "Ragas runtime config: "
        f"timeout={config.ragas_timeout}s, "
        f"max_retries={config.ragas_max_retries}, "
        f"max_wait={config.ragas_max_wait}s, "
        f"max_workers={config.ragas_max_workers}, "
        f"batch_size={config.ragas_batch_size}"
    )
    result = evaluate(
        ragas_dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
        batch_size=config.ragas_batch_size,
    )

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
