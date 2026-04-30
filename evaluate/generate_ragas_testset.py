"""Generate a Ragas testset from local project documents.

Reference:
- Ragas official guide: https://docs.ragas.org.cn/en/stable/getstarted/rag_testset_generation/

This script follows the official generation flow:
1. load local documents
2. initialize an LLM + embeddings
3. wrap them for ragas
4. call TestsetGenerator.generate_with_langchain_docs(...)

By default, testset_size is 10 and can be adjusted via CLI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evaluate" / "output"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Ragas testset from local documents.")
    parser.add_argument(
        "--documents-dir",
        default=str(DEFAULT_DOCUMENTS_DIR),
        help="Directory containing source documents.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_DIR / "ragas_generated_testset.csv"),
        help="Where to save the generated CSV.",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=10,
        help="Number of generated samples. Default is 10.",
    )
    parser.add_argument(
        "--generator-model",
        default=os.getenv("TESTSET_GENERATOR_MODEL") or os.getenv("GRADE_MODEL") or os.getenv("MODEL") or "gpt-4o-mini",
        help="LLM used for testset generation.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("TESTSET_EMBEDDING_MODEL") or "text-embedding-3-small",
        help="Embedding model used by Ragas during generation.",
    )
    return parser.parse_args()


def _lazy_import_deps():
    try:
        from langchain_community.document_loaders import (
            Docx2txtLoader,
            PyPDFLoader,
            TextLoader,
            UnstructuredExcelLoader,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.testset import TestsetGenerator
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for testset generation. "
            "Install ragas and project dependencies in the target environment first."
        ) from exc

    return {
        "Docx2txtLoader": Docx2txtLoader,
        "PyPDFLoader": PyPDFLoader,
        "TextLoader": TextLoader,
        "UnstructuredExcelLoader": UnstructuredExcelLoader,
        "ChatOpenAI": ChatOpenAI,
        "OpenAIEmbeddings": OpenAIEmbeddings,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "TestsetGenerator": TestsetGenerator,
    }


def _install_ragas_token_count_fallback() -> None:
    """Patch ragas token counting so generation does not fail on tiktoken fetch issues."""
    try:
        import ragas.utils as ragas_utils
    except ImportError:
        return

    original = ragas_utils.num_tokens_from_string

    def _safe_num_tokens_from_string(text: str, encoding_name: str = "cl100k_base") -> int:
        try:
            return original(text, encoding_name=encoding_name)
        except Exception:
            stripped = (text or "").strip()
            if not stripped:
                return 0
            # Fallback heuristic:
            # - Chinese-heavy text: count non-space characters
            # - Otherwise: count whitespace-separated tokens
            ascii_chars = sum(1 for ch in stripped if ch.isascii() and not ch.isspace())
            non_ascii_chars = sum(1 for ch in stripped if not ch.isspace()) - ascii_chars
            if non_ascii_chars > ascii_chars:
                return sum(1 for ch in stripped if not ch.isspace())
            return len(stripped.split())

    ragas_utils.num_tokens_from_string = _safe_num_tokens_from_string


def _iter_supported_files(documents_dir: Path) -> Iterable[Path]:
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".txt"}:
            yield path


def _load_documents(documents_dir: Path, deps: dict) -> list:
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    docs = []
    for path in _iter_supported_files(documents_dir):
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = deps["PyPDFLoader"](str(path))
            elif suffix in {".docx", ".doc"}:
                loader = deps["Docx2txtLoader"](str(path))
            elif suffix in {".xlsx", ".xls"}:
                loader = deps["UnstructuredExcelLoader"](str(path))
            else:
                loader = deps["TextLoader"](str(path), encoding="utf-8")

            loaded = loader.load()
        except Exception as exc:
            if suffix == ".pdf":
                print(f"Primary PDF loader failed for {path.name}: {exc}. Falling back to PyPDF2.")
                loaded = _load_pdf_with_pypdf2(path)
            else:
                raise

        for doc in loaded:
            doc.metadata["source_file"] = path.name
            doc.metadata["source_path"] = str(path)
        docs.extend(loaded)

    if not docs:
        raise ValueError(f"No supported documents were loaded from: {documents_dir}")
    return docs


def _load_pdf_with_pypdf2(path: Path) -> list[Document]:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PyPDF2 is required for PDF fallback loading but is not installed."
        ) from exc

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_number,
                    "source": str(path),
                },
            )
        )
    if not docs:
        raise ValueError(f"Fallback PyPDF2 loader could not extract text from {path}")
    return docs


def _build_generator_components(args: argparse.Namespace, deps: dict):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ARK_API_KEY") or ""
    base_url = (
        os.getenv("OPENAI_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("BASE_URL")
        or None
    )
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or ARK_API_KEY in .env for testset generation.")

    llm = deps["ChatOpenAI"](
        model=args.generator_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    embeddings = deps["OpenAIEmbeddings"](
        model=args.embedding_model,
        api_key=api_key,
        base_url=base_url,
    )

    generator_llm = deps["LangchainLLMWrapper"](llm)
    generator_embeddings = deps["LangchainEmbeddingsWrapper"](embeddings)
    return generator_llm, generator_embeddings


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    os.environ["LANGSMITH_TRACING"] = "false"

    args = _parse_args()
    deps = _lazy_import_deps()
    _install_ragas_token_count_fallback()

    documents_dir = Path(args.documents_dir).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from: {documents_dir}")
    docs = _load_documents(documents_dir, deps)
    print(f"Loaded {len(docs)} document chunks/pages for generation")

    print(f"Using generator model: {args.generator_model}")
    print(f"Using embedding model: {args.embedding_model}")
    generator_llm, generator_embeddings = _build_generator_components(args, deps)

    generator = deps["TestsetGenerator"](
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    print(f"Generating ragas testset with size={args.testset_size}")
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=args.testset_size,
    )

    df = testset.to_pandas()
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved generated testset to: {output_path}")
    print(f"Generated rows: {len(df)}")


if __name__ == "__main__":
    main()
