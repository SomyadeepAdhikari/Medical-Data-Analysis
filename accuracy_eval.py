import os
import re
import argparse
import csv
import datetime as _dt
from typing import Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TEST = os.path.join(_BASE_DIR, "Documents", "test.parquet")
_DEFAULT_TRAIN = os.path.join(_BASE_DIR, "Documents", "train.parquet")


def _default_data_path() -> str:
    # Prefer test.parquet when present (user requested full test evaluation).
    return _DEFAULT_TEST if os.path.exists(_DEFAULT_TEST) else _DEFAULT_TRAIN


def _detect_best_gpu_device() -> str:
    """Return best available accelerator device string: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _apply_llm_overrides(args: argparse.Namespace) -> None:
    """Apply CLI overrides to env vars before importing Rag_final."""
    preset = (getattr(args, "preset", "") or "").lower().strip()

    # Presets set baseline values; explicit CLI args override after.
    if preset == "cpu":
        os.environ["HUGGINGFACE_DEVICE"] = "cpu"
        # Force a small, CPU-friendly model unless user explicitly overrides.
        os.environ["HUGGINGFACE_LLM_MODEL"] = os.getenv(
            "EVAL_CPU_MODEL",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )
    elif preset == "gpu":
        os.environ["HUGGINGFACE_DEVICE"] = _detect_best_gpu_device()
        # Force a stronger model for GPU/MPS evaluation unless user explicitly overrides.
        os.environ["HUGGINGFACE_LLM_MODEL"] = os.getenv(
            "EVAL_GPU_MODEL",
            "Qwen/Qwen2.5-3B-Instruct",
        )

    # Explicit CLI args override preset/env
    if getattr(args, "device", None):
        os.environ["HUGGINGFACE_DEVICE"] = args.device
    if getattr(args, "llm_model", None):
        os.environ["HUGGINGFACE_LLM_MODEL"] = args.llm_model


def _load_rag():
    """Import Rag_final lazily so env overrides take effect."""
    from Rag_final import init_vector_store, answer_query  # type: ignore
    return init_vector_store, answer_query


def _normalize(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_question(raw_text: str) -> str:
    if not raw_text:
        return ""
    marker = "the patient may have"
    lower_text = raw_text.lower()
    idx = lower_text.find(marker)
    question = raw_text
    if idx != -1:
        question = raw_text[:idx].strip()
    # Ensure the question ends with a question mark if it appears to be a prompt
    if question and not question.strip().endswith("?"):
        if "what disease" in question.lower() or "what condition" in question.lower():
            question = question.rstrip(".") + "?"
    return question.strip()


def _extract_prediction(answer_text: str) -> str:
    """Best-effort extraction of the predicted diagnosis from a free-form answer."""
    if not answer_text:
        return ""
    text = answer_text.strip()
    lowered = text.lower()

    # Common patterns in this dataset / model responses
    patterns = [
        r"the patient may have\s+([^\.\n\r]+)",
        r"patient may have\s+([^\.\n\r]+)",
        r"diagnosis\s*:\s*([^\n\r]+)",
        r"answer\s*:\s*([^\n\r]+)",
    ]
    for pat in patterns:
        m = re.search(pat, lowered, flags=re.IGNORECASE)
        if m:
            # Use span on original text for better casing
            span = m.span(1)
            return text[span[0]:span[1]].strip(" \t\r\n:.-")

    # Fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.strip(" \t\r\n:.-")
    return ""


def _score(expected: str, answer_text: str, *, mode: str) -> Tuple[bool, str]:
    """Return (is_correct, predicted_string)."""
    expected_norm = _normalize(expected)
    predicted = _extract_prediction(answer_text)
    predicted_norm = _normalize(predicted)

    if not expected_norm:
        return False, predicted

    if mode == "exact":
        return predicted_norm == expected_norm, predicted
    if mode == "contains":
        # Accept answers that contain the label anywhere
        return expected_norm in _normalize(answer_text), predicted
    if mode == "pred_contains":
        # Accept answers where extracted prediction contains the label
        return expected_norm in predicted_norm, predicted

    raise ValueError(f"Unknown scoring mode: {mode}")


def _evaluate_row(answer_query_fn, retriever, question: str, expected: str, *, scoring: str, eval_target: str) -> Dict[str, Optional[str]]:
    result = answer_query_fn(question, retriever)
    answer = result.get("answer_en") or result.get("answer") or ""
    context = result.get("context") or ""

    if eval_target == "context":
        is_match, predicted = _score(expected, context, mode=scoring)
    else:
        is_match, predicted = _score(expected, answer, mode=scoring)

    return {
        "question": question,
        "expected": expected,
        "predicted": predicted,
        "answer": answer,
        "context": context,
        "is_match": bool(is_match),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG model accuracy on a parquet dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.getenv("EVAL_DATA_PATH", "") or _default_data_path(),
        help="Path to parquet file with columns: diagnosis, text (default prefers Documents/test.parquet if present)",
    )
    parser.add_argument(
        "--preset",
        choices=["", "cpu", "gpu"],
        default=os.getenv("EVAL_PRESET", ""),
        help="Convenience preset: 'gpu' picks an accelerator (mps/cuda if available) and a stronger model; 'cpu' keeps CPU",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("EVAL_DEVICE", ""),
        help="Override HUGGINGFACE_DEVICE for this eval run (cpu|cuda|mps)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("EVAL_LLM_MODEL", ""),
        help="Override HUGGINGFACE_LLM_MODEL for this eval run",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=int(os.getenv("EVAL_SAMPLE_SIZE", "0")),
        help="Number of random rows to evaluate; 0 means evaluate ALL rows (default)",
    )
    parser.add_argument("--max-rows", type=int, default=int(os.getenv("EVAL_MAX_ROWS", "0")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("EVAL_SEED", "42")))
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default=os.getenv("RETRIEVAL_MODE", "dense"))
    parser.add_argument("--k", type=int, default=int(os.getenv("RETRIEVER_K", "3")))
    parser.add_argument(
        "--eval-target",
        choices=["answer", "context"],
        default=os.getenv("EVAL_TARGET", "answer"),
        help="Score against model answer (answer) or retrieved context only (context)",
    )
    parser.add_argument(
        "--scoring",
        choices=["contains", "pred_contains", "exact"],
        default=os.getenv("EVAL_SCORING", "contains"),
        help="contains: label in full text; pred_contains: label in extracted prediction; exact: strict match on extracted prediction",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=int(os.getenv("EVAL_LOG_EVERY", "0")),
        help="Print progress every N evaluated rows (0 disables)",
    )
    parser.add_argument(
        "--report-txt",
        type=str,
        default=os.getenv("EVAL_REPORT_TXT", "eval_report.txt"),
        help="Path to a TXT report file that stores parameters + summary + optional per-entry logs",
    )
    parser.add_argument(
        "--log-each",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a per-entry log line to the TXT report for every evaluated row (default: enabled)",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print a per-entry log line to console for every evaluated row (very verbose)",
    )
    parser.add_argument(
        "--log-max-chars",
        type=int,
        default=int(os.getenv("EVAL_LOG_MAX_CHARS", "220")),
        help="Truncate long question/prediction fields in per-entry logs (0 disables)",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include retrieved context in CSV output (can make CSV very large)",
    )
    parser.add_argument("--output-csv", type=str, default=os.getenv("EVAL_OUTPUT_CSV", ""))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    _apply_llm_overrides(args)

    # Load RAG after env overrides
    init_vector_store, answer_query_fn = _load_rag()

    # Resolve effective runtime parameters (after overrides)
    effective_llm_model = os.getenv("HUGGINGFACE_LLM_MODEL", "")
    effective_device = os.getenv("HUGGINGFACE_DEVICE", "")
    effective_embed_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "")
    effective_collection = os.getenv("CHROMA_COLLECTION", "medical_docs")
    skip_llm = os.getenv("SKIP_LLM", "false").lower() in {"1", "true", "yes"}

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Parquet not found at {args.data_path}")

    df = pd.read_parquet(args.data_path)
    if "diagnosis" not in df.columns or "text" not in df.columns:
        raise ValueError("Parquet must contain 'diagnosis' and 'text' columns")

    if os.getenv("SKIP_LLM", "false").lower() in {"1", "true", "yes"}:
        print("[WARN] SKIP_LLM=true. Accuracy results will be invalid without generation.")

    if len(df) == 0:
        raise ValueError("Parquet dataset is empty")

    # Optional max-rows cap (useful for debugging)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.head(args.max_rows)

    if args.sample_size and args.sample_size > 0:
        sample_size = min(args.sample_size, len(df))
        df_eval = df.sample(n=sample_size, random_state=args.seed)
    else:
        df_eval = df

    retriever = init_vector_store(mode=args.retrieval, k=args.k)

    correct = 0
    evaluated = 0

    def _trunc(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        if args.log_max_chars and args.log_max_chars > 0 and len(s) > args.log_max_chars:
            return s[: args.log_max_chars].rstrip() + "…"
        return s

    def _one_line(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    csv_writer = None
    csv_file = None
    if args.output_csv:
        csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
        fieldnames = ["question", "expected", "predicted", "answer", "is_match"]
        if args.include_context:
            fieldnames.append("context")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    report_file = None
    if args.report_txt:
        report_file = open(args.report_txt, "w", encoding="utf-8")
        report_file.write("Medical Data Analysis RAG - Accuracy Evaluation\n")
        report_file.write(f"Run timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        report_file.write("\n--- Parameters ---\n")
        # Raw args (useful for reproducibility)
        try:
            report_file.write(f"args: {vars(args)}\n")
        except Exception:
            pass
        report_file.write(f"data_path: {args.data_path}\n")
        report_file.write(f"sample_size: {args.sample_size}\n")
        report_file.write(f"max_rows: {args.max_rows}\n")
        report_file.write(f"seed: {args.seed}\n")
        report_file.write(f"retrieval: {args.retrieval}\n")
        report_file.write(f"k: {args.k}\n")
        report_file.write(f"eval_target: {args.eval_target}\n")
        report_file.write(f"scoring: {args.scoring}\n")
        report_file.write(f"output_csv: {args.output_csv or ''}\n")
        report_file.write(f"include_context_in_csv: {bool(args.include_context)}\n")
        report_file.write(f"log_each_to_txt: {bool(args.log_each)}\n")
        report_file.write(f"print_each_to_console: {bool(args.print_each)}\n")
        report_file.write(f"log_max_chars: {args.log_max_chars}\n")
        report_file.write("\n--- Effective Runtime ---\n")
        report_file.write(f"HUGGINGFACE_LLM_MODEL: {effective_llm_model}\n")
        report_file.write(f"HUGGINGFACE_DEVICE: {effective_device}\n")
        report_file.write(f"SKIP_LLM: {skip_llm}\n")
        report_file.write(f"HUGGINGFACE_EMBEDDING_MODEL: {effective_embed_model}\n")
        report_file.write(f"CHROMA_COLLECTION: {effective_collection}\n")
        report_file.write("\n--- Entry Logs ---\n")

    try:
        for i, (_, row) in enumerate(df_eval.iterrows(), start=1):
            question = _extract_question(str(row.get("text", "")))
            expected = str(row.get("diagnosis", ""))
            if not question or not expected:
                continue
            res = _evaluate_row(answer_query_fn, retriever, question, expected, scoring=args.scoring, eval_target=args.eval_target)
            evaluated += 1
            if res["is_match"]:
                correct += 1

            # Per-entry logs (TXT + optional console)
            if report_file is not None and args.log_each:
                report_file.write(
                    f"row={i} expected={_trunc(_one_line(res.get('expected') or ''))} | predicted={_trunc(_one_line(res.get('predicted') or ''))} | match={res.get('is_match')} | question={_trunc(_one_line(res.get('question') or ''))}\n"
                )
            if args.print_each:
                print(
                    f"[ROW {i}] expected={_trunc(_one_line(res.get('expected') or ''))} | predicted={_trunc(_one_line(res.get('predicted') or ''))} | match={res.get('is_match')}"
                )

            if csv_writer is not None:
                out_row = {
                    "question": res.get("question"),
                    "expected": res.get("expected"),
                    "predicted": res.get("predicted"),
                    "answer": res.get("answer"),
                    "is_match": res.get("is_match"),
                }
                if args.include_context:
                    out_row["context"] = res.get("context")
                csv_writer.writerow(out_row)

            if args.log_every and i % args.log_every == 0:
                acc = (correct / evaluated) * 100 if evaluated else 0.0
                print(f"[PROGRESS] {i}/{len(df_eval)} processed | evaluated={evaluated} | acc={acc:.2f}%")
    finally:
        if csv_file is not None:
            csv_file.close()
        if report_file is not None:
            # Summary at end
            report_file.write("\n--- Summary ---\n")
            acc = (correct / evaluated) * 100 if evaluated else 0.0
            report_file.write(f"Total evaluated: {evaluated}\n")
            report_file.write(f"Correct: {correct}\n")
            report_file.write(f"Accuracy: {acc:.2f}%\n")
            report_file.close()

    accuracy = (correct / evaluated) * 100 if evaluated else 0.0

    print("--- Accuracy Report ---")
    print(f"Data: {args.data_path}")
    print(f"Eval target: {args.eval_target} | scoring: {args.scoring} | retrieval={args.retrieval} | k={args.k}")
    print(f"Total evaluated: {evaluated}")
    print(f"Correct (string match): {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    if args.output_csv:
        print(f"Saved detailed results to: {args.output_csv}")
    if args.report_txt:
        print(f"Saved TXT report to: {args.report_txt}")


if __name__ == "__main__":
    main()
