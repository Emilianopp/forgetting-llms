"""Unified reward function for multi-domain VeRL GRPO training.

Routes to the correct grading logic based on data_source field.
Reuses extraction logic from math_reward.py for math domains.
Code and QA domains use simple exact-match stubs.

VeRL custom reward function interface:
    def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
"""

import re


# --- Math grading (shared by gsm8k and math) ---

def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} in model output."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_answer_after_hash(text: str) -> str | None:
    """Extract answer after #### markers (GSM8K format)."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def normalize_number(s: str) -> str | None:
    """Normalize a number string for comparison."""
    s = s.strip().replace(",", "").replace(" ", "")
    s = re.sub(r"\.0+$", "", s)
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


def math_score(solution_str: str, ground_truth: str) -> float:
    """Score a math response (gsm8k or MATH)."""
    model_answer = extract_boxed_answer(solution_str)
    if model_answer is None:
        model_answer = extract_answer_after_hash(solution_str)
    if model_answer is None:
        return 0.0

    norm_model = normalize_number(model_answer)
    norm_gt = normalize_number(ground_truth)

    if norm_model is not None and norm_gt is not None and norm_model == norm_gt:
        return 1.0

    # Fallback: exact string match (handles symbolic MATH answers like fractions)
    if model_answer.strip() == ground_truth.strip():
        return 1.0

    return 0.0


# --- Code grading (stub) ---

def code_score(solution_str: str, ground_truth: str) -> float:
    """Score a code response. Stub: exact match on output."""
    extracted = solution_str.strip()
    expected = ground_truth.strip()
    return 1.0 if extracted == expected else 0.0


# --- QA grading (alias-aware for TriviaQA) ---

def extract_answer_after_marker(text: str) -> str:
    """Extract answer after 'The answer is:' marker."""
    match = re.search(r"[Tt]he answer is:\s*(.+?)(?:\.|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: take last line as the answer
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else text.strip()


def normalize_qa(s: str) -> str:
    """Normalize a QA answer for comparison."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = normalize_qa(prediction).split()
    ref_tokens = normalize_qa(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_score(solution_str: str, ground_truth: str) -> float:
    """Score a QA response against |||‑separated aliases.

    Checks: exact match (normalized), containment, token F1 >= 0.5.
    """
    model_answer = extract_answer_after_marker(solution_str)
    norm_model = normalize_qa(model_answer)

    aliases = ground_truth.split("|||")
    for alias in aliases:
        norm_alias = normalize_qa(alias)
        if not norm_alias:
            continue
        # Exact match
        if norm_model == norm_alias:
            return 1.0
        # Containment (model answer contains alias or vice versa)
        if norm_alias in norm_model or norm_model in norm_alias:
            return 1.0

    # Token F1 fallback — best score across aliases
    best_f1 = max((token_f1(model_answer, a) for a in aliases), default=0.0)
    if best_f1 >= 0.5:
        return 1.0

    return 0.0


# --- Dispatcher ---

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Route to the correct grading logic based on data_source.

    Returns:
        1.0 if correct, 0.0 if incorrect.
    """
    if data_source in ("gsm8k", "math"):
        return math_score(solution_str, ground_truth)
    elif data_source == "codecontest":
        return code_score(solution_str, ground_truth)
    elif data_source in ("triviaqa", "naturalquestions", "nq"):
        return qa_score(solution_str, ground_truth)
    else:
        # Unknown domain — fall back to exact match
        return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
