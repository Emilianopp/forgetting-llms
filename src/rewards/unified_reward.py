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


# --- QA grading (stub) ---

def qa_score(solution_str: str, ground_truth: str) -> float:
    """Score a QA response. Stub: normalized exact match."""
    def normalize(s: str) -> str:
        s = s.lower().strip()
        # Remove articles
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # Remove punctuation
        s = re.sub(r"[^\w\s]", "", s)
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    return 1.0 if normalize(solution_str) == normalize(ground_truth) else 0.0


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
    elif data_source in ("naturalquestions", "nq"):
        return qa_score(solution_str, ground_truth)
    else:
        # Unknown domain — fall back to exact match
        return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
