"""Math reward function for VeRL GRPO training.

Uses GEM's math grading utilities for answer verification.
Falls back to simple string matching if GEM is not available.

VeRL custom reward function interface:
    def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
"""

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} in model output."""
    # Find the last \boxed{...} in the text
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
    # Remove trailing .0, .00 etc.
    s = re.sub(r"\.0+$", "", s)
    try:
        # Try to parse as float and back to string for normalization
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Compute reward score for a math response.

    Tries to extract the answer from \\boxed{} first, then #### format.
    Compares normalized numbers with ground truth.

    Returns:
        1.0 if correct, 0.0 if incorrect.
    """
    # Note: GEM's MathEnv.check_correct uses signal.SIGALRM which fails in
    # VeRL's worker threads. Use direct extraction + number comparison instead.
    # This is sufficient for GSM8K (numeric answers).
    model_answer = extract_boxed_answer(solution_str)
    if model_answer is None:
        model_answer = extract_answer_after_hash(solution_str)
    if model_answer is None:
        return 0.0

    norm_model = normalize_number(model_answer)
    norm_gt = normalize_number(ground_truth)

    if norm_model is not None and norm_gt is not None and norm_model == norm_gt:
        return 1.0

    return 0.0
