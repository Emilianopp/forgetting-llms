from __future__ import annotations

import json
from pathlib import Path
from typing import Any


UNREASONABLE_MODEL_MAX_LENGTH = 10**9


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _collect_int_candidates(payload: Any, key_path: list[str] | None = None) -> list[int]:
    key_path = key_path or []
    candidates: list[int] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            lower_key = str(key).lower()
            next_path = key_path + [lower_key]
            if isinstance(value, int):
                if lower_key in {
                    "model_max_length",
                    "max_position_embeddings",
                    "n_positions",
                    "max_sequence_length",
                    "seq_length",
                    "max_seq_len",
                    "sliding_window",
                    "original_max_position_embeddings",
                }:
                    candidates.append(value)
            else:
                candidates.extend(_collect_int_candidates(value, next_path))
    elif isinstance(payload, list):
        for item in payload:
            candidates.extend(_collect_int_candidates(item, key_path))
    return candidates


def _reasonable_lengths(candidates: list[int]) -> list[int]:
    return [value for value in candidates if 0 < value < UNREASONABLE_MODEL_MAX_LENGTH]


def infer_model_max_len(model_ref: str) -> int | None:
    model_path = Path(model_ref).expanduser()
    if not model_path.exists() or not model_path.is_dir():
        return None

    tokenizer_config = _load_json(model_path / "tokenizer_config.json")
    config = _load_json(model_path / "config.json")
    generation_config = _load_json(model_path / "generation_config.json")

    tokenizer_candidates = _reasonable_lengths(_collect_int_candidates(tokenizer_config))
    config_candidates = _reasonable_lengths(_collect_int_candidates(config))
    generation_candidates = _reasonable_lengths(_collect_int_candidates(generation_config))

    if tokenizer_candidates:
        tokenizer_max = max(tokenizer_candidates)
        config_max = max(config_candidates) if config_candidates else None
        generation_max = max(generation_candidates) if generation_candidates else None

        # Tokenizer metadata often advertises a longer logical context than the
        # checkpoint config that vLLM will actually accept. Clamp to the
        # strongest config-derived limit when one is available so callers get a
        # safe default instead of a validation error.
        hard_limit = config_max if config_max is not None else generation_max
        if hard_limit is not None:
            return min(tokenizer_max, hard_limit)
        return tokenizer_max
    all_candidates = config_candidates + generation_candidates
    if all_candidates:
        return max(all_candidates)
    return None
