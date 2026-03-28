#!/usr/bin/env python3
"""Run a LAB-Bench evaluation against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", default="LitQA2", help="LAB-Bench eval name, for example LitQA2.")
    parser.add_argument("--model-name", required=True, help="Served model name exposed by the endpoint.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="OpenAI-compatible API key.",
    )
    parser.add_argument("--output-path", required=True, help="Where to write the raw LAB-Bench results JSON.")
    parser.add_argument("--threads", type=int, default=20, help="Concurrency for LAB-Bench evaluation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max completion tokens.")
    return parser.parse_args()


def extract_field(item: object, *names: str) -> object | None:
    if isinstance(item, dict):
        for name in names:
            if name in item and item[name] not in (None, ""):
                return item[name]
        return None
    for name in names:
        value = getattr(item, name, None)
        if value not in (None, ""):
            return value
    return None


def render_choices(choices: object) -> str:
    if not choices:
        return ""
    if isinstance(choices, dict):
        items = choices.items()
    else:
        items = enumerate(choices, start=1)
    lines = []
    for key, value in items:
        label = str(key)
        if label.isdigit():
            label = chr(ord("A") + int(label) - 1)
        lines.append(f"{label}. {value}")
    return "\n".join(lines)


def build_prompt(item: object) -> str:
    question = extract_field(item, "question", "prompt", "problem", "query")
    choices = extract_field(item, "choices", "options", "answers")
    context = extract_field(item, "context", "background", "passage")
    prompt_parts = [
        "Answer the multiple-choice question with the single best option letter.",
    ]
    if context:
        prompt_parts.append(f"Context:\n{context}")
    if question:
        prompt_parts.append(f"Question:\n{question}")
    rendered_choices = render_choices(choices)
    if rendered_choices:
        prompt_parts.append(f"Choices:\n{rendered_choices}")
    prompt_parts.append("Respond with only the final option letter.")
    return "\n\n".join(prompt_parts)


def extract_answer_letter(text: str) -> str:
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        return match.group(1)
    return text.strip()


async def main_async(args: argparse.Namespace) -> int:
    if not args.base_url:
        raise RuntimeError("LAB-Bench evaluation requires --base-url or OPENAI_BASE_URL/OPENAI_API_BASE.")

    try:
        import labbench
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Missing dependency: labbench") from exc
    if not hasattr(labbench, "Eval") or not hasattr(labbench, "Evaluator"):
        raise RuntimeError(
            "Installed 'labbench' package is not the Future-House LAB-Bench client "
            f"(module path: {getattr(labbench, '__file__', 'unknown')}). "
            "Install the LAB-Bench package from the Future-House repo."
        )

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Missing dependency: openai") from exc

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    eval_name = getattr(labbench.Eval, args.eval)
    evaluator = labbench.Evaluator(eval_name)

    async def agent_fn(item: object) -> str:
        prompt = build_prompt(item)
        completion = await client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        content = completion.choices[0].message.content or ""
        return extract_answer_letter(content)

    results = await evaluator.score_agent(agent_fn, n_threads=args.threads)
    with output_path.open("w") as handle:
        json.dump(results, handle, indent=2, default=str)
        handle.write("\n")
    print(f"Wrote LAB-Bench results to {output_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
