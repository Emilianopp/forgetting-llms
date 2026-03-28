#!/usr/bin/env python3
"""Run the RG-mix benchmark against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import urllib.error
import urllib.request

from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rg-mix-root", default=None, help="Optional path containing rg_mix_env.py")
    parser.add_argument("--model-name", required=True, help="Served model name for the endpoint")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for an existing endpoint")
    parser.add_argument("--model-path", default=None, help="Optional local model path; if set, this script starts a local vLLM server")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-eval", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--server-log", default=None)
    parser.add_argument(
        "--server-python",
        default=None,
        help="Optional Python interpreter used to launch the local vLLM server. Defaults to VLLM_SERVER_PYTHON or a detected project env.",
    )
    args = parser.parse_args()
    if bool(args.base_url) == bool(args.model_path):
        parser.error("Provide exactly one of --base-url or --model-path.")
    return args


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
        match = re.search(r"<answer>(.*?)</answer>", tail, re.DOTALL)
        if match:
            return match.group(1).strip()
        return tail
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def load_rg_mix_env(root: Path):
    sys.path.insert(0, str(root))
    import rg_mix_env  # type: ignore

    return rg_mix_env


def build_env(args: argparse.Namespace):
    if args.rg_mix_root:
        rg_mix_root = Path(args.rg_mix_root).expanduser().resolve()
        rg_mix_env = load_rg_mix_env(rg_mix_root)
        return rg_mix_env.RGMixEnv(
            num_train_examples=100,
            num_eval_examples=args.num_eval,
            seed=args.seed,
            dataset_path=args.dataset_path,
        )

    import verifiers as vf

    return vf.load_environment(
        "rg-mix-env",
        num_train_examples=100,
        num_eval_examples=args.num_eval,
        seed=args.seed,
        dataset_path=args.dataset_path,
    )


def models_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid OpenAI-compatible base URL: {base_url!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        models_path = path + "/models"
    elif path:
        models_path = path + "/v1/models"
    else:
        models_path = "/v1/models"
    return parsed._replace(path=models_path, params="", query="", fragment="").geturl()


def ensure_endpoint_ready(base_url: str, api_key: str, model_name: str) -> None:
    url = models_url(base_url)
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"Endpoint preflight failed: GET {url} returned HTTP {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the OpenAI-compatible endpoint at {url}. "
            "Make sure the local vLLM server is running and use 127.0.0.1 rather than localhost on cluster nodes."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Endpoint preflight returned invalid JSON from {url}") from exc

    available_models = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    print(f"RG-mix endpoint ready: {url}")
    if available_models:
        print(f"RG-mix served models: {', '.join(available_models[:8])}")
    if model_name not in available_models:
        print(
            f"RG-mix warning: requested model {model_name!r} not listed by /models; "
            "generation may fail if the served model name does not match."
        )
    sys.stdout.flush()


def health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def normalize_local_client_env() -> None:
    no_proxy_value = "127.0.0.1,localhost,::1"
    os.environ["NO_PROXY"] = no_proxy_value
    os.environ["no_proxy"] = no_proxy_value
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def python_has_vllm(python_exe: str) -> bool:
    try:
        subprocess.run(
            [python_exe, "-c", "import vllm"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return True


def resolve_vllm_server_python(args: argparse.Namespace) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        args.server_python,
        os.environ.get("VLLM_SERVER_PYTHON"),
        os.path.expanduser("~/scratch/forgetting-llms/.venv/bin/python"),
        str(repo_root / ".venv" / "bin" / "python"),
        sys.executable,
        shutil.which("python3"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        expanded = str(Path(candidate).expanduser()) if "/" in candidate else candidate
        resolved = shutil.which(expanded) if "/" not in expanded else expanded
        if not resolved or resolved in seen or not Path(resolved).exists():
            continue
        seen.add(resolved)
        if python_has_vllm(resolved):
            return resolved
    raise RuntimeError(
        "Could not find a Python interpreter with vllm installed for the local RG-mix server. "
        "Set --server-python or VLLM_SERVER_PYTHON to something like "
        "~/scratch/forgetting-llms/.venv/bin/python."
    )


def start_local_vllm_server(args: argparse.Namespace, output_dir: Path) -> tuple[subprocess.Popen[bytes], Path]:
    run_root = output_dir / "_server"
    run_root.mkdir(parents=True, exist_ok=True)
    server_log = Path(args.server_log).expanduser().resolve() if args.server_log else run_root / "vllm_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server_python = resolve_vllm_server_python(args)

    env = os.environ.copy()
    env["VLLM_USE_V1"] = env.get("VLLM_SERVER_USE_V1", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    enforce_eager = env.get("VLLM_SERVER_ENFORCE_EAGER", "1")

    served_model_name = args.served_model_name or args.model_name
    cmd = [
        server_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(Path(args.model_path).expanduser().resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--served-model-name",
        served_model_name,
    ]
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if enforce_eager == "1":
        cmd.append("--enforce-eager")
    if os.environ.get("VLLM_SERVER_EXTRA_ARGS"):
        cmd.extend(shlex.split(os.environ["VLLM_SERVER_EXTRA_ARGS"]))

    print("RG-mix auto-starting local vLLM server")
    print(f"RG-mix server log: {server_log}")
    print(f"RG-mix server python: {server_python}")
    print(f"RG-mix server model path: {args.model_path}")
    print(f"RG-mix served model name: {served_model_name}")
    print(f"RG-mix server command: {' '.join(shlex.quote(part) for part in cmd)}")
    sys.stdout.flush()

    handle = server_log.open("w")
    process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    setattr(process, "_forgetting_llms_log_handle", handle)
    return process, server_log


def wait_for_local_vllm_server(
    process: subprocess.Popen[bytes],
    server_log: Path,
    host: str,
    port: int,
    timeout: int,
    api_key: str,
    model_name: str,
) -> None:
    url = health_url(host, port)
    started = time.monotonic()
    print(f"Waiting for local vLLM server: {url}")
    sys.stdout.flush()
    while True:
        if process.poll() is not None:
            tail = ""
            if server_log.exists():
                tail = server_log.read_text(errors="replace")[-4000:]
            raise RuntimeError(
                f"Local vLLM server exited before becoming ready. Check {server_log}.\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() - started >= timeout:
            raise RuntimeError(f"Timed out waiting for local vLLM server after {timeout}s. Check {server_log}.")
        time.sleep(5)
    ensure_endpoint_ready(f"http://{host}:{port}/v1", api_key, model_name)


def stop_local_vllm_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    finally:
        handle = getattr(process, "_forgetting_llms_log_handle", None)
        if handle is not None:
            handle.close()


async def main_async(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalize_local_client_env()
    ensure_endpoint_ready(args.base_url, args.api_key, args.model_name)

    env = build_env(args)
    eval_ds = list(env.get_eval_dataset())
    print(
        f"RG-mix eval loaded: examples={len(eval_ds)} concurrency={args.concurrency} "
        f"temperature={args.temperature} top_p={args.top_p}"
    )
    sys.stdout.flush()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)

    async def generate_one(idx: int, row: dict[str, Any]) -> tuple[int, str]:
        async with sem:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=row["prompt"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            text = response.choices[0].message.content or ""
            return idx, text

    tasks = [asyncio.create_task(generate_one(i, row)) for i, row in enumerate(eval_ds)]
    outputs: list[tuple[int, str]] = []
    total_requests = len(tasks)
    completed = 0
    started = time.monotonic()
    for task in asyncio.as_completed(tasks):
        try:
            outputs.append(await task)
        except Exception as exc:
            raise RuntimeError(
                f"RG-mix generation failed after {completed}/{total_requests} completed requests. "
                f"Check that the endpoint is still reachable at {args.base_url} "
                f"and that model={args.model_name!r} is being served."
            ) from exc
        completed += 1
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = completed / elapsed
        sys.stdout.write(
            f"\rRG-mix generate: {completed}/{total_requests} elapsed={elapsed:0.1f}s rate={rate:0.2f} req/s"
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    outputs.sort(key=lambda item: item[0])

    total_correct = 0
    total = 0
    per_task = defaultdict(lambda: {"correct": 0, "total": 0})
    details: list[dict[str, Any]] = []

    for idx, completion_text in outputs:
        row = eval_ds[idx]
        task = row["task"]
        answer_idx = int(row["answer"])
        vid, entry_idx = env._entry_map[answer_idx]
        ds = env._variant_datasets[vid]
        entry = env._entries_cache[answer_idx]

        extracted = extract_answer(completion_text)
        score = 0.0
        try:
            score = ds.score_answer(answer=extracted, entry=entry)
        except Exception:
            score = 0.0
        if score < 0.5:
            try:
                score = max(score, ds.score_answer(answer=completion_text, entry=entry))
            except Exception:
                pass

        correct = 1 if score >= 0.5 else 0
        total_correct += correct
        total += 1
        per_task[task]["correct"] += correct
        per_task[task]["total"] += 1
        details.append(
            {
                "idx": idx,
                "task": task,
                "correct": correct,
                "score": float(score),
                "completion": completion_text,
                "extracted_answer": extracted,
            }
        )

    overall = total_correct / total if total else 0.0
    metrics = {
        "pass@1": overall,
        "overall_pass_at_1": overall,
        "total_correct": total_correct,
        "total": total,
        "per_task": {
            task: {"pass@1": values["correct"] / values["total"], **values}
            for task, values in sorted(per_task.items())
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "all.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.model_path:
        args.base_url = f"http://{args.host}:{args.port}/v1"

    server_process: subprocess.Popen[bytes] | None = None
    server_log: Path | None = None
    try:
        if args.model_path:
            server_process, server_log = start_local_vllm_server(
                args,
                Path(args.output_dir).expanduser().resolve(),
            )
            wait_for_local_vllm_server(
                server_process,
                server_log,
                args.host,
                args.port,
                args.startup_timeout,
                args.api_key,
                args.model_name,
            )
        return asyncio.run(main_async(args))
    finally:
        stop_local_vllm_server(server_process)


if __name__ == "__main__":
    raise SystemExit(main())
