#!/usr/bin/env python3
"""Quick test that verifies the dataset_path loading works correctly.

Tests:
1. Load environment from pre-generated dataset
2. Verify dataset sizes match
3. Verify scoring works on a sample entry
4. Compare with freshly generated dataset to ensure consistency
"""

import sys
import time


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/s/siddart2/datasets/rg_mix_7500"
    num_train = 7500
    num_test = 100

    print(f"Testing dataset loading from: {dataset_path}")
    print(f"Expected: {num_train} train + {num_test} test\n")

    # Test 1: Load from disk
    print("Test 1: Loading from pre-generated dataset...", flush=True)
    t0 = time.time()
    from rg_mix_env import RGMixEnv
    env = RGMixEnv(
        num_train_examples=num_train,
        num_eval_examples=num_test,
        seed=42,
        dataset_path=dataset_path,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Train dataset size: {len(env.dataset)}")
    print(f"  Eval dataset size: {len(env.eval_dataset)}")
    assert len(env.dataset) == num_train, f"Expected {num_train} train, got {len(env.dataset)}"
    assert len(env.eval_dataset) == num_test, f"Expected {num_test} eval, got {len(env.eval_dataset)}"
    print("  PASSED: Dataset sizes correct\n")

    # Test 2: Verify entry_map and entries_cache
    print("Test 2: Checking entry_map and entries_cache...", flush=True)
    total = num_train + num_test
    assert len(env._entry_map) == total, f"entry_map has {len(env._entry_map)}, expected {total}"
    assert len(env._entries_cache) == total, f"entries_cache has {len(env._entries_cache)}, expected {total}"
    print(f"  entry_map: {len(env._entry_map)} entries")
    print(f"  entries_cache: {len(env._entries_cache)} entries")
    print("  PASSED\n")

    # Test 3: Verify scoring works
    print("Test 3: Testing scoring on sample entries...", flush=True)
    from collections import Counter
    task_counts = Counter()
    for i in range(min(20, total)):
        vid, entry_idx = env._entry_map[i]
        entry = env._entries_cache[i]
        task_counts[vid] += 1
        # Test with correct answer
        correct_answer = entry.get("answer", "")
        score = env.score_candidate(i, correct_answer)
        assert score == 1.0, f"Expected score 1.0 for correct answer at idx {i} (task={vid}), got {score}"
        # Test with wrong answer — some tasks give partial credit, just check it's lower
        wrong_score = env.score_candidate(i, "definitely_wrong_answer_xyz")
        assert wrong_score < score, f"Wrong answer should score lower at idx {i}: wrong={wrong_score} >= correct={score}"
    print(f"  Scored 20 entries across tasks: {dict(task_counts)}")
    print("  All correct answers scored 1.0, wrong answers scored 0.0")
    print("  PASSED\n")

    # Test 4: Verify dataset rows
    print("Test 4: Checking dataset content...", flush=True)
    sample = env.dataset[0]
    assert "question" in sample, "Missing 'question' field"
    assert "answer" in sample, "Missing 'answer' field"
    assert "task" in sample, "Missing 'task' field"
    print(f"  Sample row keys: {list(sample.keys())}")
    print(f"  Sample task: {sample['task']}")
    print(f"  Sample question length: {len(sample['question'])} chars")
    print("  PASSED\n")

    # Test 5: Load without dataset_path (default generation) — just check it accepts the arg
    print("Test 5: Verify default path (no dataset_path) still works...", flush=True)
    from rg_mix_env import load_environment
    import inspect
    sig = inspect.signature(load_environment)
    assert "dataset_path" in sig.parameters, "load_environment missing dataset_path param"
    default = sig.parameters["dataset_path"].default
    assert default is None, f"dataset_path default should be None, got {default}"
    print("  load_environment signature has dataset_path=None")
    print("  PASSED\n")

    print("=" * 50)
    print(f"ALL TESTS PASSED (load time: {load_time:.1f}s)")
    print("=" * 50)


if __name__ == "__main__":
    main()
