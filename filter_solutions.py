"""Filter distilled solutions by validating against test examples.

Removes solutions that pass training examples but fail on test,
which indicates hardcoded/overfit solutions.
"""

import json
import signal
import sys
from pathlib import Path


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def validate_solution(code_str, examples, timeout=5):
    """Run the solve function against all examples. Returns (success, message)."""
    local_vars = {}
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        exec(code_str, {}, local_vars)
        signal.alarm(0)
    except TimeoutError:
        signal.alarm(0)
        return False, "Timeout during compilation"
    except Exception as e:
        signal.alarm(0)
        return False, f"Compilation error: {e}"

    if "solve" not in local_vars:
        return False, "No 'solve' function defined"

    solve_func = local_vars["solve"]

    for i, ex in enumerate(examples):
        test_input = [list(row) for row in ex["input"]]
        expected = [list(row) for row in ex["output"]]
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            result = solve_func(test_input)
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            return False, f"Timeout on example {i}"
        except Exception as e:
            signal.alarm(0)
            return False, f"Runtime error on example {i}: {e}"

        if isinstance(result, (list, tuple)):
            result = [list(row) if isinstance(row, (list, tuple)) else row for row in result]

        if result != expected:
            return False, f"Output mismatch on example {i}"

    return True, "All examples passed"


def main():
    solutions_path = Path("datasets/distilled_solutions.json")
    traces_path = Path("datasets/distilled_traces.json")

    with open(solutions_path) as f:
        solutions = json.load(f)
    traces = {}
    if traces_path.exists():
        with open(traces_path) as f:
            traces = json.load(f)

    with open("arc-prize-2024/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)
    with open("arc-prize-2024/arc-agi_training_solutions.json") as f:
        test_solutions = json.load(f)

    failed = []
    for task_id, code in solutions.items():
        if task_id not in test_solutions:
            continue
        test_examples = [
            {"input": t["input"], "output": out}
            for t, out in zip(challenges[task_id]["test"], test_solutions[task_id])
        ]
        ok, msg = validate_solution(code, test_examples)
        if not ok:
            failed.append((task_id, msg))

    if not failed:
        print("All solutions pass test validation.")
        return

    print(f"Found {len(failed)} solutions that fail on test examples:\n")
    for task_id, msg in failed:
        print(f"  {task_id}: {msg}")

    print(f"\nRemoving {len(failed)} hardcoded solutions...")
    for task_id, _ in failed:
        del solutions[task_id]
        traces.pop(task_id, None)

    with open(solutions_path, "w") as f:
        json.dump(solutions, f, indent=2)
    with open(traces_path, "w") as f:
        json.dump(traces, f, indent=2)

    print(f"Done. {len(solutions)} solutions remaining.")


if __name__ == "__main__":
    main()
