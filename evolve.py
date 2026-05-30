"""AlphaEvolve-style harness for iterating on Python programs that solve ARC-AGI puzzles.

Per task, maintain a population of candidate `solve(grid)` programs. Each iteration:
  1. Sample parents from the top-K of the database (rank-weighted).
  2. Ask an open-source LLM (served via vLLM) for improvements. Hybrid strategy:
     first try SEARCH/REPLACE diffs against the parent; if the diff fails to
     parse or apply, fall back to a full-program rewrite for that slot.
  3. Execute each candidate in a SIGALRM-guarded sandbox, score it against the
     task's train pairs (exact match + cell-level Hamming similarity as a
     tiebreaker gradient), and store it in the database.
  4. Early-stop when a candidate passes every train pair exactly.
  5. After the loop, evaluate the best candidate on the held-out test pairs.

Usage:
    python evolve.py --coder_model <hf-or-local-model> --eval_set evaluation \
        --num_tasks 50 --max_iterations 20 --parallel_parents 2 --batch_size 8
"""
import argparse
import builtins
import copy
import json
import random
import re
import signal
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants and dataclasses
# ---------------------------------------------------------------------------

SEED_PROGRAM = "def solve(grid):\n    return grid\n"

SYSTEM_DIFF = (
    "You are improving a Python program that solves an ARC-AGI puzzle. "
    "The program must define a function `solve(grid)` that takes a 2D list "
    "of integers and returns a 2D list of integers. Use only the Python "
    "standard library. You will be shown the current program and how it "
    "scored on the task's train pairs; propose an improved version by "
    "returning one or more SEARCH/REPLACE blocks that edit the current "
    "program. Each block must match existing code EXACTLY."
)

SYSTEM_FULL = (
    "You are writing a Python program that solves an ARC-AGI puzzle. "
    "Define a function `solve(grid)` that takes a 2D list of integers and "
    "returns a 2D list of integers. Use only the Python standard library. "
    "Return the complete function inside a <solution>...</solution> block."
)

SYSTEM_PLAN = (
    "You are analyzing an ARC-AGI puzzle and a candidate Python solver. "
    "Infer the transformation rule from the train pairs, explain why the "
    "current program fails, and propose a concise implementation strategy. "
    "Do not write code. Return the plan inside a <plan>...</plan> block."
)

DIFF_INSTRUCTIONS = """Return your edits as one or more blocks in this exact format:

<<<<<<< SEARCH
<text that appears verbatim in the current program>
=======
<replacement text>
>>>>>>> REPLACE

The SEARCH text must match a contiguous chunk of the current program character-for-character.
You may emit multiple blocks; they are applied in order. To rewrite the whole program, use a single block whose SEARCH is the entire current program.
Do not include any other prose."""

DIFF_BLOCK_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE",
    re.DOTALL,
)


@dataclass
class PairResult:
    exact: bool
    cell_match: float
    shape_match: bool
    error: Optional[str] = None
    predicted: Any = None  # list[list[int]] for a valid grid, str repr for other returns, None on error


@dataclass
class ScoreResult:
    score: float
    pass_count: int
    avg_cell_match: float
    solved: bool
    num_pairs: int
    per_pair: list
    compile_error: Optional[str] = None


@dataclass
class Candidate:
    id: int
    program: str
    score: float
    pass_count: int
    avg_cell_match: float
    solved: bool
    num_pairs: int
    per_pair: list
    compile_error: Optional[str]
    parent_id: Optional[int]
    iteration: int
    origin: str  # "seed" | "diff" | "full"


# ---------------------------------------------------------------------------
# Sandbox + scoring
# ---------------------------------------------------------------------------


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_("Execution timed out")


def parse_code(generated_text: str) -> str:
    """Pull code out of <solution> tags or markdown fences.

    Handles truncated generations where decoding stopped on the closing tag/fence.
    """
    code_str = generated_text.strip()
    if "<solution>" in code_str:
        code_str = code_str.split("<solution>", 1)[1]
        if "</solution>" in code_str:
            code_str = code_str.split("</solution>", 1)[0]
        code_str = code_str.strip()
    if "```python" in code_str:
        code_str = code_str.split("```python", 1)[1]
        if "```" in code_str:
            code_str = code_str.split("```", 1)[0]
        code_str = code_str.strip()
    elif "```" in code_str:
        code_str = code_str.split("```", 1)[1]
        if "```" in code_str:
            code_str = code_str.split("```", 1)[0]
        code_str = code_str.strip()
    return code_str


def parse_plan(generated_text: str) -> str:
    """Pull planner text out of <plan> tags, tolerating truncated generations."""
    plan_str = generated_text.strip()
    if "<plan>" in plan_str:
        plan_str = plan_str.split("<plan>", 1)[1]
        if "</plan>" in plan_str:
            plan_str = plan_str.split("</plan>", 1)[0]
    return plan_str.strip()


def _coerce_predicted(value) -> Any:
    """Best-effort conversion of `solve()`'s return into a list-of-lists-of-ints.

    Returns the grid if it looks well-formed; otherwise a short repr so the model
    can see what it produced (e.g. `None`, a string, a numpy array).
    """
    try:
        rows = [list(row) for row in value]
    except TypeError:
        try:
            return repr(value)[:200]
        except Exception:
            return "<unreprable>"
    try:
        return [[int(c) for c in row] for row in rows]
    except (TypeError, ValueError):
        try:
            return repr(value)[:200]
        except Exception:
            return "<unreprable>"


def _cell_match_ratio(predicted, expected) -> tuple[float, bool]:
    """Return (match_ratio, shape_match). Ratio is 0.0 on shape mismatch."""
    try:
        pred_rows = list(predicted)
        exp_rows = list(expected)
        if len(pred_rows) != len(exp_rows):
            return 0.0, False
        total = 0
        matches = 0
        for pr, er in zip(pred_rows, exp_rows):
            pr_list = list(pr)
            er_list = list(er)
            if len(pr_list) != len(er_list):
                return 0.0, False
            for pc, ec in zip(pr_list, er_list):
                total += 1
                if pc == ec:
                    matches += 1
        if total == 0:
            return 0.0, True
        return matches / total, True
    except Exception:
        return 0.0, False


def _candidate_namespace() -> dict:
    """Isolated namespace for candidate programs.

    Using the same dict for globals/locals keeps helper defs/imports visible to
    `solve()` without exposing harness globals to generated code.
    """
    return {"__builtins__": dict(vars(builtins))}


def _pairs_have_outputs(pairs: list) -> bool:
    return all("output" in pair for pair in pairs)


def evaluate_program(code_str: str, train_pairs: list, timeout: int = 2) -> ScoreResult:
    """Exec the program, run solve() on every train pair, return per-pair results and aggregate score.

    Unlike the baseline `evaluate_solution` in evaluate_finetuned.py, this does not short-circuit on
    the first failure — we need every pair's partial-credit signal so the LLM has a gradient to climb.
    """
    exec_env = _candidate_namespace()
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        exec(code_str, exec_env, exec_env)
        signal.alarm(0)
    except TimeoutError_:
        signal.alarm(0)
        return ScoreResult(0.0, 0, 0.0, False, len(train_pairs), [], f"Timeout during compile (>{timeout}s)")
    except Exception as e:
        signal.alarm(0)
        return ScoreResult(0.0, 0, 0.0, False, len(train_pairs), [], f"Compile/exec error: {type(e).__name__}: {e}")

    if "solve" not in exec_env:
        return ScoreResult(0.0, 0, 0.0, False, len(train_pairs), [], "No `solve` function defined")

    solve_func = exec_env["solve"]
    per_pair: list[PairResult] = []

    for pair in train_pairs:
        test_input = [list(row) for row in pair["input"]]
        expected = [list(row) for row in pair["output"]]
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            predicted = solve_func(copy.deepcopy(test_input))
            signal.alarm(0)
        except TimeoutError_:
            signal.alarm(0)
            per_pair.append(PairResult(False, 0.0, False, f"Timeout (>{timeout}s)"))
            continue
        except Exception as e:
            signal.alarm(0)
            per_pair.append(PairResult(False, 0.0, False, f"{type(e).__name__}: {e}"))
            continue

        ratio, shape_ok = _cell_match_ratio(predicted, expected)
        exact = shape_ok and ratio == 1.0
        per_pair.append(PairResult(exact, ratio, shape_ok, None, _coerce_predicted(predicted)))

    pass_count = sum(1 for p in per_pair if p.exact)
    avg_cell = (sum(p.cell_match for p in per_pair) / len(per_pair)) if per_pair else 0.0
    score = pass_count + 0.01 * avg_cell
    solved = len(per_pair) > 0 and pass_count == len(per_pair)
    return ScoreResult(
        score=score,
        pass_count=pass_count,
        avg_cell_match=avg_cell,
        solved=solved,
        num_pairs=len(train_pairs),
        per_pair=[asdict(p) for p in per_pair],
        compile_error=None,
    )


# ---------------------------------------------------------------------------
# Diff parsing / application
# ---------------------------------------------------------------------------


def parse_diff_blocks(text: str) -> list[tuple[str, str]]:
    """Extract SEARCH/REPLACE blocks from model output."""
    return [(m.group(1), m.group(2)) for m in DIFF_BLOCK_RE.finditer(text)]


def apply_diff(parent: str, blocks: list[tuple[str, str]]) -> Optional[str]:
    """Apply SEARCH/REPLACE blocks to `parent` sequentially. Returns None if any block fails to match.

    Empty SEARCH means 'replace the whole program' (convenience for small models)."""
    if not blocks:
        return None
    current = parent
    for search, replace in blocks:
        if search == "":
            current = replace
            continue
        if search not in current:
            # Try a forgiving retry: strip leading/trailing whitespace from the search
            stripped = search.strip("\n")
            if stripped and stripped in current:
                current = current.replace(stripped, replace, 1)
                continue
            return None
        current = current.replace(search, replace, 1)
    if current == parent:
        return None
    return current


# ---------------------------------------------------------------------------
# Program database
# ---------------------------------------------------------------------------


class ProgramDatabase:
    def __init__(self, top_k_parents: int = 5):
        self.candidates: list[Candidate] = []
        self.seen: set[str] = set()
        self.top_k_parents = top_k_parents
        self._next_id = 0

    def _key(self, program: str) -> str:
        return program.strip()

    def contains(self, program: str) -> bool:
        return self._key(program) in self.seen

    def add(self, **kwargs) -> Candidate:
        cand = Candidate(id=self._next_id, **kwargs)
        self._next_id += 1
        self.candidates.append(cand)
        self.seen.add(self._key(cand.program))
        return cand

    def best(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.score)

    def top_k(self) -> list[Candidate]:
        return sorted(self.candidates, key=lambda c: c.score, reverse=True)[: self.top_k_parents]

    def sample_parent(self, rng: random.Random) -> Candidate:
        pool = self.top_k()
        if len(pool) == 1:
            return pool[0]
        # rank-weighted: weight ∝ 1 / (rank + 1)
        weights = [1.0 / (i + 1) for i in range(len(pool))]
        return rng.choices(pool, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _format_grid(grid) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _format_grid_block(label: str, grid, indent: int = 4) -> str:
    prefix = " " * indent
    body = "\n".join(f"{prefix}  {line}" for line in _format_grid(grid).split("\n"))
    return f"{prefix}{label}:\n{body}"


def _format_train_pairs(train_pairs: list) -> str:
    lines = []
    for i, pair in enumerate(train_pairs):
        lines.append(f"--- Train pair {i + 1} ---")
        lines.append("Input:")
        lines.append(_format_grid(pair["input"]))
        lines.append("Output:")
        lines.append(_format_grid(pair["output"]))
    return "\n".join(lines)


def _format_pair_feedback(score: ScoreResult) -> str:
    if score.compile_error:
        return f"Program did not compile/run: {score.compile_error}"
    lines = [
        f"Current score: {score.pass_count}/{score.num_pairs} train pairs solved exactly "
        f"(avg cell match {score.avg_cell_match:.3f}).",
        "Per-pair breakdown:",
    ]
    for i, p in enumerate(score.per_pair):
        if p["error"]:
            lines.append(f"  Pair {i + 1}: EXCEPTION — {p['error']}")
            continue
        if p["exact"]:
            lines.append(f"  Pair {i + 1}: EXACT match")
            continue
        if not p["shape_match"]:
            lines.append(f"  Pair {i + 1}: wrong output shape")
        else:
            lines.append(f"  Pair {i + 1}: shape OK, {p['cell_match'] * 100:.1f}% cells correct")
        predicted = p.get("predicted")
        if isinstance(predicted, list) and predicted and isinstance(predicted[0], list):
            lines.append(_format_grid_block("Got", predicted, indent=4))
        elif isinstance(predicted, str):
            lines.append(f"    Got (non-grid): {predicted}")
    return "\n".join(lines)


def _format_plan_section(plan: Optional[str]) -> str:
    if not plan:
        return ""
    return f"Proposed strategy:\n{plan}\n\n"


def build_plan_prompt(tokenizer, train_pairs: list, parent: Candidate) -> str:
    user = (
        f"Task train pairs:\n{_format_train_pairs(train_pairs)}\n\n"
        f"Current program:\n```python\n{parent.program}\n```\n\n"
        f"{_format_pair_feedback(_score_from_candidate(parent))}\n\n"
        "Write a short plan that identifies the likely transformation, notes the "
        "main failure mode(s), and suggests what code changes should be attempted. "
        "Return the plan inside <plan>...</plan>."
    )
    return _apply_chat(tokenizer, SYSTEM_PLAN, user)


def build_diff_prompt(tokenizer, train_pairs: list, parent: Candidate, plan: Optional[str] = None) -> str:
    user = (
        f"Task train pairs:\n{_format_train_pairs(train_pairs)}\n\n"
        f"Current program:\n```python\n{parent.program}\n```\n\n"
        f"{_format_pair_feedback(_score_from_candidate(parent))}\n\n"
        f"{_format_plan_section(plan)}"
        f"{DIFF_INSTRUCTIONS}"
    )
    return _apply_chat(tokenizer, SYSTEM_DIFF, user)


def build_full_prompt(
    tokenizer,
    train_pairs: list,
    parent: Optional[Candidate],
    plan: Optional[str] = None,
) -> str:
    if parent is None:
        user = (
            f"Task train pairs:\n{_format_train_pairs(train_pairs)}\n\n"
            f"{_format_plan_section(plan)}"
            "Write a `solve(grid)` function that reproduces the train outputs. "
            "Return your answer inside <solution>...</solution>."
        )
    else:
        user = (
            f"Task train pairs:\n{_format_train_pairs(train_pairs)}\n\n"
            f"A previous attempt:\n```python\n{parent.program}\n```\n\n"
            f"{_format_pair_feedback(_score_from_candidate(parent))}\n\n"
            f"{_format_plan_section(plan)}"
            "Rewrite `solve(grid)` from scratch to fix the failing pairs. "
            "Return the complete function inside <solution>...</solution>."
        )
    return _apply_chat(tokenizer, SYSTEM_FULL, user)


def _apply_chat(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\n"


def _score_from_candidate(cand: Candidate) -> ScoreResult:
    return ScoreResult(
        score=cand.score,
        pass_count=cand.pass_count,
        avg_cell_match=cand.avg_cell_match,
        solved=cand.solved,
        num_pairs=cand.num_pairs,
        per_pair=cand.per_pair,
        compile_error=cand.compile_error,
    )


# ---------------------------------------------------------------------------
# Proposer (vLLM)
# ---------------------------------------------------------------------------


class Proposer:
    def __init__(
        self,
        coder_llm,
        coder_tokenizer,
        max_new_tokens: int,
        temperature: float,
        planner_llm=None,
        planner_tokenizer=None,
        planner_max_new_tokens: int = 256,
        planner_temperature: float = 0.2,
    ):
        self.coder_llm = coder_llm
        self.coder_tokenizer = coder_tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.planner_llm = planner_llm
        self.planner_tokenizer = planner_tokenizer
        self.planner_max_new_tokens = planner_max_new_tokens
        self.planner_temperature = planner_temperature

    def _generate(
        self,
        llm,
        prompts: list[str],
        stop: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=stop,
        )
        outputs = llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def warm_start(self, train_pairs: list, num_samples: int) -> list[str]:
        """Generate a handful of from-scratch seed programs to avoid starting from identity.

        Uses the full-rewrite prompt with `parent=None`. Returns unique, non-empty programs
        that contain a `def solve` definition.
        """
        if num_samples <= 0:
            return []
        prompt = build_full_prompt(self.coder_tokenizer, train_pairs, parent=None)
        outputs = self._generate(
            self.coder_llm,
            [prompt] * num_samples,
            stop=["</solution>"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        seen: set[str] = set()
        programs: list[str] = []
        for text in outputs:
            code = parse_code(text)
            if not code or "def solve" not in code:
                continue
            key = code.strip()
            if key in seen:
                continue
            seen.add(key)
            programs.append(code)
        return programs

    def propose(
        self,
        train_pairs: list,
        parents: list[Candidate],
        batch_size_per_parent: int,
    ) -> list[tuple[str, Candidate, str]]:
        """Return a list of (child_program, parent, origin) tuples.

        origin ∈ {"diff", "full"} depending on which prompt produced the child.
        """
        plans: list[Optional[str]] = [None] * len(parents)
        if self.planner_llm is not None:
            plan_prompts = [
                build_plan_prompt(self.planner_tokenizer, train_pairs, parent) for parent in parents
            ]
            plan_outputs = self._generate(
                self.planner_llm,
                plan_prompts,
                stop=["</plan>"],
                max_new_tokens=self.planner_max_new_tokens,
                temperature=self.planner_temperature,
            )
            plans = [parse_plan(text) or None for text in plan_outputs]

        # --- Diff pass ---
        diff_prompts: list[str] = []
        diff_parents: list[Candidate] = []
        diff_plans: list[Optional[str]] = []
        for parent, plan in zip(parents, plans):
            prompt = build_diff_prompt(self.coder_tokenizer, train_pairs, parent, plan=plan)
            for _ in range(batch_size_per_parent):
                diff_prompts.append(prompt)
                diff_parents.append(parent)
                diff_plans.append(plan)

        diff_outputs = self._generate(
            self.coder_llm,
            diff_prompts,
            stop=[],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        children: list[tuple[str, Candidate, str]] = []
        fallback_pairs: list[tuple[Candidate, Optional[str]]] = []
        for parent, plan, text in zip(diff_parents, diff_plans, diff_outputs):
            blocks = parse_diff_blocks(text)
            child = apply_diff(parent.program, blocks) if blocks else None
            if child is None:
                # Maybe the model ignored the diff format and dropped a full program.
                fallback_code = parse_code(text)
                if fallback_code and fallback_code.strip() != parent.program.strip() and "def solve" in fallback_code:
                    children.append((fallback_code, parent, "full"))
                    continue
                fallback_pairs.append((parent, plan))
                continue
            children.append((child, parent, "diff"))

        # --- Full-rewrite fallback pass (only for slots the diff path dropped) ---
        if fallback_pairs:
            full_prompts = [
                build_full_prompt(self.coder_tokenizer, train_pairs, parent, plan=plan)
                for parent, plan in fallback_pairs
            ]
            full_outputs = self._generate(
                self.coder_llm,
                full_prompts,
                stop=["</solution>"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            for (parent, _plan), text in zip(fallback_pairs, full_outputs):
                code = parse_code(text)
                if code and "def solve" in code:
                    children.append((code, parent, "full"))

        return children


# ---------------------------------------------------------------------------
# Per-task evolution loop
# ---------------------------------------------------------------------------


def evolve_task(
    task_id: str,
    task_data: dict,
    test_pairs: list,
    proposer: Proposer,
    max_iterations: int,
    parallel_parents: int,
    batch_size: int,
    timeout: int,
    top_k_parents: int,
    rng: random.Random,
    warm_start_samples: int = 0,
    scorer=None,
    jepa_predictor: bool = False,
    jepa_steering: bool = False,
    jepa_oversample: int = 3,
    jepa_prior_weight: float = 0.01,
    jepa_behavioral_weight: float = 0.01,
    jepa_exploration_reserve: float = 0.25,
    verbose: bool = False,
) -> dict:
    train_pairs = task_data["train"]
    db = ProgramDatabase(top_k_parents=top_k_parents)

    if scorer is not None and (jepa_predictor or jepa_steering):
        scorer.begin_task(train_pairs)

    def _predicted_grids(result):
        grids = []
        for p in result.per_pair:
            pred = p.get("predicted")
            if isinstance(pred, list) and pred and isinstance(pred[0], list):
                grids.append(pred)
            else:
                return None
        return grids if len(grids) == len(train_pairs) else None

    def _final_score(code, result):
        if scorer is None or not jepa_predictor:
            return result.score
        from jepa import compose_score
        behavioral, prior = scorer.score(code, _predicted_grids(result))
        return compose_score(result.pass_count, prior, behavioral,
                             jepa_prior_weight, jepa_behavioral_weight)

    # Seeds: identity + any warm-start programs from the coder.
    seed_items: list[tuple[str, str]] = [(SEED_PROGRAM, "seed")]
    if warm_start_samples > 0:
        for code in proposer.warm_start(train_pairs, warm_start_samples):
            seed_items.append((code, "warm_start"))

    for code, origin in seed_items:
        if db.contains(code):
            continue
        result = evaluate_program(code, train_pairs, timeout=timeout)
        db.add(
            program=code,
            score=_final_score(code, result),
            pass_count=result.pass_count,
            avg_cell_match=result.avg_cell_match,
            solved=result.solved,
            num_pairs=result.num_pairs,
            per_pair=result.per_pair,
            compile_error=result.compile_error,
            parent_id=None,
            iteration=-1,
            origin=origin,
        )

    score_history = [(0, db.best().score, db.best().pass_count)]

    for it in range(max_iterations):
        if db.best().solved:
            break

        parents = [db.sample_parent(rng) for _ in range(parallel_parents)]
        if scorer is not None and jepa_steering:
            children = proposer.propose(
                train_pairs, parents, batch_size_per_parent=batch_size * jepa_oversample
            )
            keep_n = batch_size * len(parents)
            children = scorer.rerank(children, keep_n, jepa_exploration_reserve, rng)
        else:
            children = proposer.propose(train_pairs, parents, batch_size_per_parent=batch_size)

        seen_in_batch: set[str] = set()
        for child_program, parent, origin in children:
            key = child_program.strip()
            if db.contains(child_program) or key in seen_in_batch:
                continue
            seen_in_batch.add(key)
            result = evaluate_program(child_program, train_pairs, timeout=timeout)
            db.add(
                program=child_program,
                score=_final_score(child_program, result),
                pass_count=result.pass_count,
                avg_cell_match=result.avg_cell_match,
                solved=result.solved,
                num_pairs=result.num_pairs,
                per_pair=result.per_pair,
                compile_error=result.compile_error,
                parent_id=parent.id,
                iteration=it,
                origin=origin,
            )

        best = db.best()
        score_history.append((it + 1, best.score, best.pass_count))
        if verbose:
            print(
                f"  [{task_id}] iter {it + 1}/{max_iterations} "
                f"best={best.pass_count}/{best.num_pairs} "
                f"cell={best.avg_cell_match:.3f} pop={len(db.candidates)}"
            )
        if best.solved:
            break

    best = db.best()
    test_score = evaluate_program(best.program, test_pairs, timeout=timeout) if _pairs_have_outputs(test_pairs) else None

    return {
        "task_id": task_id,
        "best_program": best.program,
        "train_pass_count": best.pass_count,
        "train_num_pairs": best.num_pairs,
        "train_cell_match": best.avg_cell_match,
        "train_solved": best.solved,
        "test_evaluated": test_score is not None,
        "test_pass_count": test_score.pass_count if test_score is not None else None,
        "test_num_pairs": test_score.num_pairs if test_score is not None else len(test_pairs),
        "test_solved": test_score.solved if test_score is not None else None,
        "test_cell_match": test_score.avg_cell_match if test_score is not None else None,
        "test_compile_error": test_score.compile_error if test_score is not None else None,
        "population_size": len(db.candidates),
        "score_history": score_history,
        "best_origin": best.origin,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="AlphaEvolve-style harness for ARC-AGI")
    parser.add_argument("--coder_model", type=str, default=None, help="Coder model path or Hugging Face ID")
    parser.add_argument("--planner_model", type=str, default=None, help="Optional planner model path or Hugging Face ID")
    parser.add_argument("--merged_model", type=str, default=None, help="Legacy alias for --coder_model")
    parser.add_argument("--eval_set", type=str, default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--num_tasks", type=int, default=50)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--parallel_parents", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8, help="Samples per parent per iteration")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--planner_max_new_tokens", type=int, default=256)
    parser.add_argument("--planner_temperature", type=float, default=0.2)
    parser.add_argument("--top_k_parents", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--warm_start_samples", type=int, default=4, help="Coder-generated seed programs per task (0 disables)")
    parser.add_argument("--no_prefix_cache", action="store_true", help="Disable vLLM prefix caching")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="evolve_results.jsonl")
    parser.add_argument("--wandb_project", type=str, default="arc-evolve")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--llm_jepa_model", type=str, default=None,
                        help="Unified LLM-JEPA model (coder generation + embedding source)")
    parser.add_argument("--llm_jepa_base_model", type=str, default=None,
                        help="Base weights when --llm_jepa_model is an unmerged LoRA adapter")
    parser.add_argument("--jepa_predictor", action="store_true",
                        help="Fold JEPA behavioral+prior cosines into fitness (requires --llm_jepa_model)")
    parser.add_argument("--jepa_steering", action="store_true",
                        help="Rerank oversampled children toward Pred(Enc(task)) (requires --llm_jepa_model)")
    parser.add_argument("--jepa_pred_token", type=str, default="[PRED]")
    parser.add_argument("--jepa_embed_layer", type=int, default=-1)
    parser.add_argument("--jepa_oversample", type=int, default=3)
    parser.add_argument("--jepa_prior_weight", type=float, default=0.01)
    parser.add_argument("--jepa_behavioral_weight", type=float, default=0.01)
    parser.add_argument("--jepa_exploration_reserve", type=float, default=0.25)
    args = parser.parse_args()

    if (args.jepa_predictor or args.jepa_steering) and not args.llm_jepa_model:
        parser.error("--jepa_predictor/--jepa_steering require --llm_jepa_model.")

    coder_model = args.coder_model or args.merged_model or args.llm_jepa_model
    if not coder_model:
        parser.error("Provide --coder_model (or --llm_jepa_model / legacy --merged_model).")

    rng = random.Random(args.seed)

    if not args.no_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    # Load model
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"Loading coder model {coder_model} with vLLM...")
    coder_tokenizer = AutoTokenizer.from_pretrained(coder_model)
    coder_llm = LLM(
        model=coder_model,
        dtype="bfloat16",
        tokenizer=coder_model,
        max_model_len=args.max_model_len,
        enable_prefix_caching=not args.no_prefix_cache,
    )

    planner_tokenizer = None
    planner_llm = None
    if args.planner_model:
        print(f"Loading planner model {args.planner_model} with vLLM...")
        planner_tokenizer = AutoTokenizer.from_pretrained(args.planner_model)
        planner_llm = LLM(
            model=args.planner_model,
            dtype="bfloat16",
            tokenizer=args.planner_model,
            max_model_len=args.max_model_len,
            enable_prefix_caching=not args.no_prefix_cache,
        )

    proposer = Proposer(
        coder_llm=coder_llm,
        coder_tokenizer=coder_tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        planner_llm=planner_llm,
        planner_tokenizer=planner_tokenizer,
        planner_max_new_tokens=args.planner_max_new_tokens,
        planner_temperature=args.planner_temperature,
    )

    scorer = None
    if args.jepa_predictor or args.jepa_steering:
        from jepa import JepaLLM, JepaScorer
        print(f"Loading LLM-JEPA embedder {args.llm_jepa_model} (HF)...")
        jepa_llm = JepaLLM(
            model_path=args.llm_jepa_model,
            base_model=args.llm_jepa_base_model,
            pred_token=args.jepa_pred_token,
            embed_layer=args.jepa_embed_layer,
        )
        scorer = JepaScorer(
            jepa_llm,
            prior_weight=args.jepa_prior_weight,
            behavioral_weight=args.jepa_behavioral_weight,
        )

    # Load challenges / solutions
    challenge_file = f"arc-prize-2024/arc-agi_{args.eval_set}_challenges.json"
    solution_file = f"arc-prize-2024/arc-agi_{args.eval_set}_solutions.json"
    with open(challenge_file) as f:
        challenges = json.load(f)
    try:
        with open(solution_file) as f:
            solutions = json.load(f)
    except FileNotFoundError:
        solutions = {}

    task_ids = list(challenges.keys())[: args.num_tasks]

    solved_train = 0
    solved_test = 0
    scored_test_total = 0
    total = 0
    t0 = time.time()

    with open(args.output, "w") as out:
        for task_id in tqdm(task_ids, desc="Tasks"):
            task_data = challenges[task_id]
            if task_id in solutions:
                test_pairs = [
                    {"input": task_data["test"][k]["input"], "output": solutions[task_id][k]}
                    for k in range(len(solutions[task_id]))
                ]
            else:
                test_pairs = task_data["test"]

            result = evolve_task(
                task_id=task_id,
                task_data=task_data,
                test_pairs=test_pairs,
                proposer=proposer,
                max_iterations=args.max_iterations,
                parallel_parents=args.parallel_parents,
                batch_size=args.batch_size,
                timeout=args.timeout,
                top_k_parents=args.top_k_parents,
                rng=rng,
                warm_start_samples=args.warm_start_samples,
                scorer=scorer,
                jepa_predictor=args.jepa_predictor,
                jepa_steering=args.jepa_steering,
                jepa_oversample=args.jepa_oversample,
                jepa_prior_weight=args.jepa_prior_weight,
                jepa_behavioral_weight=args.jepa_behavioral_weight,
                jepa_exploration_reserve=args.jepa_exploration_reserve,
                verbose=args.verbose,
            )
            out.write(json.dumps(result) + "\n")
            out.flush()

            total += 1
            if result["train_solved"]:
                solved_train += 1
            if result["test_evaluated"]:
                scored_test_total += 1
            if result["test_solved"]:
                solved_test += 1

            elapsed = time.time() - t0
            test_status = "N/A" if not result["test_evaluated"] else ("PASS" if result["test_solved"] else "FAIL")
            print(
                f"[{total}/{len(task_ids)}] {task_id} "
                f"train {result['train_pass_count']}/{result['train_num_pairs']} "
                f"test {test_status} "
                f"pop={result['population_size']} origin={result['best_origin']} "
                f"elapsed={elapsed:.0f}s"
            )

            if not args.no_wandb:
                payload = {
                    "task_index": total,
                    "train_solved_cum": solved_train,
                    "test_solved_cum": solved_test,
                    "train_accuracy": solved_train / total,
                    "population_size": result["population_size"],
                }
                if scored_test_total:
                    payload["test_accuracy"] = solved_test / scored_test_total
                    payload["test_scored_tasks"] = scored_test_total
                wandb.log(payload)

    test_summary = (
        f"test_solved {solved_test}/{scored_test_total} ({100 * solved_test / max(scored_test_total, 1):.2f}%)"
        if scored_test_total
        else "test_solved N/A (no test outputs available)"
    )
    print(f"\nFinal: train_solved {solved_train}/{total} ({100 * solved_train / max(total, 1):.2f}%)  {test_summary}")
    if not args.no_wandb:
        payload = {
            "final_train_solved": solved_train,
            "final_test_solved": solved_test,
            "final_train_accuracy": solved_train / max(total, 1),
        }
        if scored_test_total:
            payload["final_test_accuracy"] = solved_test / scored_test_total
            payload["final_test_scored_tasks"] = scored_test_total
        wandb.log(payload)
        wandb.finish()


if __name__ == "__main__":
    main()
