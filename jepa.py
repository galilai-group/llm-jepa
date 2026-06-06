"""LLM-JEPA embedding utilities for the evolutionary loop.

The LLM-JEPA model (arXiv:2509.14252) is the coder model fine-tuned with a joint-embedding
objective over (serialized train pairs, transformation code). We reuse its hidden states at a
chosen layer as a semantic ranking/steering signal. The exact-match oracle in evolve.py stays
PRIMARY; these signals only rank/guide and are weighted << 1.

Two embedding backends sit behind one informal `Embedder` interface (encode / predict_target):

  * HFEmbedder              - transformers forward with output_hidden_states. Verified path; loads
                              a SECOND copy of the weights (separate from the vLLM generator).
  * VllmHiddenStateEmbedder - uses vLLM's hidden-state extraction system (vllm >= 0.18). Can reuse
                              the generator engine (one weight copy) when that engine is built with
                              `extraction_config(...)`; otherwise it spins up its own extraction
                              engine. Coded to the documented example API -- the file format of the
                              example KV connector is version-specific, so ALWAYS run `smoke_check`
                              (evolve.py does this automatically on the vLLM path) before trusting
                              it. If it fails, the prime suspect is `_read_states`.
"""
import glob
import os
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
# Metrics / scoring
# --------------------------------------------------------------------------- #

def cosine(a, b) -> float:
    """Cosine similarity between two vectors; 0.0 if either is the zero vector."""
    a = a.flatten().float()
    b = b.flatten().float()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 0.0
    return torch.dot(a, b).item() / denom


def compose_score(pass_count, avg_cell, prior, behavioral,
                  prior_weight, behavioral_weight,
                  use_prior=True, use_behavioral=True, cell_weight=0.01) -> float:
    """Fitness = pass_count (primary) + cell_weight*avg_cell + small cosine terms.

    The cell-match tiebreaker is ALWAYS retained. (Previously it was dropped whenever the predictor
    was enabled, which conflated "add cosine" with "remove cell-match" and broke the ablation.)
    All cosine/cell weights are << 1, so one extra exact-match (+1) dominates every soft term.
    `prior`/`behavioral` may be None (no embedding available); each is added only when present and
    its flag is on.
    """
    score = pass_count + cell_weight * avg_cell
    if use_prior and prior is not None:
        score += prior_weight * prior
    if use_behavioral and behavioral is not None:
        score += behavioral_weight * behavioral
    return score


def _fmt_grid(grid) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def serialize_task(train_pairs) -> str:
    """Canonical task serialization for JEPA embedding (view-1).

    MUST match the task view used during the LLM-JEPA fine-tune. Kept independent of evolve.py's
    prompt formatting so the two contracts can diverge without coupling. If this drifts from the
    trainer, every cosine becomes off-distribution noise (smoke_check catches only the catastrophic
    cases) -- keep it in sync with the training script.
    """
    blocks = []
    for pair in train_pairs:
        blocks.append("input:\n" + _fmt_grid(pair["input"]))
        blocks.append("output:\n" + _fmt_grid(pair["output"]))
    return "\n".join(blocks)


def build_synthetic_task(train_inputs, predicted_outputs) -> str:
    """Reassemble a task from a candidate's predicted outputs on the real train inputs."""
    pairs = [{"input": i, "output": o} for i, o in zip(train_inputs, predicted_outputs)]
    return serialize_task(pairs)


def prior_score(pred_target_emb, program_emb) -> float:
    """Program-prior: cos(Pred(Enc(task)), Enc(program)). The learned 'right kind of program'
    signal -- the high-value, high-Goodhart term."""
    return cosine(pred_target_emb, program_emb)


def behavioral_score(real_task_emb, synthetic_task_emb) -> float:
    """Behavioral-as-task: cos(Enc(real task), Enc(task rebuilt from candidate outputs)). Grounded
    in what the program DID; note it is largely an embedding-space proxy for avg_cell_match."""
    return cosine(real_task_emb, synthetic_task_emb)


def _is_grid(value) -> bool:
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)


# --------------------------------------------------------------------------- #
# Per-task scorer
# --------------------------------------------------------------------------- #

class JepaScorer:
    """Caches the two task-level embeddings and exposes batched scoring + an opt-in prefilter."""

    def __init__(self, embedder, prior_weight=0.01, behavioral_weight=0.01,
                 use_prior=True, use_behavioral=True):
        self.embedder = embedder
        self.prior_weight = prior_weight
        self.behavioral_weight = behavioral_weight
        self.use_prior = use_prior
        self.use_behavioral = use_behavioral
        self._real_task_emb = None
        self._pred_target_emb = None
        self._train_inputs = None

    def begin_task(self, train_pairs):
        task_str = serialize_task(train_pairs)
        self._real_task_emb = self.embedder.encode([task_str])[0]
        self._pred_target_emb = self.embedder.predict_target([task_str])[0]
        self._train_inputs = [p["input"] for p in train_pairs]

    def _synthetic_for(self, per_pair_predicted) -> Optional[str]:
        if (per_pair_predicted is not None
                and len(per_pair_predicted) == len(self._train_inputs)
                and all(_is_grid(p) for p in per_pair_predicted)):
            return build_synthetic_task(self._train_inputs, per_pair_predicted)
        return None

    def score(self, program, per_pair_predicted):
        """Single-candidate score (used for the handful of seeds). Returns (behavioral, prior)."""
        prior = None
        if self.use_prior:
            prior = prior_score(self._pred_target_emb, self.embedder.encode([program])[0])
        behavioral = None
        if self.use_behavioral:
            synth = self._synthetic_for(per_pair_predicted)
            if synth is not None:
                behavioral = behavioral_score(self._real_task_emb, self.embedder.encode([synth])[0])
        return behavioral, prior

    def score_many(self, programs, per_pair_predicted_list):
        """Batched scoring for a whole generation: two encode() calls instead of 2N. Returns a list
        of (behavioral, prior) aligned with `programs`."""
        priors = [None] * len(programs)
        if self.use_prior and programs:
            prog_embs = self.embedder.encode(programs)
            priors = [prior_score(self._pred_target_emb, prog_embs[i]) for i in range(len(programs))]

        behaviorals = [None] * len(programs)
        if self.use_behavioral and programs:
            synth_texts, synth_idx = [], []
            for i, pred in enumerate(per_pair_predicted_list):
                synth = self._synthetic_for(pred)
                if synth is not None:
                    synth_texts.append(synth)
                    synth_idx.append(i)
            if synth_texts:
                synth_embs = self.embedder.encode(synth_texts)
                for j, i in enumerate(synth_idx):
                    behaviorals[i] = behavioral_score(self._real_task_emb, synth_embs[j])
        return list(zip(behaviorals, priors))

    def prefilter(self, children, keep_n, reserve_frac, rng):
        """OPT-IN, RISKY: cull oversampled children by cosine to the predicted target BEFORE they
        are executed. This can discard a program that would have passed the train pairs, so it is
        off by default; prefer executing all children (cheap) and letting compose_score rank them.
        Keeps the top-by-cosine plus a random exploration reserve drawn from the low-cosine tail.

        `children` is a list of (program, parent, origin) tuples. No-op if not oversampled.
        """
        if len(children) <= keep_n:
            return children
        embs = self.embedder.encode([c[0] for c in children])
        sims = [cosine(self._pred_target_emb, embs[i]) for i in range(len(children))]
        order = sorted(range(len(children)), key=lambda i: sims[i], reverse=True)
        reserve = max(int(reserve_frac * keep_n), 0)
        cut = max(keep_n - reserve, 0)
        top = order[:cut]
        remaining = order[cut:]
        reserve_picks = rng.sample(remaining, min(reserve, len(remaining)))
        return [children[i] for i in top + reserve_picks]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def smoke_check(embedder):
    """Cheap fail-fast for a broken/degenerate embedder -- the most likely failure mode of a
    mis-restored [PRED] row or a mis-read vLLM extraction file. This is NOT a semantic-quality
    check; it only verifies the embedder discriminates inputs and isn't collapsed. Returns
    (ok: bool, report: str)."""
    prog_a = "def solve(grid):\n    return grid\n"
    prog_b = "def solve(grid):\n    return [list(reversed(r)) for r in grid]\n"
    task_a = serialize_task([{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}])
    task_b = serialize_task([{"input": [[0, 0], [0, 0]], "output": [[5, 5], [5, 5]]}])
    ea, eb = embedder.encode([prog_a, prog_b])
    pa, pb = embedder.predict_target([task_a, task_b])
    self_sim = cosine(ea, ea)
    prog_diff = cosine(ea, eb)
    pred_diff = cosine(pa, pb)
    ok = (self_sim > 0.999) and (prog_diff < 0.9995) and (pred_diff < 0.9995)
    report = (f"smoke_check: self_sim={self_sim:.4f} (want >0.999), "
              f"prog_diff={prog_diff:.4f} (want <1), pred_diff={pred_diff:.4f} (want <1) -> "
              f"{'OK' if ok else 'FAIL (embedder looks collapsed / [PRED] or _read_states wrong)'}")
    return ok, report


# --------------------------------------------------------------------------- #
# Backend: transformers (verified)
# --------------------------------------------------------------------------- #

class HFEmbedder:
    """transformers forward path for hidden states. Loads a LoRA adapter onto its base (via peft)
    or a merged checkpoint. Verified; uses a second copy of the weights."""

    def __init__(self, model_path, base_model=None, pred_token="[PRED]", embed_layer=-1,
                 device="cuda", dtype=None, batch_size=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype is None:
            dtype = torch.bfloat16
        self.embed_layer = embed_layer
        self.batch_size = batch_size
        self.device = device
        self.pred_token = pred_token

        # Load the tokenizer FROM THE CHECKPOINT (not the base) so [PRED] keeps its trained id.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "right"

        if base_model:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=dtype, output_hidden_states=True
            )
            # Resize the base to the checkpoint vocab BEFORE attaching the adapter, so a saved
            # [PRED] row (modules_to_save / trainable_token_indices) lines up instead of being
            # clobbered by a post-hoc fresh init.
            if base.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
                base.resize_token_embeddings(len(self.tokenizer))
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, output_hidden_states=True
            )
            if self.model.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        assert pred_token in self.tokenizer.get_vocab(), (
            f"{pred_token!r} not in tokenizer at {model_path!r}. The JEPA checkpoint must save the "
            f"tokenizer with the trained predictor token."
        )
        self.pred_id = self.tokenizer.convert_tokens_to_ids(pred_token)
        self.model.to(device).eval()
        self._warn_if_pred_untrained()

    def _warn_if_pred_untrained(self):
        """Cheap guard: a freshly resized [PRED] row is a small-norm random vector, so its norm is
        usually a strong low outlier vs trained rows. Raise on all-zeros; warn on extreme outlier.
        smoke_check / golden logging is the real validation."""
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight.detach()
            row = emb[self.pred_id].float()
            if torch.allclose(row, torch.zeros_like(row)):
                raise ValueError(
                    f"{self.pred_token} embedding is all zeros -- it was not restored from the JEPA "
                    f"checkpoint. Check modules_to_save / trainable_token_indices in training."
                )
            med = emb.float().norm(dim=1).median().item()
            row_norm = row.norm().item()
            if med > 0 and row_norm < 0.2 * med:
                print(f"[jepa] WARNING: {self.pred_token} row norm {row_norm:.3f} << median token "
                      f"norm {med:.3f}; the trained predictor row may not have loaded. Inspect "
                      f"smoke_check output (run with --jepa_golden_check).")

    @torch.no_grad()
    def _embed(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            hidden = self.model(**enc).hidden_states[self.embed_layer]   # [B, T, d]
            last_idx = enc["attention_mask"].sum(dim=1) - 1              # right padding -> last real token
            gather = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            last = hidden.gather(1, gather).squeeze(1)                   # [B, d]
            out.append(last.float().cpu())
        return torch.cat(out, dim=0)

    def encode(self, texts):
        return self._embed(list(texts))

    def predict_target(self, tasks):
        return self._embed([t + self.pred_token for t in tasks])


# --------------------------------------------------------------------------- #
# Backend: vLLM hidden-state extraction (vllm >= 0.18)
# --------------------------------------------------------------------------- #

def extraction_config(layers, storage_path):
    """Return the kwargs that make a vLLM `LLM(...)` emit hidden states for `layers` (vllm >= 0.18).

    Splat into the engine constructor: `LLM(model=..., **extraction_config([16,24,31], path))`.
    The mechanism is built for spec-decode data generation; pairing it with long-form generation on
    the SAME engine is off-label (hence --jepa_share_engine is opt-in). VERIFY these field names
    against your installed vLLM build -- the example connector and config keys do move.
    """
    return {
        "speculative_config": {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": list(layers)}},
        },
        "kv_transfer_config": {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(storage_path)},
        },
    }


class VllmHiddenStateEmbedder:
    """Hidden states via vLLM's extraction system (vllm >= 0.18).

    Reuses an existing vLLM engine when `engine` is given (one weight copy -- that engine must have
    been built with extraction_config(...)); otherwise builds its own extraction-only engine
    (correct, but a second weight copy, so no memory win vs HF).

    ##################### VERIFY BEFORE TRUSTING #####################
    The example KV connector writes per-request hidden states to `storage_path`; the on-disk naming
    and tensor layout are version-specific. `_read_states` is coded to the documented shape
    [seq_len, n_extracted_layers, hidden]. If `smoke_check` FAILS, fix `_read_states` for your
    connector/version -- it is by far the most likely thing to be wrong. Fall back any time with
    --jepa_backend hf.
    """

    def __init__(self, model_path, storage_path, extract_layers, embed_layer,
                 pred_token="[PRED]", engine=None, batch_size=16,
                 gpu_memory_utilization=0.45, max_model_len=None):
        from transformers import AutoTokenizer

        if not extract_layers:
            raise ValueError("vLLM backend needs explicit --jepa_extract_layers (e.g. 16,24,31); "
                             "embed_layer=-1 is ambiguous for extraction.")
        if embed_layer not in extract_layers:
            raise ValueError(f"--jepa_embed_layer {embed_layer} must be one of the extracted layers "
                             f"{list(extract_layers)}.")
        self.embed_layer = embed_layer
        self.extract_layers = list(extract_layers)
        self._layer_pos = self.extract_layers.index(embed_layer)
        self.storage_path = str(storage_path)
        self.pred_token = pred_token
        self.batch_size = batch_size
        os.makedirs(self.storage_path, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        assert pred_token in self.tokenizer.get_vocab(), (
            f"{pred_token!r} not in tokenizer at {model_path!r}."
        )

        if engine is not None:
            self.engine = engine          # shared with the coder; must carry extraction_config(...)
            self._owns_engine = False
        else:
            from vllm import LLM
            self.engine = LLM(
                model=model_path, dtype="bfloat16", tokenizer=model_path,
                max_model_len=max_model_len, enable_prefix_caching=False,
                gpu_memory_utilization=gpu_memory_utilization,
                **extraction_config(self.extract_layers, self.storage_path),
            )
            self._owns_engine = True

    def _clear_storage(self):
        for f in glob.glob(os.path.join(self.storage_path, "*")):
            try:
                os.remove(f)
            except OSError:
                pass

    def _read_states(self, n):
        """Load the n most-recent per-request hidden-state files in request order.

        ##### VERIFY: connector-specific file format. #####
        Expected per-request tensor shape: [seq_len, len(extract_layers), hidden]. We take the last
        token (prefill of the full prompt -- the [PRED] token for predict_target) at the requested
        layer position. Adapt the key/indexing to your connector if smoke_check fails.
        """
        from safetensors import safe_open

        files = sorted(glob.glob(os.path.join(self.storage_path, "*.safetensors")),
                       key=os.path.getmtime)
        if len(files) < n:
            raise RuntimeError(
                f"Expected {n} hidden-state files in {self.storage_path}, found {len(files)}. The "
                f"example connector's output format likely differs in your vLLM build -- adapt "
                f"VllmHiddenStateEmbedder._read_states, or use --jepa_backend hf."
            )
        vecs = []
        for path in files[-n:]:
            with safe_open(path, framework="pt") as fh:
                key = next(iter(fh.keys()))
                hs = fh.get_tensor(key)               # [seq_len, n_layers, hidden]  (VERIFY layout)
            vecs.append(hs[-1, self._layer_pos, :].float().cpu())
        return torch.stack(vecs, dim=0)

    @torch.no_grad()
    def _embed(self, texts):
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=1, temperature=0.0)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            self._clear_storage()
            self.engine.generate(batch, params, use_tqdm=False)
            out.append(self._read_states(len(batch)))
        return torch.cat(out, dim=0)

    def encode(self, texts):
        return self._embed(list(texts))

    def predict_target(self, tasks):
        return self._embed([t + self.pred_token for t in tasks])


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_embedder(backend, model_path, base_model=None, pred_token="[PRED]", embed_layer=-1,
                   shared_engine=None, storage_path=None, extract_layers=None, batch_size=16,
                   gpu_memory_utilization=0.45, max_model_len=None):
    if backend == "hf":
        return HFEmbedder(model_path, base_model=base_model, pred_token=pred_token,
                          embed_layer=embed_layer, batch_size=batch_size)
    if backend == "vllm":
        return VllmHiddenStateEmbedder(
            model_path, storage_path=storage_path, extract_layers=extract_layers,
            embed_layer=embed_layer, pred_token=pred_token, engine=shared_engine,
            batch_size=batch_size, gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    raise ValueError(f"unknown jepa backend {backend!r} (use 'hf' or 'vllm').")