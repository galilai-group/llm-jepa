"""LLM-JEPA embedding utilities for the evolutionary loop.

The LLM-JEPA model (arXiv:2509.14252) is the coder model fine-tuned with a joint-embedding
objective over (serialized train pairs, transformation code). We reuse its last-token hidden
states as a semantic fitness signal and a generation-steering signal. The exact-match oracle in
evolve.py stays primary; these signals only rank/guide. See docs/jepa-fitness-design.md and
docs/superpowers/specs/2026-05-30-jepa-flags-design.md.
"""
from typing import Optional

import torch


def cosine(a, b) -> float:
    """Cosine similarity between two vectors; 0.0 if either is the zero vector."""
    a = a.flatten().float()
    b = b.flatten().float()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 0.0
    return torch.dot(a, b).item() / denom


def compose_score(pass_count, prior, behavioral, prior_weight, behavioral_weight) -> float:
    """Fitness = pass_count (primary) + small cosine terms.

    Weights are << 1 so an extra exact-match always dominates the cosine terms (the oracle stays
    primary). `behavioral` is None when the candidate did not produce a grid for every train pair.
    """
    score = pass_count + prior_weight * prior
    if behavioral is not None:
        score += behavioral_weight * behavioral
    return score


def _fmt_grid(grid) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def serialize_task(train_pairs) -> str:
    """Canonical serialization of a task for JEPA embedding (view-1).

    Must match the format used as the task view during the LLM-JEPA fine-tune. Kept independent of
    evolve.py's prompt formatting so the two contracts can diverge without coupling.
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
    """Program-prior: cosine between the predicted target and a candidate program embedding."""
    return cosine(pred_target_emb, program_emb)


def behavioral_score(real_task_emb, synthetic_task_emb) -> float:
    """Behavioral-as-task: cosine between the real task and the candidate-output synthetic task."""
    return cosine(real_task_emb, synthetic_task_emb)


def _is_grid(value) -> bool:
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)


class JepaScorer:
    """Per-task coordinator: caches the two task embeddings and exposes score / rerank."""

    def __init__(self, jepa_llm, prior_weight=0.01, behavioral_weight=0.01):
        self.llm = jepa_llm
        self.prior_weight = prior_weight
        self.behavioral_weight = behavioral_weight
        self._real_task_emb = None
        self._pred_target_emb = None
        self._train_inputs = None

    def begin_task(self, train_pairs):
        task_str = serialize_task(train_pairs)
        self._real_task_emb = self.llm.encode([task_str])[0]
        self._pred_target_emb = self.llm.predict_target([task_str])[0]
        self._train_inputs = [p["input"] for p in train_pairs]

    def score(self, program, per_pair_predicted):
        """Return (behavioral, prior). behavioral is None unless a grid exists for every pair."""
        prog_emb = self.llm.encode([program])[0]
        prior = prior_score(self._pred_target_emb, prog_emb)
        behavioral = None
        if (
            per_pair_predicted is not None
            and len(per_pair_predicted) == len(self._train_inputs)
            and all(_is_grid(p) for p in per_pair_predicted)
        ):
            synth = build_synthetic_task(self._train_inputs, per_pair_predicted)
            synth_emb = self.llm.encode([synth])[0]
            behavioral = behavioral_score(self._real_task_emb, synth_emb)
        return behavioral, prior

    def rerank(self, children, keep_n, reserve_frac, rng):
        """Keep keep_n children: top by cosine to the target + an unsteered exploration reserve.

        `children` is a list of (program, parent, origin) tuples. No-op if not oversampled.
        """
        if len(children) <= keep_n:
            return children
        embs = self.llm.encode([c[0] for c in children])
        sims = [cosine(self._pred_target_emb, embs[i]) for i in range(len(children))]
        order = sorted(range(len(children)), key=lambda i: sims[i], reverse=True)
        reserve = int(reserve_frac * keep_n)
        top = order[: keep_n - reserve]
        remaining = order[keep_n - reserve:]
        reserve_picks = rng.sample(remaining, min(reserve, len(remaining)))
        return [children[i] for i in top + reserve_picks]


class JepaLLM:
    """Wrapper around the LLM-JEPA fine-tuned model used only for embeddings/prediction.

    Loads a LoRA adapter onto its base (via peft) or a merged checkpoint. Generation goes through
    vLLM elsewhere; this is the HF forward path for hidden states.
    """

    def __init__(self, model_path, base_model=None, pred_token="[PRED]", embed_layer=-1,
                 device="cuda", dtype=None, batch_size=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype is None:
            dtype = torch.bfloat16
        self.embed_layer = embed_layer
        self.batch_size = batch_size
        self.device = device
        self.pred_token = pred_token

        self.tokenizer = AutoTokenizer.from_pretrained(base_model or model_path)
        self.tokenizer.padding_side = "right"
        if base_model:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=dtype, output_hidden_states=True
            )
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, output_hidden_states=True
            )
        if pred_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [pred_token]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(device).eval()

    @torch.no_grad()
    def _embed(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            hidden = self.model(**enc).hidden_states[self.embed_layer]  # [B, T, d]
            last_idx = enc["attention_mask"].sum(dim=1) - 1               # right padding -> last real token
            gather = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            last = hidden.gather(1, gather).squeeze(1)                    # [B, d]
            out.append(last.float().cpu())
        return torch.cat(out, dim=0)

    def encode(self, texts):
        return self._embed(list(texts))

    def predict_target(self, tasks):
        return self._embed([t + self.pred_token for t in tasks])
