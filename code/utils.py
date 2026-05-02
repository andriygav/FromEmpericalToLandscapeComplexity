from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm


def configure_torch_cpu(num_threads: int | None = None) -> None:
    if num_threads is None:
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(max(1, num_threads))


configure_torch_cpu()


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == "code" else cwd


def artifacts_dir() -> Path:
    out = project_root() / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kaplan_flops_estimate(n_params: int, n_tokens: int, flops_per_param_token: float = 6.0) -> float:
    return float(flops_per_param_token * n_params * n_tokens)


def batch_iter_from_tokens(tokens: torch.Tensor, block_size: int, batch_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    max_start = tokens.numel() - block_size - 1
    offsets = torch.arange(block_size, device=tokens.device)
    while True:
        starts = torch.randint(0, max_start, (batch_size,), device=tokens.device)
        idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        x = tokens[idx]
        y = tokens[idx + 1]
        yield x, y


def load_real_token_streams(
    *,
    dataset_name: str,
    dataset_config: str | None,
    tokenizer_name: str,
    train_tokens: int,
    val_tokens: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(dataset_name, dataset_config)
    if "validation" in ds:
        train_texts = list(ds["train"]["text"])
        val_texts = list(ds["validation"]["text"])
    else:
        train_all = list(ds["train"]["text"])
        split = max(1, int(0.95 * len(train_all)))
        train_texts = train_all[:split]
        val_texts = train_all[split:]

    rng = np.random.default_rng(seed)
    rng.shuffle(train_texts)
    rng.shuffle(val_texts)

    def tokenize_until_budget(texts: List[str], budget: int) -> torch.Tensor:
        # Batched tokenization is much faster than per-text encode loops.
        chunks: List[torch.Tensor] = []
        total = 0
        batch_size = 512

        filtered = [t for t in texts if t and t.strip()]
        n_batches = (len(filtered) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(filtered), batch_size), total=n_batches, desc=f"tokenize<{budget}>", leave=False)
        for i in pbar:
            batch = filtered[i : i + batch_size]
            enc = tok(
                batch,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            for ids in enc["input_ids"]:
                if not ids:
                    continue
                t = torch.tensor(ids, dtype=torch.long)
                chunks.append(t)
                total += int(t.numel())
                if total >= budget:
                    pbar.set_postfix(tokens=total)
                    pbar.close()
                    return torch.cat(chunks, dim=0)[:budget].to(device)
            pbar.set_postfix(tokens=total)
        pbar.close()

        if not chunks:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(chunks, dim=0)[:budget].to(device)

    train_ids = tokenize_until_budget(train_texts, train_tokens + 5000)
    val_ids = tokenize_until_budget(val_texts, val_tokens + 5000)
    return train_ids, val_ids, int(tok.vocab_size)


def _model_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss_fn = getattr(model, "loss", None)
    if callable(loss_fn):
        return loss_fn(x, y)
    out: Any = model(input_ids=x, labels=y)
    if hasattr(out, "loss"):
        return out.loss
    raise AttributeError("Model must expose .loss(x, y) or return .loss from forward(input_ids, labels).")


def _hvp_per_example(model: nn.Module, x_i: torch.Tensor, y_i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    # HVP needs second-order gradients; force SDPA math backend to avoid
    # unsupported double-backward paths in efficient CUDA kernels.
    if x_i.is_cuda:
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            loss_i = _model_loss(model, x_i.unsqueeze(0), y_i.unsqueeze(0))
    else:
        loss_i = _model_loss(model, x_i.unsqueeze(0), y_i.unsqueeze(0))
    g = torch.autograd.grad(loss_i, params, create_graph=True)
    g_flat = torch.cat([t.reshape(-1) for t in g])
    gv = torch.dot(g_flat, v)
    hv = torch.autograd.grad(gv, params, retain_graph=False)
    return torch.cat([t.reshape(-1) for t in hv]).detach()


def landscape_complexity_mu(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    rank: int,
    n_examples: int,
) -> float:
    """Estimate per-example Hessian dispersion mu (spectral-norm proxy via random probes)."""
    n_examples = max(2, min(n_examples, x.shape[0]))
    n_probes = max(1, rank)
    x_probe = x[:n_examples]
    y_probe = y[:n_examples]
    params = [p for p in model.parameters() if p.requires_grad]
    numel = sum(p.numel() for p in params)

    probes: List[torch.Tensor] = []
    for _ in range(n_probes):
        v = torch.randn(numel, device=x.device, dtype=torch.float32)
        v = v / (torch.linalg.norm(v) + 1e-12)
        probes.append(v)

    chunk_size = max(1, min(4, n_examples))
    hbar_actions = [torch.zeros(numel, device=x.device) for _ in range(n_probes)]

    seen = 0
    for start in range(0, n_examples, chunk_size):
        end = min(start + chunk_size, n_examples)
        for i in range(start, end):
            for j, v in enumerate(probes):
                hbar_actions[j] += _hvp_per_example(model, x_probe[i], y_probe[i], v)
        seen += end - start
    hbar_actions = [h / max(seen, 1) for h in hbar_actions]

    mu_sum = 0.0
    seen = 0
    for start in range(0, n_examples, chunk_size):
        end = min(start + chunk_size, n_examples)
        for i in range(start, end):
            probe_norms = []
            for j, v in enumerate(probes):
                hi_v = _hvp_per_example(model, x_probe[i], y_probe[i], v)
                probe_norms.append(float(torch.linalg.vector_norm(hi_v - hbar_actions[j], ord=2).item()))
            mu_sum += max(probe_norms) if probe_norms else 0.0
            seen += 1
    return mu_sum / max(seen, 1)


def evaluate_val_loss(model: nn.Module, val_tokens: torch.Tensor, block_size: int, batch_size: int, n_batches: int = 20) -> float:
    model.eval()
    it = batch_iter_from_tokens(val_tokens, block_size, batch_size)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            xb, yb = next(it)
            losses.append(float(_model_loss(model, xb, yb).item()))
    model.train()
    return float(np.mean(losses))
