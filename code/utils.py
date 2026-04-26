from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, channels)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = idx.shape
        pos = torch.arange(0, seqlen, device=idx.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    seed: int
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    batch_size: int
    train_tokens: int
    val_tokens: int
    lr: float
    weight_decay: float
    max_steps: int
    mu_probe_examples: int
    mu_probe_rank: int


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kaplan_flops_estimate(n_params: int, n_tokens: int, flops_per_param_token: float = 6.0) -> float:
    return float(flops_per_param_token * n_params * n_tokens)


def batch_iter_from_tokens(tokens: torch.Tensor, block_size: int, batch_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    max_start = tokens.numel() - block_size - 1
    while True:
        starts = torch.randint(0, max_start, (batch_size,), device=tokens.device)
        x = torch.stack([tokens[s : s + block_size] for s in starts], dim=0)
        y = torch.stack([tokens[s + 1 : s + block_size + 1] for s in starts], dim=0)
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
        chunks: List[torch.Tensor] = []
        total = 0
        for txt in texts:
            if not txt or not txt.strip():
                continue
            ids = tok.encode(txt, add_special_tokens=False)
            if not ids:
                continue
            t = torch.tensor(ids, dtype=torch.long)
            chunks.append(t)
            total += int(t.numel())
            if total >= budget:
                break
        return torch.cat(chunks, dim=0)[:budget].to(device)

    train_ids = tokenize_until_budget(train_texts, train_tokens + 5000)
    val_ids = tokenize_until_budget(val_texts, val_tokens + 5000)
    return train_ids, val_ids, int(tok.vocab_size)


def _hvp_per_example(model: TinyGPT, x_i: torch.Tensor, y_i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    loss_i = model.loss(x_i.unsqueeze(0), y_i.unsqueeze(0))
    g = torch.autograd.grad(loss_i, params, create_graph=True)
    g_flat = torch.cat([t.reshape(-1) for t in g])
    gv = torch.dot(g_flat, v)
    hv = torch.autograd.grad(gv, params, retain_graph=False)
    return torch.cat([t.reshape(-1) for t in hv]).detach()


def landscape_complexity_mu(
    model: TinyGPT,
    x: torch.Tensor,
    y: torch.Tensor,
    rank: int,
    n_examples: int,
) -> float:
    n_examples = max(2, min(n_examples, x.shape[0]))
    n_probes = max(1, rank)
    x_probe = x[:n_examples]
    y_probe = y[:n_examples]
    params = [p for p in model.parameters() if p.requires_grad]
    numel = sum(p.numel() for p in params)

    probes: List[torch.Tensor] = []
    for _ in range(n_probes):
        v = torch.randn(numel, device=x.device)
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


def evaluate_val_loss(model: TinyGPT, val_tokens: torch.Tensor, block_size: int, batch_size: int, n_batches: int = 20) -> float:
    model.eval()
    it = batch_iter_from_tokens(val_tokens, block_size, batch_size)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            xb, yb = next(it)
            losses.append(float(model.loss(xb, yb).item()))
    model.train()
    return float(np.mean(losses))
