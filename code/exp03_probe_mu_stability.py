from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

from utils import (
    artifacts_dir,
    batch_iter_from_tokens,
    count_parameters,
    landscape_complexity_mu,
    load_real_token_streams,
)


def exp03_dir() -> Path:
    d = artifacts_dir() / "exp03"
    d.mkdir(parents=True, exist_ok=True)
    return d


def attach_loss(model: GPT2LMHeadModel) -> None:
    model.loss = lambda x, y: model(input_ids=x, labels=y).loss  # type: ignore[attr-defined]


def train_small_model(
    *,
    device: torch.device,
    train_tokens: int,
    batch_size: int,
    seed: int,
    lr: float,
    min_lr: float,
    warmup_frac: float,
    use_amp: bool,
) -> Tuple[GPT2LMHeadModel, torch.Tensor, str, str, str]:
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"
    tokenizer_name = "gpt2"

    train_stream, val_stream, vocab_size = load_real_token_streams(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        train_tokens=train_tokens + 5000,
        val_tokens=100_000,
        seed=seed,
        device=device,
    )

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_ctx=128,
        n_layer=2,
        n_head=2,
        n_embd=48,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config).to(device)
    attach_loss(model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_iter = batch_iter_from_tokens(train_stream, 128, batch_size)
    tokens_per_step = 128 * batch_size
    max_steps = max(1, int(np.ceil(train_tokens / tokens_per_step)))
    warmup_steps = max(1, int(round(warmup_frac * max_steps)))
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def lr_at_step(step_idx: int) -> float:
        if step_idx <= warmup_steps:
            return lr * (step_idx / warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr + (lr - min_lr) * cosine

    model.train()
    for step in tqdm(range(1, max_steps + 1), desc="exp03 train L2_H2_E48", unit="step"):
        lr_now = float(lr_at_step(step))
        for pg in optim.param_groups:
            pg["lr"] = lr_now
        xb, yb = next(train_iter)
        optim.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model.loss(xb, yb)
        else:
            loss = model.loss(xb, yb)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

    return model, train_stream, dataset_name, dataset_config, tokenizer_name


def mean_loss_grad_norm(model: GPT2LMHeadModel, x: torch.Tensor, y: torch.Tensor, n_examples: int) -> float:
    model.train()
    model.zero_grad(set_to_none=True)
    n_examples = max(1, min(n_examples, x.shape[0]))
    losses = []
    for i in range(n_examples):
        losses.append(model.loss(x[i : i + 1], y[i : i + 1]))
    lm = torch.stack(losses).mean()
    lm.backward()
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum().cpu())
    model.zero_grad(set_to_none=True)
    return float(math.sqrt(sq))


def fixed_probe_batch(
    train_stream: torch.Tensor,
    batch_size: int,
    batch_seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(batch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(batch_seed)
    it = batch_iter_from_tokens(train_stream, 128, batch_size)
    xb, yb = next(it)
    return xb, yb


def summarize(values: List[float]) -> Dict[str, float]:
    a = np.array(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "cv": float(a.std(ddof=1) / (abs(a.mean()) + 1e-12)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": float(len(a)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--probe-trials", type=int, default=40)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--fixed-batch-seed", type=int, default=12345)
    parser.add_argument("--batch-noise-base-seed", type=int, default=9000)
    parser.add_argument("--direction-noise-base-seed", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = not args.no_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, train_stream, dataset_name, dataset_config, tokenizer_name = train_small_model(
        device=device,
        train_tokens=args.train_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_frac=args.warmup_frac,
        use_amp=use_amp,
    )

    n_ex = args.n_examples
    rank = args.rank
    K = max(2, args.probe_trials)

    x_fixed, y_fixed = fixed_probe_batch(
        train_stream, args.batch_size, args.fixed_batch_seed, device
    )
    grad_norm_fixed = mean_loss_grad_norm(model, x_fixed, y_fixed, n_ex)

    rows: List[Dict[str, Any]] = []

    mus_dir: List[float] = []
    for t in range(K):
        ps = args.direction_noise_base_seed + t
        mu = landscape_complexity_mu(model, x_fixed, y_fixed, rank=rank, n_examples=n_ex, probe_seed=ps)
        mus_dir.append(mu)
        rows.append(
            {
                "mode": "direction_noise",
                "trial": t,
                "probe_seed": ps,
                "batch_seed": args.fixed_batch_seed,
                "mu": mu,
                "grad_norm_fixed_batch": grad_norm_fixed,
            }
        )

    mus_batch: List[float] = []
    for t in range(K):
        bs = args.batch_noise_base_seed + t
        xb, yb = fixed_probe_batch(train_stream, args.batch_size, bs, device)
        mu = landscape_complexity_mu(
            model, xb, yb, rank=rank, n_examples=n_ex, probe_seed=args.direction_noise_base_seed
        )
        mus_batch.append(mu)
        rows.append(
            {
                "mode": "batch_noise",
                "trial": t,
                "probe_seed": args.direction_noise_base_seed,
                "batch_seed": bs,
                "mu": mu,
                "grad_norm_fixed_batch": grad_norm_fixed,
            }
        )

    mus_joint: List[float] = []
    for t in range(K):
        bs = args.batch_noise_base_seed + 10_000 + t
        ps = args.direction_noise_base_seed + 10_000 + t
        xb, yb = fixed_probe_batch(train_stream, args.batch_size, bs, device)
        mu = landscape_complexity_mu(model, xb, yb, rank=rank, n_examples=n_ex, probe_seed=ps)
        mus_joint.append(mu)
        rows.append(
            {
                "mode": "joint",
                "trial": t,
                "probe_seed": ps,
                "batch_seed": bs,
                "mu": mu,
                "grad_norm_fixed_batch": grad_norm_fixed,
            }
        )

    out_csv = exp03_dir() / "exp03_probe_trials.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "model_id": "L2_H2_E48",
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 48,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "tokenizer_name": tokenizer_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_params": count_parameters(model),
        "train_tokens": args.train_tokens,
        "probe_trials": K,
        "n_examples": n_ex,
        "rank": rank,
        "fixed_batch_seed": args.fixed_batch_seed,
        "grad_norm_mean_loss_fixed_batch": grad_norm_fixed,
        "mu_direction_noise": summarize(mus_dir),
        "mu_batch_noise": summarize(mus_batch),
        "mu_joint": summarize(mus_joint),
    }
    out_json = exp03_dir() / "exp03_probe_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
