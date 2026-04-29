from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

from utils import (
    artifacts_dir,
    batch_iter_from_tokens,
    count_parameters,
    evaluate_val_loss,
    kaplan_flops_estimate,
    landscape_complexity_mu,
    load_real_token_streams,
)


RESULT_FIELDS = [
    "run_name",
    "seed",
    "model_id",
    "n_layer",
    "n_head",
    "n_embd",
    "batch_size",
    "n_params",
    "checkpoint_idx",
    "checkpoint_tokens_target",
    "train_tokens_real",
    "val_loss",
    "val_ppl",
    "mu_landscape",
    "flops_estimate",
    "d_over_n",
    "dataset_name",
    "dataset_config",
    "tokenizer_name",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    n_layer: int
    n_head: int
    n_embd: int


def selected_three_models() -> List[ModelSpec]:
    # Minimal / middle / maximal representative architectures.
    return [
        ModelSpec("L2_H2_E48", 2, 2, 48),
        ModelSpec("L5_H2_E96", 5, 2, 96),
        ModelSpec("L7_H4_E128", 7, 4, 128),
    ]


def build_full_trajectory_checkpoints(max_train_tokens: int, n_points: int, min_tokens: int) -> List[int]:
    vals = np.geomspace(min_tokens, max_train_tokens, num=max(8, n_points))
    out = sorted({int(round(v / 1_000.0) * 1_000) for v in vals if v > 0})
    if out[-1] != max_train_tokens:
        out.append(int(max_train_tokens))
    return sorted(set(out))


def write_results_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})


def load_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--max-train-tokens", type=int, default=150_000_000)
    parser.add_argument("--min-checkpoint-tokens", type=int, default=20_000)
    parser.add_argument("--checkpoints", type=int, default=36)
    parser.add_argument("--mu-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out-csv", type=str, default="results_few_models.csv")
    args = parser.parse_args()

    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"
    tokenizer_name = "gpt2"
    resume = True

    device = torch.device(args.device)
    use_amp = not args.no_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_specs = selected_three_models()
    seeds = list(range(max(1, args.seeds)))
    checkpoints = build_full_trajectory_checkpoints(
        max_train_tokens=args.max_train_tokens,
        n_points=args.checkpoints,
        min_tokens=args.min_checkpoint_tokens,
    )

    out_path = artifacts_dir() / args.out_csv
    existing_rows = load_existing_rows(out_path) if resume else []
    existing_run_names = {r.get("run_name", "") for r in existing_rows}

    out_rows = list(existing_rows)
    total_runs = len(model_specs) * len(seeds)
    runs_pbar = tqdm(total=total_runs, desc="few-model runs", unit="run")
    run_idx = 0
    for seed in seeds:
        max_val_tokens = 100_000
        train_stream, val_stream, vocab_size = load_real_token_streams(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            tokenizer_name=tokenizer_name,
            train_tokens=args.max_train_tokens,
            val_tokens=max_val_tokens,
            seed=seed,
            device=device,
        )

        for spec in model_specs:
            run_idx += 1
            base_name = f"few_s{seed}_{spec.model_id}_B{args.batch_size}"
            expected_names = [f"{base_name}_T{t}" for t in checkpoints]
            if all(n in existing_run_names for n in expected_names):
                runs_pbar.update(1)
                continue

            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=128,
                n_ctx=128,
                n_layer=spec.n_layer,
                n_head=spec.n_head,
                n_embd=spec.n_embd,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
            model = GPT2LMHeadModel(config).to(device)
            model.loss = lambda x, y: model(input_ids=x, labels=y).loss  # type: ignore[attr-defined]
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            train_iter = batch_iter_from_tokens(train_stream, 128, args.batch_size)
            tokens_per_step = 128 * args.batch_size
            max_steps = max(1, int(np.ceil(args.max_train_tokens / tokens_per_step)))
            warmup_steps = max(1, int(round(args.warmup_frac * max_steps)))
            amp_enabled = use_amp and device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

            def lr_at_step(step_idx: int) -> float:
                # Kaplan-style schedule in practice: short warmup then smooth decay.
                if step_idx <= warmup_steps:
                    return args.lr * (step_idx / warmup_steps)
                progress = (step_idx - warmup_steps) / max(1, max_steps - warmup_steps)
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                return args.min_lr + (args.lr - args.min_lr) * cosine

            checkpoint_ptr = 0
            model.train()
            steps_pbar = tqdm(total=max_steps, desc=base_name, leave=False, unit="step")
            last_ckpt_msg = "-"
            for step in range(1, max_steps + 1):
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

                trained_tokens = int(step * tokens_per_step)
                while checkpoint_ptr < len(checkpoints) and trained_tokens >= checkpoints[checkpoint_ptr]:
                    ckpt_tokens = checkpoints[checkpoint_ptr]
                    run_name = f"{base_name}_T{ckpt_tokens}"
                    checkpoint_idx = checkpoint_ptr + 1
                    checkpoint_ptr += 1
                    if run_name in existing_run_names:
                        continue

                    val_loss = evaluate_val_loss(
                        model=model,
                        val_tokens=val_stream,
                        block_size=128,
                        batch_size=args.batch_size,
                        n_batches=20,
                    )
                    ppl = float(np.exp(val_loss))
                    mu = np.nan
                    if (checkpoint_idx - 1) % max(1, args.mu_every) == 0:
                        xb_mu, yb_mu = next(train_iter)
                        mu = landscape_complexity_mu(model=model, x=xb_mu, y=yb_mu, rank=4, n_examples=8)

                    n_params = count_parameters(model)
                    flops = kaplan_flops_estimate(n_params=n_params, n_tokens=trained_tokens)
                    row = {
                        "run_name": run_name,
                        "seed": seed,
                        "model_id": spec.model_id,
                        "n_layer": spec.n_layer,
                        "n_head": spec.n_head,
                        "n_embd": spec.n_embd,
                        "batch_size": args.batch_size,
                        "n_params": n_params,
                        "checkpoint_idx": checkpoint_idx,
                        "checkpoint_tokens_target": ckpt_tokens,
                        "train_tokens_real": trained_tokens,
                        "val_loss": val_loss,
                        "val_ppl": ppl,
                        "mu_landscape": mu,
                        "flops_estimate": flops,
                        "d_over_n": float(trained_tokens / max(n_params, 1)),
                        "dataset_name": dataset_name,
                        "dataset_config": dataset_config,
                        "tokenizer_name": tokenizer_name,
                    }
                    out_rows.append(row)
                    existing_run_names.add(run_name)
                    write_results_csv(out_rows, out_path)
                    last_ckpt_msg = f"T{ckpt_tokens} L={val_loss:.3f} mu={mu if np.isfinite(mu) else float('nan'):.3g}"
                steps_pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{lr_now:.2e}", ckpt=last_ckpt_msg)
                steps_pbar.update(1)
            steps_pbar.close()
            runs_pbar.update(1)

        del train_stream, val_stream
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Saved {len(out_rows)} rows to {out_path}", flush=True)
    runs_pbar.close()


if __name__ == "__main__":
    main()
