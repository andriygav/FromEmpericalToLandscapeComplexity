from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

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


def build_model_grid() -> List[ModelSpec]:
    depths = [2, 3, 4, 5, 6, 7]
    widths = [48, 64, 80, 96, 112, 128]
    specs: List[ModelSpec] = []
    for d in depths:
        for w in widths:
            h = 2 if w <= 96 else 4
            assert w % h == 0
            specs.append(ModelSpec(model_id=f"L{d}_H{h}_E{w}", n_layer=d, n_head=h, n_embd=w))
    return specs


def build_checkpoint_tokens(max_train_tokens: int, n_points: int) -> List[int]:
    vals = np.geomspace(20_000, max_train_tokens, num=n_points)
    out = sorted({int(round(v / 1_000.0) * 1_000) for v in vals})
    return [v for v in out if v > 0]


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
    parser.add_argument("--max-train-tokens", type=int, default=12_000_000)
    parser.add_argument("--checkpoints", type=int, default=20)
    parser.add_argument("--mu-every", type=int, default=2, help="Compute mu every K checkpoints.")
    parser.add_argument("--no-amp", action="store_true")
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

    model_specs = build_model_grid()
    seeds = list(range(max(1, args.seeds)))
    checkpoints = build_checkpoint_tokens(max_train_tokens=args.max_train_tokens, n_points=args.checkpoints)
    print(
        f"Planned token-sliced run: models={len(model_specs)}, seeds={len(seeds)}, "
        f"max_train_tokens={args.max_train_tokens}, checkpoints/model={len(checkpoints)}, "
        f"total_rows={len(model_specs) * len(seeds) * len(checkpoints)}",
        flush=True,
    )

    out_path = artifacts_dir() / "results_grid.csv"
    existing_rows = load_existing_rows(out_path) if resume else []
    existing_run_names = {r.get("run_name", "") for r in existing_rows}

    max_val_tokens = 100_000
    stream_cache = {}
    for seed in seeds:
        train_stream, val_stream, vocab_size = load_real_token_streams(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            tokenizer_name=tokenizer_name,
            train_tokens=args.max_train_tokens,
            val_tokens=max_val_tokens,
            seed=seed,
            device=device,
        )
        stream_cache[seed] = (train_stream, val_stream, vocab_size)
        print(
            f"Prepared streams seed={seed}: train_tokens~{train_stream.numel()}, val_tokens~{val_stream.numel()}",
            flush=True,
        )

    out_rows = list(existing_rows)
    total_runs = len(model_specs) * len(seeds)
    run_idx = 0
    for seed in seeds:
        for spec in model_specs:
            run_idx += 1
            base_name = f"s{seed}_{spec.model_id}_B{args.batch_size}"
            expected_names = [f"{base_name}_T{t}" for t in checkpoints]
            if all(n in existing_run_names for n in expected_names):
                print(f"[{run_idx}/{total_runs}] {base_name}: all checkpoints already present, skip", flush=True)
                continue

            train_tokens_stream, val_tokens_stream, vocab_size = stream_cache[seed]
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

            # Keep compatibility with existing utilities expecting model.loss(x, y).
            model.loss = lambda x, y: model(input_ids=x, labels=y).loss  # type: ignore[attr-defined]
            optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            train_iter = batch_iter_from_tokens(train_tokens_stream, 128, args.batch_size)
            tokens_per_step = 128 * args.batch_size
            max_steps = max(1, int(np.ceil(args.max_train_tokens / tokens_per_step)))
            amp_enabled = use_amp and device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

            checkpoint_ptr = 0
            print(f"[{run_idx}/{total_runs}] {base_name}: training {max_steps} steps", flush=True)
            model.train()
            for step in range(1, max_steps + 1):
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
                        val_tokens=val_tokens_stream,
                        block_size=128,
                        batch_size=args.batch_size,
                        n_batches=20,
                    )
                    ppl = float(np.exp(val_loss))
                    mu = np.nan
                    if (checkpoint_idx - 1) % max(1, args.mu_every) == 0:
                        xb_mu, yb_mu = next(train_iter)
                        mu = landscape_complexity_mu(
                            model=model,
                            x=xb_mu,
                            y=yb_mu,
                            rank=4,
                            n_examples=8,
                        )
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
                    print(
                        f"[{run_idx}/{total_runs}] {run_name}: val_loss={val_loss:.4f}, "
                        f"mu={mu if np.isfinite(mu) else float('nan'):.6g}",
                        flush=True,
                    )

                if step == 1 or step % 300 == 0 or step == max_steps:
                    print(
                        f"[{run_idx}/{total_runs}] {base_name}: step {step}/{max_steps}, train_loss={float(loss.item()):.4f}",
                        flush=True,
                    )

    print(f"Saved {len(out_rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
