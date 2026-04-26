from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch

from utils import (
    RunConfig,
    TinyGPT,
    artifacts_dir,
    batch_iter_from_tokens,
    count_parameters,
    evaluate_val_loss,
    kaplan_flops_estimate,
    landscape_complexity_mu,
    load_real_token_streams,
    save_json,
)


RESULT_FIELDS = [
    "run_name",
    "seed",
    "n_layer",
    "n_head",
    "n_embd",
    "n_params",
    "train_tokens_target",
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


def build_grid() -> List[RunConfig]:
    grid: List[RunConfig] = []
    seeds = [0, 1]
    model_grid = [
        {"n_layer": 2, "n_head": 2, "n_embd": 64},
        {"n_layer": 2, "n_head": 2, "n_embd": 80},
        {"n_layer": 2, "n_head": 2, "n_embd": 96},
        {"n_layer": 2, "n_head": 2, "n_embd": 128},
        {"n_layer": 3, "n_head": 4, "n_embd": 136},
        {"n_layer": 3, "n_head": 4, "n_embd": 144},
        {"n_layer": 3, "n_head": 4, "n_embd": 160},
        {"n_layer": 4, "n_head": 4, "n_embd": 192},
        {"n_layer": 5, "n_head": 8, "n_embd": 200},
        {"n_layer": 5, "n_head": 8, "n_embd": 208},
        {"n_layer": 5, "n_head": 8, "n_embd": 224},
        {"n_layer": 6, "n_head": 8, "n_embd": 256},
    ]
    token_grid = [
        20_000, 40_000, 60_000,
        80_000, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 500_000, 600_000, 800_000,
        1_000_000, 1_200_000, 1_600_000,
        2_000_000, 2_500_000, 3_200_000,
    ]
    for seed in seeds:
        for model_cfg in model_grid:
            for train_tokens in token_grid:
                run_name = (
                    f"s{seed}_L{model_cfg['n_layer']}"
                    f"_H{model_cfg['n_head']}_E{model_cfg['n_embd']}_T{train_tokens}"
                )
                grid.append(
                    RunConfig(
                        run_name=run_name,
                        seed=seed,
                        vocab_size=2048,
                        block_size=128,
                        n_layer=model_cfg["n_layer"],
                        n_head=model_cfg["n_head"],
                        n_embd=model_cfg["n_embd"],
                        dropout=0.0,
                        batch_size=16,
                        train_tokens=train_tokens,
                        val_tokens=100_000,
                        lr=3e-4,
                        weight_decay=0.01,
                        max_steps=1200,
                        mu_probe_examples=8,
                        mu_probe_rank=4,
                    )
                )
    return grid


def train_one(
    cfg: RunConfig,
    device: torch.device,
    *,
    dataset_name: str,
    dataset_config: str | None,
    tokenizer_name: str,
    run_idx: int,
    total_runs: int,
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_tokens_stream, val_tokens_stream, vocab_size = load_real_token_streams(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        train_tokens=cfg.train_tokens,
        val_tokens=cfg.val_tokens,
        seed=cfg.seed,
        device=device,
    )
    print(
        f"[{run_idx}/{total_runs}] {cfg.run_name}: "
        f"loaded dataset='{dataset_name}' tokenizer='{tokenizer_name}', "
        f"train_tokens~{train_tokens_stream.numel()}, val_tokens~{val_tokens_stream.numel()}",
        flush=True,
    )

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = batch_iter_from_tokens(train_tokens_stream, cfg.block_size, cfg.batch_size)
    tokens_per_step = cfg.block_size * cfg.batch_size
    max_steps = min(cfg.max_steps, max(1, cfg.train_tokens // tokens_per_step))

    print(f"[{run_idx}/{total_runs}] {cfg.run_name}: training {max_steps} steps on {device}", flush=True)
    model.train()
    for step in range(1, max_steps + 1):
        xb, yb = next(train_iter)
        optim.zero_grad(set_to_none=True)
        loss = model.loss(xb, yb)
        loss.backward()
        optim.step()
        if step == 1 or step % 200 == 0 or step == max_steps:
            print(f"[{run_idx}/{total_runs}] {cfg.run_name}: step {step}/{max_steps}, train_loss={float(loss.item()):.4f}", flush=True)

    val_loss = evaluate_val_loss(model=model, val_tokens=val_tokens_stream, block_size=cfg.block_size, batch_size=cfg.batch_size, n_batches=20)
    ppl = float(np.exp(val_loss))

    xb_mu, yb_mu = next(train_iter)
    print(f"[{run_idx}/{total_runs}] {cfg.run_name}: computing mu...", flush=True)
    mu = landscape_complexity_mu(model=model, x=xb_mu, y=yb_mu, rank=cfg.mu_probe_rank, n_examples=cfg.mu_probe_examples)

    n_params = count_parameters(model)
    trained_tokens = int(max_steps * tokens_per_step)
    flops = kaplan_flops_estimate(n_params=n_params, n_tokens=trained_tokens)

    print(
        f"[{run_idx}/{total_runs}] {cfg.run_name}: done val_loss={val_loss:.4f}, ppl={ppl:.2f}, mu={mu:.6g}, N={n_params}, D={trained_tokens}",
        flush=True,
    )

    return {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "n_params": n_params,
        "train_tokens_target": cfg.train_tokens,
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


def write_results_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RESULT_FIELDS)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def load_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    tokenizer_name = "gpt2"
    resume = True

    out_path = artifacts_dir() / "results_grid.csv"
    existing_rows = load_existing_rows(out_path) if resume else []
    done_run_names = {r.get("run_name", "") for r in existing_rows}

    grid = build_grid()
    pending = [cfg for cfg in grid if cfg.run_name not in done_run_names]

    if existing_rows:
        print(f"Resume mode: found {len(existing_rows)} completed runs in {out_path}, pending {len(pending)} runs.", flush=True)
        skipped = [cfg.run_name for cfg in grid if cfg.run_name in done_run_names]
        print(f"Skipped runs ({len(skipped)}):", flush=True)
        for s in skipped:
            print(f"  - {s}", flush=True)
    else:
        print(f"Output file: {out_path}", flush=True)
        print(f"Starting fresh: pending {len(pending)} runs.", flush=True)

    out_rows = list(existing_rows)
    total_runs = len(pending)
    for i, cfg in enumerate(pending, start=1):
        row = train_one(
            cfg,
            device=torch.device(args.device),
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            tokenizer_name=tokenizer_name,
            run_idx=i,
            total_runs=total_runs,
        )
        out_rows.append(row)
        write_results_csv(out_rows, out_path)
        print(f"Checkpoint saved: {len(out_rows)} runs -> {out_path}", flush=True)

    save_json(
        out_path.with_suffix(".meta.json"),
        {"num_runs_total": len(out_rows), "num_runs_new": len(pending), "resume_enabled": resume, "runs": [asdict(c) for c in grid]},
    )
    print(f"Saved {len(out_rows)} total runs to {out_path}")


if __name__ == "__main__":
    main()
