#!/usr/bin/env python3
"""Run a compact BOOF budget-control demonstration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boof import brao, costs, greedy_performance, mado_step, sample_slot, uniform_policy


def run(method: str, seed: int, horizon: int, agents: int, budget: float) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    queues = np.zeros(agents)
    policy = None
    previous = None
    rows = []
    for t in range(horizon):
        slot = sample_slot(rng, agents)
        if method == "BOOF-BRAO":
            actions = brao(slot, queues, warm_start=previous)
        elif method == "BOOF-MADO":
            if policy is None:
                policy = uniform_policy(slot)
            actions, policy = mado_step(slot, queues, policy, rng, t)
        else:
            actions = greedy_performance(slot)
        previous = actions
        spend = slot.spending[np.arange(agents), actions]
        queues = np.maximum(queues + spend - budget, 0.0)
        rows.append({
            "t": t,
            "method": method,
            "mean_spend": float(spend.mean()),
            "max_queue": float(queues.max()),
            "mean_objective": float(costs(slot, actions, queues).mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in ("BOOF-BRAO", "BOOF-MADO", "No-Lyapunov"):
        rows.extend(run(method, args.seed, args.horizon, args.agents, args.budget))
    with (args.output / "budget_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in ("BOOF-BRAO", "BOOF-MADO", "No-Lyapunov"):
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {
            "mean_spend": float(np.mean([r["mean_spend"] for r in subset])),
            "final_max_queue": float(subset[-1]["max_queue"]),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for method, color in (("BOOF-BRAO", "#0072B2"), ("BOOF-MADO", "#D55E00"), ("No-Lyapunov", "#777777")):
        subset = [r for r in rows if r["method"] == method]
        ax.plot([r["t"] for r in subset], [r["max_queue"] for r in subset], label=method, color=color)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Maximum virtual budget queue")
    ax.set_title("Compact BOOF budget-control demo")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "budget_queue.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
