# BOOF: compact reproducibility example

This repository contains a small, self-contained reference implementation of the
Budget-aware Online Offloading and Resource Allocation Framework (BOOF) from:

> *Online LLM inference routing for multi-agent systems under long-term budget
> and layered privacy constraints.*

The code is intentionally compact and uses synthetic workloads. It demonstrates
the control loop and the two per-slot solvers described in the paper:

- **BOOF-BRAO**: queue-weighted best-response updates when routing intents are shared;
- **BOOF-MADO**: local policy updates with exploration when intents are private;
- **No-Lyapunov**: a greedy performance-only reference policy.

## Quick start

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python run_demo.py --output demo_output
```

The command writes `demo_output/summary.json`, `demo_output/budget_trace.csv`,
and `demo_output/budget_queue.png`. Use `--seed`, `--horizon`, and `--agents`
to change the deterministic demo instance.

## Model map

Each time slot samples local, edge, and cloud inference options. The virtual
budget queue is updated as

```text
Q[n] <- max(Q[n] + spending[n] - B_max, 0)
```

BOOF adds `Q[n] * spending[n]` to each agent's per-slot objective. BRAO then
searches unilateral improvements, while MADO samples from and updates a local
probability vector. The implementation is a teaching and reproducibility aid;
it is not a production LLM serving system.

## Repository layout

```text
boof.py          core synthetic environment and BOOF/BRAO/MADO routines
run_demo.py      runnable budget-control example
requirements.txt Python dependencies
LICENSE          MIT license
```

The paper's full experimental workspace contains additional validation scripts
and plots; this compact repository is the public code artifact referenced by the
paper.

## Citation

If you use this code, please cite the paper and link back to this repository.
