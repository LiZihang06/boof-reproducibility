"""Compact synthetic BOOF/BRAO/MADO reference implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Slot:
    """Per-slot options for N agents: local, edge servers, and cloud."""

    performance: np.ndarray       # shape (N, A), lower is better
    spending: np.ndarray          # shape (N, A)
    congestion: np.ndarray        # shape (N, E), added to edge actions
    feasible: np.ndarray          # shape (N, A), boolean

    @property
    def agents(self) -> int:
        return self.performance.shape[0]

    @property
    def actions(self) -> int:
        return self.performance.shape[1]

    @property
    def edge_count(self) -> int:
        return self.actions - 2


def sample_slot(rng: np.random.Generator, agents: int, edge_count: int = 2) -> Slot:
    """Sample a small synthetic CEE routing instance."""
    local = rng.uniform(1.15, 1.70, (agents, 1))
    edge = rng.uniform(0.58, 0.98, (agents, edge_count))
    cloud = rng.uniform(0.50, 0.88, (agents, 1))
    performance = np.concatenate((local, edge, cloud), axis=1)
    spending = np.column_stack((
        np.zeros(agents),
        rng.uniform(0.85, 1.35, (agents, edge_count)),
        rng.uniform(1.45, 2.05, agents),
    ))
    congestion = rng.uniform(0.035, 0.080, (agents, edge_count))
    feasible = np.ones_like(performance, dtype=bool)
    for n in range(agents):
        draw = rng.random()
        if draw < 0.08:
            feasible[n, 1:] = False       # local-only privacy constraint
        elif draw < 0.26:
            feasible[n, 0] = False        # local execution unavailable
    return Slot(performance, spending, congestion, feasible)


def costs(slot: Slot, actions: np.ndarray, queues: np.ndarray | None = None,
          control_v: float = 5.0) -> np.ndarray:
    """Evaluate the BOOF per-agent objective for a joint action."""
    value = slot.performance[np.arange(slot.agents), actions].copy() * control_v
    if queues is not None:
        value += queues * slot.spending[np.arange(slot.agents), actions]
    for edge in range(slot.edge_count):
        users = actions == edge + 1
        value[users] += slot.congestion[users, edge] * max(int(users.sum()) - 1, 0) * control_v
    return value


def greedy_performance(slot: Slot) -> np.ndarray:
    """Choose the cheapest performance option for each agent independently."""
    actions = np.zeros(slot.agents, dtype=int)
    for n in range(slot.agents):
        allowed = np.flatnonzero(slot.feasible[n])
        actions[n] = allowed[np.argmin(slot.performance[n, allowed])]
    return actions


def brao(slot: Slot, queues: np.ndarray, control_v: float = 5.0,
         rounds: int = 40, warm_start: np.ndarray | None = None) -> np.ndarray:
    """Sequential best-response updates for shared routing intents."""
    actions = greedy_performance(slot) if warm_start is None else warm_start.copy()
    actions = np.asarray(actions, dtype=int)
    for _ in range(rounds):
        changed = False
        for n in range(slot.agents):
            current = costs(slot, actions, queues, control_v)[n]
            best_action, best_value = actions[n], current
            for action in np.flatnonzero(slot.feasible[n]):
                trial = actions.copy()
                trial[n] = int(action)
                value = costs(slot, trial, queues, control_v)[n]
                if value < best_value - 1e-12:
                    best_action, best_value = int(action), float(value)
            if best_action != actions[n]:
                actions[n] = best_action
                changed = True
        if not changed:
            break
    return actions


def mado_step(slot: Slot, queues: np.ndarray, policy: np.ndarray,
              rng: np.random.Generator, t: int, control_v: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """One local policy update with decaying exploration."""
    epsilon = 0.22 / math.sqrt(t + 1.0)
    actions = np.empty(slot.agents, dtype=int)
    for n in range(slot.agents):
        allowed = np.flatnonzero(slot.feasible[n])
        p = (1.0 - epsilon) * policy[n, allowed] + epsilon / len(allowed)
        actions[n] = int(rng.choice(allowed, p=p / p.sum()))
    observed = costs(slot, actions, queues, control_v)
    updated = policy.copy()
    for n, action in enumerate(actions):
        allowed = np.flatnonzero(slot.feasible[n])
        reward = math.exp(-2.0 * np.clip(observed[n] / (control_v * 2.0), 0.0, 1.0))
        step = 0.12 * reward
        updated[n] *= 1.0 - step
        updated[n, action] += step
        updated[n, ~slot.feasible[n]] = 0.0
        updated[n] /= updated[n].sum()
    return actions, updated


def uniform_policy(slot: Slot) -> np.ndarray:
    policy = np.zeros_like(slot.performance)
    for n in range(slot.agents):
        allowed = np.flatnonzero(slot.feasible[n])
        policy[n, allowed] = 1.0 / len(allowed)
    return policy
