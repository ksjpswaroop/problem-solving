"""Simulated Annealing optimization algorithm."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any


def sa_solve(
    objective: Callable[[list[float]], float],
    initial: list[float],
    neighbor_fn: Callable[[list[float]], list[float]],
    temperature: float = 100.0,
    cooling_rate: float = 0.995,
    max_iterations: int = 10000,
    min_temperature: float = 1e-10,
) -> dict[str, Any]:
    """Simulated Annealing — minimizes objective function.

    Can escape local minima via probabilistic acceptance of worse solutions.
    Returns dict with 'solution', 'objective_value', 'iterations'.
    """
    current = list(initial)
    current_cost = objective(current)

    best = list(current)
    best_cost = current_cost

    temp = temperature

    for i in range(max_iterations):
        if temp < min_temperature:
            break

        candidate = neighbor_fn(current)
        candidate_cost = objective(candidate)

        delta = candidate_cost - current_cost

        # Accept if better, or probabilistically if worse
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best = list(current)
                best_cost = current_cost

        temp *= cooling_rate

    return {
        "solution": best,
        "objective_value": best_cost,
        "iterations": i + 1,  # noqa: B023
    }


def sa_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for Simulated Annealing."""
    initial = input_data.get("initial", [0.0])
    temp = input_data.get("temperature", 100.0)
    cooling = input_data.get("cooling_rate", 0.995)
    max_iter = input_data.get("max_iterations", 10000)

    # For protocol use, objective defined by target
    target = input_data.get("target", [0.0] * len(initial))
    step_size = input_data.get("step_size", 0.5)

    def objective(x: list[float]) -> float:
        return float(sum((xi - ti) ** 2 for xi, ti in zip(x, target)))

    def neighbor_fn(x: list[float]) -> list[float]:
        return [xi + random.gauss(0, step_size) for xi in x]

    return sa_solve(
        objective=objective,
        initial=initial,
        neighbor_fn=neighbor_fn,
        temperature=temp,
        cooling_rate=cooling,
        max_iterations=max_iter,
    )
