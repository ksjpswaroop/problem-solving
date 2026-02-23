"""Gradient Descent optimization algorithm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def gd_solve(
    objective: Callable[[list[float]], float],
    gradient: Callable[[list[float]], list[float]],
    initial: list[float],
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Gradient Descent minimization.

    Returns dict with 'solution', 'objective_value', 'iterations'.
    """
    x = list(initial)

    for i in range(max_iterations):
        grad = gradient(x)
        grad_norm = sum(g ** 2 for g in grad) ** 0.5

        if grad_norm < tolerance:
            return {
                "solution": x,
                "objective_value": objective(x),
                "iterations": i + 1,
            }

        x = [xi - learning_rate * gi for xi, gi in zip(x, grad)]

    return {
        "solution": x,
        "objective_value": objective(x),
        "iterations": max_iterations,
    }


def gd_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for Gradient Descent."""
    # For protocol use, objective/gradient would need to be provided
    # as expressions or via a predefined function registry.
    # This adapter handles the simple case where coefficients define a quadratic.
    coeffs = input_data.get("coefficients", [])
    offsets = input_data.get("offsets", [])
    initial = input_data.get("initial", [0.0] * len(coeffs))
    lr = input_data.get("learning_rate", 0.01)
    max_iter = input_data.get("max_iterations", 1000)

    def objective(x: list[float]) -> float:
        return float(sum(c * (xi - o) ** 2 for c, xi, o in zip(coeffs, x, offsets)))

    def gradient(x: list[float]) -> list[float]:
        return [2 * c * (xi - o) for c, xi, o in zip(coeffs, x, offsets)]

    return gd_solve(
        objective=objective,
        gradient=gradient,
        initial=initial,
        learning_rate=lr,
        max_iterations=max_iter,
    )
