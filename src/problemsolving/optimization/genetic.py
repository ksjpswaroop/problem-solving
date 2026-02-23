"""Genetic Algorithm optimization."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any


def ga_solve(
    fitness: Callable[[list[float]], float],
    bounds: list[tuple[float, float]],
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    elite_fraction: float = 0.1,
) -> dict[str, Any]:
    """Genetic Algorithm — maximizes fitness function.

    Returns dict with 'solution', 'fitness', 'generations'.
    """
    n_elite = max(1, int(population_size * elite_fraction))

    # Initialize random population within bounds
    population = [
        [random.uniform(lo, hi) for lo, hi in bounds]
        for _ in range(population_size)
    ]

    best_individual: list[float] | None = None
    best_fitness = float("-inf")

    for gen in range(generations):
        # Evaluate fitness
        scored = [(ind, fitness(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Track best
        if scored[0][1] > best_fitness:
            best_fitness = scored[0][1]
            best_individual = list(scored[0][0])

        # Elitism: keep top individuals
        elites = [list(ind) for ind, _ in scored[:n_elite]]

        # Selection + crossover + mutation for next generation
        new_population = list(elites)

        while len(new_population) < population_size:
            # Tournament selection (size 3)
            parent1 = _tournament_select(scored, 3)
            parent2 = _tournament_select(scored, 3)

            # Crossover
            if random.random() < crossover_rate:
                child = _crossover(parent1, parent2)
            else:
                child = list(parent1)

            # Mutation
            child = _mutate(child, bounds, mutation_rate)

            new_population.append(child)

        population = new_population[:population_size]

    return {
        "solution": best_individual or population[0],
        "fitness": best_fitness,
        "generations": generations,
    }


def _tournament_select(
    scored: list[tuple[list[float], float]], k: int
) -> list[float]:
    """Select individual via tournament selection."""
    contestants = random.sample(scored, min(k, len(scored)))
    winner = max(contestants, key=lambda x: x[1])
    return winner[0]


def _crossover(p1: list[float], p2: list[float]) -> list[float]:
    """Uniform crossover."""
    return [
        p1[i] if random.random() < 0.5 else p2[i]
        for i in range(len(p1))
    ]


def _mutate(
    ind: list[float],
    bounds: list[tuple[float, float]],
    rate: float,
) -> list[float]:
    """Gaussian mutation with bounds clamping."""
    result = []
    for i, (lo, hi) in enumerate(bounds):
        if random.random() < rate:
            spread = (hi - lo) * 0.1
            val = ind[i] + random.gauss(0, spread)
            val = max(lo, min(hi, val))
            result.append(val)
        else:
            result.append(ind[i])
    return result


def ga_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for Genetic Algorithm."""
    bounds = [tuple(b) for b in input_data["bounds"]]
    pop_size = input_data.get("population_size", 50)
    gens = input_data.get("generations", 100)

    # For protocol use, fitness is defined by expression or predefined type
    target = input_data.get("target", [0.0] * len(bounds))

    def fitness(ind: list[float]) -> float:
        return -float(sum((xi - ti) ** 2 for xi, ti in zip(ind, target)))

    return ga_solve(
        fitness=fitness,
        bounds=bounds,
        population_size=pop_size,
        generations=gens,
    )
