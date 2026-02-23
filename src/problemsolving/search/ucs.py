"""Uniform Cost Search algorithm."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def ucs_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[tuple[T, float]]],
) -> dict[str, Any] | None:
    """Uniform Cost Search — finds cheapest-cost path.

    neighbors(node) returns list of (neighbor, edge_cost) tuples.
    Returns dict with 'path', 'cost', 'nodes_explored', or None.
    """
    if goal_test(start):
        return {"path": [start], "cost": 0, "nodes_explored": 1}

    # Priority queue: (cost, tie-breaker, node, path)
    counter = 0
    frontier: list[tuple[float, int, T, list[T]]] = [(0, counter, start, [start])]
    best_cost: dict[Any, float] = {}
    nodes_explored = 0

    while frontier:
        cost, _, node, path = heapq.heappop(frontier)

        key = node if isinstance(node, (str, int, float, tuple)) else id(node)

        if key in best_cost and best_cost[key] <= cost:
            continue

        best_cost[key] = cost
        nodes_explored += 1

        if goal_test(node):
            return {"path": path, "cost": cost, "nodes_explored": nodes_explored}

        for neighbor, edge_cost in neighbors(node):
            new_cost = cost + edge_cost
            nkey = neighbor if isinstance(neighbor, (str, int, float, tuple)) else id(neighbor)
            if nkey not in best_cost or new_cost < best_cost[nkey]:
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, neighbor, path + [neighbor]))

    return None


def ucs_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for UCS."""
    graph = input_data["graph"]
    start = input_data["start"]
    goal = input_data["goal"]

    result = ucs_solve(
        start=start,
        goal_test=lambda n: n == goal,
        neighbors=lambda n: graph.get(n, []),
    )

    if result is None:
        return {"status": "no_path", "path": [], "cost": 0, "nodes_explored": 0}
    return result
