"""A* Search algorithm."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def astar_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[tuple[T, float]]],
    heuristic: Callable[[T], float],
) -> dict[str, Any] | None:
    """A* Search — optimal with admissible heuristic.

    neighbors(node) returns list of (neighbor, edge_cost) tuples.
    heuristic(node) returns estimated cost to goal (must be admissible for optimality).
    Returns dict with 'path', 'cost', 'nodes_explored', or None.
    """
    if goal_test(start):
        return {"path": [start], "cost": 0, "nodes_explored": 1}

    counter = 0
    # (f_score, tie-breaker, g_cost, node, path)
    frontier: list[tuple[float, int, float, T, list[T]]] = [
        (heuristic(start), counter, 0, start, [start])
    ]
    best_cost: dict[Any, float] = {}
    nodes_explored = 0

    while frontier:
        _f, _, g_cost, node, path = heapq.heappop(frontier)

        key = node if isinstance(node, (str, int, float, tuple)) else id(node)

        if key in best_cost and best_cost[key] <= g_cost:
            continue

        best_cost[key] = g_cost
        nodes_explored += 1

        if goal_test(node):
            return {"path": path, "cost": g_cost, "nodes_explored": nodes_explored}

        for neighbor, edge_cost in neighbors(node):
            new_g = g_cost + edge_cost
            nkey = neighbor if isinstance(neighbor, (str, int, float, tuple)) else id(neighbor)
            if nkey not in best_cost or new_g < best_cost[nkey]:
                f_score = new_g + heuristic(neighbor)
                counter += 1
                heapq.heappush(frontier, (f_score, counter, new_g, neighbor, path + [neighbor]))

    return None


def astar_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for A*."""
    graph = input_data["graph"]
    start = input_data["start"]
    goal = input_data["goal"]
    h_values = input_data.get("heuristic", {})

    result = astar_solve(
        start=start,
        goal_test=lambda n: n == goal,
        neighbors=lambda n: graph.get(n, []),
        heuristic=lambda n: h_values.get(str(n), 0),
    )

    if result is None:
        return {"status": "no_path", "path": [], "cost": 0, "nodes_explored": 0}
    return result
