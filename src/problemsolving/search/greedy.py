"""Greedy Best-First Search algorithm."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def greedy_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[tuple[T, float]]],
    heuristic: Callable[[T], float],
) -> dict[str, Any] | None:
    """Greedy Best-First Search — expands node closest to goal by heuristic.

    Fast but NOT guaranteed optimal.
    neighbors(node) returns list of (neighbor, edge_cost) tuples.
    heuristic(node) returns estimated cost to goal.
    Returns dict with 'path', 'nodes_explored', or None.
    """
    if goal_test(start):
        return {"path": [start], "nodes_explored": 1}

    counter = 0
    # (heuristic_value, tie-breaker, node, path)
    frontier: list[tuple[float, int, T, list[T]]] = [
        (heuristic(start), counter, start, [start])
    ]
    visited: set[Any] = set()
    nodes_explored = 0

    while frontier:
        _h, _, node, path = heapq.heappop(frontier)

        key = node if isinstance(node, (str, int, float, tuple)) else id(node)

        if key in visited:
            continue

        visited.add(key)
        nodes_explored += 1

        if goal_test(node):
            return {"path": path, "nodes_explored": nodes_explored}

        for neighbor, _edge_cost in neighbors(node):
            nkey = neighbor if isinstance(neighbor, (str, int, float, tuple)) else id(neighbor)
            if nkey not in visited:
                counter += 1
                heapq.heappush(
                    frontier, (heuristic(neighbor), counter, neighbor, path + [neighbor])
                )

    return None


def greedy_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for Greedy Best-First Search."""
    graph = input_data["graph"]
    start = input_data["start"]
    goal = input_data["goal"]
    h_values = input_data.get("heuristic", {})

    result = greedy_solve(
        start=start,
        goal_test=lambda n: n == goal,
        neighbors=lambda n: graph.get(n, []),
        heuristic=lambda n: h_values.get(str(n), 0),
    )

    if result is None:
        return {"status": "no_path", "path": [], "nodes_explored": 0}
    return result
