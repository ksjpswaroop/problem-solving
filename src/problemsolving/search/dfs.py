"""Depth-First Search algorithm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def dfs_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[T]],
    max_depth: int | None = None,
) -> dict[str, Any] | None:
    """DFS with optional depth limit.

    Returns dict with 'path' and 'nodes_explored', or None if no path found.
    """
    if goal_test(start):
        return {"path": [start], "nodes_explored": 1}

    visited: set[Any] = set()
    # Stack holds (node, path, depth)
    stack: list[tuple[T, list[T], int]] = [(start, [start], 0)]
    nodes_explored = 0

    while stack:
        node, path, depth = stack.pop()

        hashable_node = node if isinstance(node, (str, int, float, tuple)) else id(node)
        if hashable_node in visited:
            continue
        visited.add(hashable_node)

        nodes_explored += 1

        if max_depth is not None and depth >= max_depth:
            continue

        for neighbor in reversed(neighbors(node)):
            hashable = neighbor if isinstance(neighbor, (str, int, float, tuple)) else id(neighbor)
            if hashable not in visited:
                new_path = path + [neighbor]
                if goal_test(neighbor):
                    return {"path": new_path, "nodes_explored": nodes_explored + 1}
                stack.append((neighbor, new_path, depth + 1))

    return None


def dfs_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for DFS."""
    graph = input_data["graph"]
    start = input_data["start"]
    goal = input_data["goal"]
    max_depth = input_data.get("max_depth")

    result = dfs_solve(
        start=start,
        goal_test=lambda n: n == goal,
        neighbors=lambda n: graph.get(n, []),
        max_depth=max_depth,
    )

    if result is None:
        return {"status": "no_path", "path": [], "nodes_explored": 0}
    return result
