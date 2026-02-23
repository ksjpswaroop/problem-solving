"""Breadth-First Search (BFS) algorithm.

Finds the shortest path (fewest hops) in an unweighted graph.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def bfs_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[T]],
) -> dict[str, Any] | None:
    """Run BFS from start until goal_test is satisfied.

    Args:
        start: Initial state.
        goal_test: Returns True if a state is the goal.
        neighbors: Returns adjacent states for a given state.

    Returns:
        Dict with 'path' and 'nodes_explored', or None if no path.
    """
    if goal_test(start):
        return {"path": [start], "nodes_explored": 1}

    queue: deque[T] = deque([start])
    visited: set[Any] = {start}
    parent: dict[Any, Any] = {}
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        for neighbor in neighbors(current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = current

            if goal_test(neighbor):
                # Reconstruct path
                path = [neighbor]
                node = neighbor
                while node in parent:
                    node = parent[node]
                    path.append(node)
                path.reverse()
                return {
                    "path": path,
                    "nodes_explored": nodes_explored + 1,
                }

            queue.append(neighbor)

    return None


def bfs_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """BFS solver for the protocol layer — accepts dict input.

    Expected input_data keys:
        start: The starting node.
        goal: The goal node.
        graph: Dict mapping node -> list of neighbor nodes.
    """
    start = input_data["start"]
    goal = input_data["goal"]
    graph: dict[str, list[str]] = input_data["graph"]

    result = bfs_solve(
        start=start,
        goal_test=lambda n: n == goal,
        neighbors=lambda n: graph.get(n, []),
    )

    if result is None:
        raise ValueError(f"No path found from {start} to {goal}")

    return result
