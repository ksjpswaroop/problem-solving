"""
File Purpose Summary:
This example demonstrates a realistic delivery-route planning workflow using the
ProblemSolving library's search engines. It shows how to let the algorithm selector
choose an engine automatically and compares that result with a speed-first heuristic
strategy so users can understand quality vs. speed tradeoffs.

Responsibilities & Behavior:
- Defines a weighted city graph that models travel times between logistics stops.
- Provides heuristic distances for informed search.
- Uses select_algorithm(...) to explain the recommended engine choice.
- Runs solve(...) with engine="auto" and with engine="greedy".
- Prints route, travel cost, and explored-node metrics for comparison.

Context & Architectural Notes:
- Uses the public solve/selector interfaces only, mirroring agent or app usage.
- Demonstrates a common production pattern: deterministic route solving with
  transparent metadata, instead of opaque path suggestions.
- The graph and heuristic are deliberately simple so behavior is easy to inspect.

Created At (ISO): 2026-02-23T18:27:40Z
Created At (PT): 2026-02-23 10:27:40 PST
Updated At (ISO): 2026-02-23T18:27:40Z
Updated At (PT): 2026-02-23 10:27:40 PST
Updated By: AI
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from problemsolving import solve
    from problemsolving.selector import select_algorithm
except ModuleNotFoundError:
    # Allow direct execution from repository root without package installation.
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_PATH = REPO_ROOT / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))
    from problemsolving import solve
    from problemsolving.selector import select_algorithm


GRAPH = {
    "warehouse": [("north_hub", 6), ("east_hub", 4), ("south_hub", 7)],
    "north_hub": [("customer_a", 2), ("customer_b", 5)],
    "east_hub": [("customer_b", 3), ("customer_c", 6)],
    "south_hub": [("customer_c", 2), ("customer_d", 5)],
    "customer_a": [("destination", 8)],
    "customer_b": [("destination", 4)],
    "customer_c": [("destination", 3)],
    "customer_d": [("destination", 1)],
    "destination": [],
}

HEURISTIC_TO_DESTINATION = {
    "warehouse": 6,
    "north_hub": 5,
    "east_hub": 4,
    "south_hub": 3,
    "customer_a": 8,
    "customer_b": 3,
    "customer_c": 2,
    "customer_d": 1,
    "destination": 0,
}


def run() -> None:
    """Run logistics route planning with auto and greedy strategies."""
    payload = {
        "graph": GRAPH,
        "start": "warehouse",
        "goal": "destination",
        "heuristic": HEURISTIC_TO_DESTINATION,
    }

    selection = select_algorithm(
        problem_type="pathfinding",
        features={"weighted": True, "has_heuristic": True, "needs_optimal": True},
    )
    print("Selector recommendation:", selection)

    auto = solve(
        engine="auto",
        input_data=payload,
        problem_type="pathfinding",
        features={"weighted": True, "has_heuristic": True, "needs_optimal": True},
    )
    greedy = solve(engine="greedy", input_data=payload)

    if auto.status != "success" or auto.result is None:
        raise RuntimeError(f"Auto route planning failed: {auto.error}")
    if greedy.status != "success" or greedy.result is None:
        raise RuntimeError(f"Greedy route planning failed: {greedy.error}")

    print("\n=== Auto-selected strategy ===")
    print("Engine:", auto.engine)
    print("Path:", " -> ".join(auto.result["path"]))
    print("Total travel cost:", auto.result.get("cost", "n/a"))
    print("Nodes explored:", auto.result.get("nodes_explored", "n/a"))

    print("\n=== Greedy strategy ===")
    print("Engine:", greedy.engine)
    print("Path:", " -> ".join(greedy.result["path"]))
    print("Total travel cost:", greedy.result.get("cost", "n/a"))
    print("Nodes explored:", greedy.result.get("nodes_explored", "n/a"))


if __name__ == "__main__":
    run()
