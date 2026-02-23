"""
File Purpose Summary:
This example shows a practical scheduling workflow that combines CSP assignment and
SAT policy validation. It models a small hospital staffing scenario where shifts are
assigned with constraints, then boolean governance rules are checked independently.

Responsibilities & Behavior:
- Solves a shift assignment problem using the csp_backtracking engine.
- Verifies boolean policy clauses using the dpll_sat engine.
- Prints concrete outputs for both phases so users can adapt the pattern.

Context & Architectural Notes:
- Demonstrates how CSP and SAT can be composed in production workflows:
  CSP handles multi-valued assignment decisions, while SAT handles binary policy checks.
- Uses the unified solve(...) API throughout to keep orchestration simple.
- Designed to be deterministic and easy to read for onboarding and demos.

Created At (ISO): 2026-02-23T18:27:40Z
Created At (PT): 2026-02-23 10:27:40 PST
Updated At (ISO): 2026-02-23T18:27:40Z
Updated At (PT): 2026-02-23 10:27:40 PST
Updated By: AI
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution from repository root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from problemsolving import solve


def run() -> None:
    """Run CSP scheduling and SAT policy checks."""
    csp_response = solve(
        engine="csp_backtracking",
        input_data={
            "variables": ["alice_monday", "bob_monday", "alice_tuesday", "bob_tuesday"],
            "domains": {
                "alice_monday": ["day", "night"],
                "bob_monday": ["day", "night"],
                "alice_tuesday": ["day", "night"],
                "bob_tuesday": ["day", "night"],
            },
            "not_equal_constraints": [
                ["alice_monday", "bob_monday"],
                ["alice_tuesday", "bob_tuesday"],
            ],
        },
    )

    if csp_response.status != "success" or csp_response.result is None:
        raise RuntimeError(f"CSP scheduling failed: {csp_response.error}")

    sat_response = solve(
        engine="dpll_sat",
        input_data={
            # x1 = Alice weekend on-call, x2 = Bob weekend on-call
            # (x1 OR x2) AND (NOT x1 OR NOT x2) means exactly one is on-call.
            "clauses": [[1, 2], [-1, -2]],
            "num_vars": 2,
        },
    )

    if sat_response.status != "success" or sat_response.result is None:
        raise RuntimeError(f"SAT policy validation failed: {sat_response.error}")

    print("=== CSP Shift Assignment ===")
    print("Satisfiable:", csp_response.result["satisfiable"])
    print("Assignment:", csp_response.result["assignment"])
    print("Nodes explored:", csp_response.result["nodes_explored"])

    print("\n=== SAT Policy Validation ===")
    print("Satisfiable:", sat_response.result["satisfiable"])
    print("Assignment:", sat_response.result["assignment"])


if __name__ == "__main__":
    run()
