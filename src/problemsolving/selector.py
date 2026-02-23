"""Algorithm selector: picks the best engine for a given problem type."""

from __future__ import annotations

from typing import Any

# Decision rules: (problem_type, feature_conditions) -> algorithm
_SELECTION_RULES: list[tuple[str, dict[str, Any], str, str]] = [
    # Pathfinding
    (
        "pathfinding",
        {"weighted": True, "has_heuristic": True, "needs_optimal": True},
        "astar",
        "Weighted graph with heuristic available — A* gives optimal path",
    ),
    (
        "pathfinding",
        {"weighted": True, "has_heuristic": True, "needs_optimal": False},
        "greedy",
        "Weighted graph with heuristic, optimality not required — Greedy is fastest",
    ),
    (
        "pathfinding",
        {"weighted": True, "has_heuristic": False, "needs_optimal": True},
        "ucs",
        "Weighted graph without heuristic — UCS guarantees optimal cost",
    ),
    (
        "pathfinding",
        {"weighted": True, "has_heuristic": False},
        "ucs",
        "Weighted graph — UCS handles varying costs",
    ),
    (
        "pathfinding",
        {"weighted": False},
        "bfs",
        "Unweighted graph — BFS finds shortest path by hops",
    ),
    (
        "pathfinding",
        {},
        "bfs",
        "Default pathfinding — BFS is simplest and guarantees shortest hops",
    ),
    # Satisfiability
    (
        "satisfiability",
        {"domain": "boolean"},
        "dpll_sat",
        "Boolean satisfiability — DPLL with unit propagation",
    ),
    (
        "satisfiability",
        {},
        "dpll_sat",
        "Default satisfiability — DPLL SAT solver",
    ),
    # Constraint satisfaction
    (
        "constraint_satisfaction",
        {},
        "csp_backtracking",
        "CSP — backtracking with MRV heuristic",
    ),
    # Optimization
    (
        "optimization",
        {"differentiable": True},
        "gradient_descent",
        "Differentiable objective — gradient descent converges fastest",
    ),
    (
        "optimization",
        {"discrete": True},
        "genetic_algorithm",
        "Discrete optimization — genetic algorithm explores combinatorial space",
    ),
    (
        "optimization",
        {},
        "simulated_annealing",
        "General optimization — SA escapes local minima",
    ),
    # Symbolic math
    (
        "symbolic_math",
        {},
        "cas",
        "Symbolic math — CAS provides exact solutions",
    ),
    # SMT
    (
        "arithmetic_constraints",
        {},
        "smt_lite",
        "Integer arithmetic constraints — SMT-lite solver",
    ),
    # Logic
    (
        "logic_programming",
        {},
        "prolog_lite",
        "Logic queries — Prolog-lite with unification",
    ),
    (
        "rule_based",
        {},
        "rule_engine",
        "Forward-chaining rule engine",
    ),
]


def select_algorithm(
    problem_type: str,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the best algorithm for a given problem type and features.

    Returns dict with 'algorithm', 'reasoning', and 'alternatives'.
    """
    features = features or {}
    matches: list[tuple[str, str, int]] = []

    for rule_type, rule_features, algo, reason in _SELECTION_RULES:
        if rule_type != problem_type:
            continue

        # Check if all rule features match
        match_count = 0
        is_match = True
        for k, v in rule_features.items():
            if k in features:
                if features[k] == v:
                    match_count += 1
                else:
                    is_match = False
                    break
            # If feature not specified, skip (don't reject)

        if is_match:
            matches.append((algo, reason, match_count))

    if not matches:
        return {
            "algorithm": "bfs",
            "reasoning": f"No specific rule for '{problem_type}', defaulting to BFS",
            "alternatives": [],
        }

    # Sort by specificity (number of matching features), most specific first
    matches.sort(key=lambda x: x[2], reverse=True)

    best_algo, best_reason, _ = matches[0]
    alternatives = [m[0] for m in matches[1:] if m[0] != best_algo]

    return {
        "algorithm": best_algo,
        "reasoning": best_reason,
        "alternatives": alternatives[:3],
    }
