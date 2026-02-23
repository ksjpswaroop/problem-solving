"""CSP solver with backtracking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def csp_solve(
    variables: list[str],
    domains: dict[str, list[Any]],
    constraints: list[tuple[str, str, Callable[..., bool]]],
) -> dict[str, Any]:
    """Solve a CSP using backtracking with constraint checking.

    variables: list of variable names.
    domains: dict mapping variable name to list of possible values.
    constraints: list of (var1, var2, check_fn) tuples.
        check_fn(val1, val2) returns True if consistent.

    Returns dict with 'satisfiable', 'assignment', 'nodes_explored'.
    """
    assignment: dict[str, Any] = {}
    context = {"nodes_explored": 0}

    # Build constraint lookup for quick access
    constraint_map: dict[str, list[tuple[str, Callable[..., bool]]]] = {
        v: [] for v in variables
    }
    for v1, v2, check_fn in constraints:
        constraint_map[v1].append((v2, check_fn))
        constraint_map[v2].append((v1, lambda a, b, fn=check_fn: fn(b, a)))

    result = _backtrack(variables, domains, constraints, assignment, context)
    if result is not None:
        return {
            "satisfiable": True,
            "assignment": result,
            "nodes_explored": context["nodes_explored"],
        }
    return {
        "satisfiable": False,
        "assignment": {},
        "nodes_explored": context["nodes_explored"],
    }


def _backtrack(
    variables: list[str],
    domains: dict[str, list[Any]],
    constraints: list[tuple[str, str, Callable[..., bool]]],
    assignment: dict[str, Any],
    context: dict[str, int],
) -> dict[str, Any] | None:
    """Recursive backtracking."""
    if len(assignment) == len(variables):
        return dict(assignment)

    # Pick next unassigned variable (MRV — minimum remaining values)
    unassigned = [v for v in variables if v not in assignment]
    var = min(unassigned, key=lambda v: len(domains[v]))

    context["nodes_explored"] += 1

    for value in domains[var]:
        if _is_consistent(var, value, assignment, constraints):
            assignment[var] = value
            result = _backtrack(variables, domains, constraints, assignment, context)
            if result is not None:
                return result
            del assignment[var]

    return None


def _is_consistent(
    var: str,
    value: Any,
    assignment: dict[str, Any],
    constraints: list[tuple[str, str, Callable[..., bool]]],
) -> bool:
    """Check if assigning value to var is consistent with constraints."""
    for v1, v2, check_fn in constraints:
        if v1 == var and v2 in assignment:
            if not check_fn(value, assignment[v2]):
                return False
        elif v2 == var and v1 in assignment:
            if not check_fn(assignment[v1], value):
                return False
    return True


def csp_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for CSP solver."""
    variables = input_data["variables"]
    domains = input_data["domains"]

    # Constraints come as inequality pairs for protocol
    constraint_pairs = input_data.get("not_equal_constraints", [])
    constraints = [
        (v1, v2, lambda a, b: a != b)
        for v1, v2 in constraint_pairs
    ]

    return csp_solve(variables=variables, domains=domains, constraints=constraints)
