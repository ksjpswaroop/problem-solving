"""DPLL SAT solver."""

from __future__ import annotations

from typing import Any


def dpll_solve(
    clauses: list[list[int]],
    num_vars: int,
) -> dict[str, Any]:
    """DPLL algorithm for Boolean satisfiability.

    clauses: list of clauses, each clause is a list of literals.
        Positive int = variable true, negative = variable false.
        e.g. [[1, 2], [-1, 3]] means (x1 ∨ x2) ∧ (¬x1 ∨ x3)
    num_vars: number of variables (1-indexed).

    Returns dict with 'satisfiable' bool and 'assignment' dict if SAT.
    """
    assignment: dict[int, bool] = {}
    result = _dpll(clauses, assignment, num_vars)
    if result is not None:
        # Fill in unassigned variables with True
        for v in range(1, num_vars + 1):
            if v not in result:
                result[v] = True
        return {"satisfiable": True, "assignment": result}
    return {"satisfiable": False, "assignment": {}}


def _dpll(
    clauses: list[list[int]],
    assignment: dict[int, bool],
    num_vars: int,
) -> dict[int, bool] | None:
    """Core DPLL recursive procedure."""
    # Simplify with current assignment
    clauses = _simplify(clauses, assignment)

    # All clauses satisfied
    if len(clauses) == 0:
        return dict(assignment)

    # Empty clause found → conflict
    if any(len(c) == 0 for c in clauses):
        return None

    # Unit propagation
    for clause in clauses:
        if len(clause) == 1:
            lit = clause[0]
            var = abs(lit)
            val = lit > 0
            new_assignment = dict(assignment)
            new_assignment[var] = val
            result = _dpll(clauses, new_assignment, num_vars)
            if result is not None:
                return result
            return None

    # Pure literal elimination
    all_lits: set[int] = set()
    for clause in clauses:
        for lit in clause:
            all_lits.add(lit)

    for lit in list(all_lits):
        if -lit not in all_lits:
            var = abs(lit)
            if var not in assignment:
                new_assignment = dict(assignment)
                new_assignment[var] = lit > 0
                result = _dpll(clauses, new_assignment, num_vars)
                if result is not None:
                    return result
                return None

    # Choose unassigned variable
    next_var = _choose_variable(clauses, assignment, num_vars)
    if next_var is None:
        return dict(assignment)

    # Try True then False
    for val in (True, False):
        new_assignment = dict(assignment)
        new_assignment[next_var] = val
        result = _dpll(clauses, new_assignment, num_vars)
        if result is not None:
            return result

    return None


def _simplify(
    clauses: list[list[int]], assignment: dict[int, bool]
) -> list[list[int]]:
    """Simplify clauses given current assignment."""
    simplified = []
    for clause in clauses:
        new_clause = []
        satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                if (lit > 0) == assignment[var]:
                    satisfied = True
                    break
                # Literal is false, skip it
            else:
                new_clause.append(lit)
        if not satisfied:
            simplified.append(new_clause)
    return simplified


def _choose_variable(
    clauses: list[list[int]],
    assignment: dict[int, bool],
    num_vars: int,
) -> int | None:
    """Choose the next unassigned variable (first unassigned)."""
    assigned = set(assignment.keys())
    for clause in clauses:
        for lit in clause:
            var = abs(lit)
            if var not in assigned:
                return var
    for v in range(1, num_vars + 1):
        if v not in assigned:
            return v
    return None


def dpll_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for DPLL SAT solver."""
    clauses = input_data["clauses"]
    num_vars = input_data["num_vars"]
    return dpll_solve(clauses, num_vars)
