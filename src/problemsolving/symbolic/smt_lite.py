"""SMT-lite solver: satisfiability of linear integer arithmetic constraints.

Uses SymPy to find integer solutions satisfying all constraints.
This is a lightweight alternative to full SMT (Z3) for simple cases.
"""

from __future__ import annotations

from typing import Any

import sympy


def smt_solve(
    variables: list[str],
    constraints: list[str],
) -> dict[str, Any]:
    """Solve SMT-lite: find integer assignment satisfying all constraints.

    variables: list of variable names.
    constraints: list of constraint strings (e.g. "x > 0", "x + y == 10").
    Returns dict with 'satisfiable' bool and 'assignment' if SAT.
    """
    syms = {name: sympy.Symbol(name, integer=True) for name in variables}

    # Parse all constraints
    # Must convert == to Eq() before sympify, since Python evaluates == immediately
    parsed = []
    for c in constraints:
        parsed.append(_parse_constraint(c, syms))

    # Check for immediate False
    for p in parsed:
        if p is sympy.false or p is False:
            return {"satisfiable": False, "assignment": {}}

    # Try SymPy's solve with relational constraints
    sym_list = [syms[v] for v in variables]

    # Extract equalities to solve first
    equalities = []
    other_constraints = []
    for p in parsed:
        if isinstance(p, sympy.Equality):
            equalities.append(p.lhs - p.rhs)
            other_constraints.append(p)
        elif p is sympy.true or p is True:
            continue
        else:
            other_constraints.append(p)

    # Solve equalities first
    if equalities:
        eq_sol = sympy.solve(equalities, sym_list, dict=True)
    else:
        eq_sol = [{}]

    if not eq_sol:
        return {"satisfiable": False, "assignment": {}}

    for sol in eq_sol:
        # Substitute into remaining constraints and check
        remaining = []
        all_ok = True
        for p in other_constraints:
            if isinstance(p, sympy.Equality):
                continue  # Already solved
            sub_p = p.subs(sol)
            if sub_p is sympy.true or sub_p is True:
                continue
            if sub_p is sympy.false or sub_p is False:
                all_ok = False
                break
            remaining.append(sub_p)

        if not all_ok:
            continue

        # Gather free variables
        free = set()
        for r in remaining:
            if hasattr(r, "free_symbols"):
                free.update(r.free_symbols)

        if not free:
            # All determined
            assignment = {str(k): int(v) for k, v in sol.items()}
            for v in variables:
                if v not in assignment:
                    assignment[v] = 0
            return {"satisfiable": True, "assignment": assignment}

        # Search for integer values of free variables
        result = _search_solution(list(free), remaining, sol, variables)
        if result is not None:
            return {"satisfiable": True, "assignment": result}

    return {"satisfiable": False, "assignment": {}}


def _parse_constraint(expr_str: str, syms: dict[str, Any]) -> Any:
    """Parse a constraint string, handling == specially for SymPy."""
    import re

    # Replace == with Eq() to avoid Python immediate evaluation
    if "==" in expr_str:
        # Split on == and create Eq
        match = re.match(r"(.+)==(.+)", expr_str)
        if match:
            lhs = sympy.sympify(match.group(1).strip(), locals=syms)
            rhs = sympy.sympify(match.group(2).strip(), locals=syms)
            return sympy.Eq(lhs, rhs)

    return sympy.sympify(expr_str, locals=syms)


def _search_solution(
    free_vars: list[sympy.Symbol],
    constraints: list[Any],
    eq_solution: dict[Any, Any],
    all_variables: list[str],
    search_range: int = 50,
) -> dict[str, int] | None:
    """Search for integer solutions by bounded enumeration."""
    from itertools import product as iproduct

    range_vals = range(-search_range, search_range + 1)

    for values in iproduct(range_vals, repeat=len(free_vars)):
        sub = dict(zip(free_vars, values))
        try:
            if all(_eval_constraint(c, sub) for c in constraints):
                assignment: dict[str, int] = {}
                for v in all_variables:
                    sym = sympy.Symbol(v, integer=True)
                    if sym in eq_solution:
                        val = eq_solution[sym]
                        if hasattr(val, "subs"):
                            val = val.subs(sub)
                        assignment[v] = int(val)
                    elif sym in sub:
                        assignment[v] = int(sub[sym])
                    else:
                        assignment[v] = 0
                return assignment
        except (TypeError, ValueError):
            continue

    return None


def _eval_constraint(constraint: Any, sub: dict[Any, Any]) -> bool:
    """Evaluate a constraint with substitution."""
    result = constraint.subs(sub) if hasattr(constraint, "subs") else constraint
    if result is sympy.true or result is True:
        return True
    if result is sympy.false or result is False:
        return False
    return bool(result)


def smt_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for SMT-lite solver."""
    return smt_solve(
        variables=input_data["variables"],
        constraints=input_data["constraints"],
    )
