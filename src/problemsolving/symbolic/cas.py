"""Computer Algebra System (CAS) backed by SymPy."""

from __future__ import annotations

from typing import Any

import sympy


def cas_solve(
    equation: str,
    variable: str,
    domain: str = "complex",
) -> dict[str, Any]:
    """Solve a single equation for a variable.

    equation: expression that equals zero (e.g. "x**2 - 4" means x^2 = 4).
    variable: variable name to solve for.
    domain: "real" or "complex".
    Returns dict with 'solutions' list.
    """
    var = sympy.Symbol(variable)
    expr = sympy.sympify(equation, locals={variable: var})

    solutions = sympy.solve(expr, var)

    # Filter to real solutions if requested
    if domain == "real":
        solutions = [s for s in solutions if s.is_real]

    result_list: list[int | float] = []
    for s in solutions:
        try:
            if s == int(s):
                result_list.append(int(s))
            else:
                result_list.append(float(s))
        except (TypeError, ValueError):
            pass

    return {
        "solutions": result_list,
        "variable": variable,
    }


def cas_solve_system(
    equations: list[str],
    variables: list[str],
) -> dict[str, Any]:
    """Solve a system of equations.

    equations: list of expressions equal to zero.
    variables: list of variable names.
    Returns dict with 'solution' mapping variable names to values.
    """
    syms = {name: sympy.Symbol(name) for name in variables}
    exprs = [sympy.sympify(eq, locals=syms) for eq in equations]
    sym_list = [syms[v] for v in variables]

    result = sympy.solve(exprs, sym_list, dict=True)

    if not result:
        return {"solution": {}, "solvable": False}

    sol = result[0]
    return {
        "solution": {
            str(k): int(v) if v == int(v) else float(v) for k, v in sol.items()
        },
        "solvable": True,
    }


def cas_differentiate(
    expression: str,
    variable: str,
) -> dict[str, Any]:
    """Differentiate an expression with respect to a variable."""
    var = sympy.Symbol(variable)
    expr = sympy.sympify(expression, locals={variable: var})
    deriv = sympy.diff(expr, var)
    return {"derivative": str(deriv), "variable": variable}


def cas_integrate(
    expression: str,
    variable: str,
) -> dict[str, Any]:
    """Integrate an expression with respect to a variable (indefinite)."""
    var = sympy.Symbol(variable)
    expr = sympy.sympify(expression, locals={variable: var})
    integral = sympy.integrate(expr, var)
    return {"integral": str(integral), "variable": variable}


def cas_simplify(expression: str) -> dict[str, Any]:
    """Simplify a symbolic expression."""
    expr = sympy.sympify(expression)
    simplified = sympy.simplify(expr)
    return {"simplified": str(simplified)}


def cas_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for CAS."""
    operation = input_data.get("operation", "solve")

    if operation == "solve":
        return cas_solve(
            equation=input_data["equation"],
            variable=input_data["variable"],
            domain=input_data.get("domain", "complex"),
        )
    if operation == "solve_system":
        return cas_solve_system(
            equations=input_data["equations"],
            variables=input_data["variables"],
        )
    if operation == "differentiate":
        return cas_differentiate(
            expression=input_data["expression"],
            variable=input_data["variable"],
        )
    if operation == "integrate":
        return cas_integrate(
            expression=input_data["expression"],
            variable=input_data["variable"],
        )
    if operation == "simplify":
        return cas_simplify(expression=input_data["expression"])

    return {"error": f"Unknown CAS operation: {operation}"}
