"""Phase 5: SMT-lite and CAS (Computer Algebra System) tests."""


class TestCAS:
    """Computer Algebra System tests (SymPy-backed)."""

    def test_solve_linear_equation(self) -> None:
        from problemsolving.symbolic.cas import cas_solve

        # 2x + 3 = 7 → x = 2
        result = cas_solve(equation="2*x + 3 - 7", variable="x")
        assert result is not None
        assert result["solutions"] == [2]

    def test_solve_quadratic(self) -> None:
        from problemsolving.symbolic.cas import cas_solve

        # x^2 - 5x + 6 = 0 → x = 2, 3
        result = cas_solve(equation="x**2 - 5*x + 6", variable="x")
        assert result is not None
        assert sorted(result["solutions"]) == [2, 3]

    def test_solve_system_of_equations(self) -> None:
        from problemsolving.symbolic.cas import cas_solve_system

        # x + y = 10, x - y = 4 → x=7, y=3
        result = cas_solve_system(
            equations=["x + y - 10", "x - y - 4"],
            variables=["x", "y"],
        )
        assert result is not None
        assert result["solution"]["x"] == 7
        assert result["solution"]["y"] == 3

    def test_differentiate(self) -> None:
        from problemsolving.symbolic.cas import cas_differentiate

        # d/dx (x^3 + 2x) = 3x^2 + 2
        result = cas_differentiate(expression="x**3 + 2*x", variable="x")
        assert result is not None
        assert result["derivative"] == "3*x**2 + 2"

    def test_integrate(self) -> None:
        from problemsolving.symbolic.cas import cas_integrate

        # ∫ 2x dx = x^2 (ignoring constant)
        result = cas_integrate(expression="2*x", variable="x")
        assert result is not None
        assert result["integral"] == "x**2"

    def test_simplify(self) -> None:
        from problemsolving.symbolic.cas import cas_simplify

        # (x^2 - 1) / (x - 1) = x + 1
        result = cas_simplify(expression="(x**2 - 1) / (x - 1)")
        assert result is not None
        assert result["simplified"] == "x + 1"

    def test_no_solution_returns_empty(self) -> None:
        from problemsolving.symbolic.cas import cas_solve

        # x^2 + 1 = 0 has no real solutions
        result = cas_solve(equation="x**2 + 1", variable="x", domain="real")
        assert result is not None
        assert result["solutions"] == []


class TestSMTLite:
    """SMT-lite solver tests (linear arithmetic over integers)."""

    def test_satisfiable_constraints(self) -> None:
        from problemsolving.symbolic.smt_lite import smt_solve

        # x > 0, x < 10, y = x + 5
        result = smt_solve(
            variables=["x", "y"],
            constraints=["x > 0", "x < 10", "y == x + 5"],
        )
        assert result is not None
        assert result["satisfiable"] is True
        x = result["assignment"]["x"]
        y = result["assignment"]["y"]
        assert 0 < x < 10
        assert y == x + 5

    def test_unsatisfiable(self) -> None:
        from problemsolving.symbolic.smt_lite import smt_solve

        # x > 10, x < 5 — impossible
        result = smt_solve(
            variables=["x"],
            constraints=["x > 10", "x < 5"],
        )
        assert result is not None
        assert result["satisfiable"] is False

    def test_equality_constraint(self) -> None:
        from problemsolving.symbolic.smt_lite import smt_solve

        # x == 42
        result = smt_solve(
            variables=["x"],
            constraints=["x == 42"],
        )
        assert result is not None
        assert result["satisfiable"] is True
        assert result["assignment"]["x"] == 42

    def test_multiple_variables(self) -> None:
        from problemsolving.symbolic.smt_lite import smt_solve

        # x + y == 10, x > y
        result = smt_solve(
            variables=["x", "y"],
            constraints=["x + y == 10", "x > y"],
        )
        assert result is not None
        assert result["satisfiable"] is True
        x = result["assignment"]["x"]
        y = result["assignment"]["y"]
        assert x + y == 10
        assert x > y
