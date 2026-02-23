"""Phase 4: SAT solver and CSP tests."""


class TestDPLL:
    """DPLL SAT solver tests."""

    def test_satisfiable_simple(self) -> None:
        from problemsolving.csp.sat import dpll_solve

        # (A ∨ B) ∧ (¬A ∨ C) — satisfiable
        clauses = [[1, 2], [-1, 3]]
        result = dpll_solve(clauses, num_vars=3)
        assert result is not None
        assert result["satisfiable"] is True
        assignment = result["assignment"]
        # Verify assignment satisfies all clauses
        for clause in clauses:
            assert any(
                (assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)])
                for lit in clause
            )

    def test_unsatisfiable(self) -> None:
        from problemsolving.csp.sat import dpll_solve

        # (A) ∧ (¬A) — unsatisfiable
        clauses = [[1], [-1]]
        result = dpll_solve(clauses, num_vars=1)
        assert result is not None
        assert result["satisfiable"] is False

    def test_unit_propagation(self) -> None:
        from problemsolving.csp.sat import dpll_solve

        # Unit clause forces A=True, then (¬A ∨ B) forces B=True
        clauses = [[1], [-1, 2]]
        result = dpll_solve(clauses, num_vars=2)
        assert result is not None
        assert result["satisfiable"] is True
        assert result["assignment"][1] is True
        assert result["assignment"][2] is True

    def test_pure_literal_elimination(self) -> None:
        from problemsolving.csp.sat import dpll_solve

        # A appears only positive → set True
        clauses = [[1, 2], [1, -2]]
        result = dpll_solve(clauses, num_vars=2)
        assert result is not None
        assert result["satisfiable"] is True

    def test_empty_clauses_satisfiable(self) -> None:
        from problemsolving.csp.sat import dpll_solve

        # No clauses → trivially satisfiable
        result = dpll_solve([], num_vars=0)
        assert result is not None
        assert result["satisfiable"] is True

    def test_3sat_instance(self) -> None:
        from problemsolving.csp.sat import dpll_solve

        # A 3-SAT instance
        clauses = [
            [1, 2, 3],
            [-1, -2, 3],
            [1, -2, -3],
            [-1, 2, -3],
        ]
        result = dpll_solve(clauses, num_vars=3)
        assert result is not None
        assert result["satisfiable"] is True


class TestCSPBacktracking:
    """CSP with backtracking tests."""

    def test_simple_map_coloring(self) -> None:
        from problemsolving.csp.backtracking import csp_solve

        # 3 regions, 3 colors, adjacent regions must differ
        variables = ["WA", "NT", "SA"]
        domains = {v: ["red", "green", "blue"] for v in variables}
        constraints = [
            ("WA", "NT", lambda a, b: a != b),
            ("WA", "SA", lambda a, b: a != b),
            ("NT", "SA", lambda a, b: a != b),
        ]
        result = csp_solve(variables=variables, domains=domains, constraints=constraints)
        assert result is not None
        assert result["satisfiable"] is True
        assignment = result["assignment"]
        assert assignment["WA"] != assignment["NT"]
        assert assignment["WA"] != assignment["SA"]
        assert assignment["NT"] != assignment["SA"]

    def test_unsatisfiable_csp(self) -> None:
        from problemsolving.csp.backtracking import csp_solve

        # 3 variables, 2 colors, all must differ — impossible
        variables = ["A", "B", "C"]
        domains = {v: ["red", "blue"] for v in variables}
        constraints = [
            ("A", "B", lambda a, b: a != b),
            ("B", "C", lambda a, b: a != b),
            ("A", "C", lambda a, b: a != b),
        ]
        result = csp_solve(variables=variables, domains=domains, constraints=constraints)
        assert result is not None
        assert result["satisfiable"] is False

    def test_4queens(self) -> None:
        from problemsolving.csp.backtracking import csp_solve

        # 4-Queens: place 4 queens on 4x4 board, no two attacking
        variables = ["Q1", "Q2", "Q3", "Q4"]
        domains = {v: [0, 1, 2, 3] for v in variables}

        def queens_safe(col_i: int, col_j: int, row_diff: int) -> bool:
            return col_i != col_j and abs(col_i - col_j) != row_diff

        constraints = []
        for i in range(4):
            for j in range(i + 1, 4):
                diff = j - i
                constraints.append(
                    (variables[i], variables[j], lambda a, b, d=diff: queens_safe(a, b, d))
                )

        result = csp_solve(variables=variables, domains=domains, constraints=constraints)
        assert result is not None
        assert result["satisfiable"] is True
        cols = [result["assignment"][v] for v in variables]
        # Verify: no two queens share column or diagonal
        for i in range(4):
            for j in range(i + 1, 4):
                assert cols[i] != cols[j]
                assert abs(cols[i] - cols[j]) != j - i

    def test_single_variable(self) -> None:
        from problemsolving.csp.backtracking import csp_solve

        result = csp_solve(
            variables=["X"],
            domains={"X": [1, 2, 3]},
            constraints=[],
        )
        assert result is not None
        assert result["satisfiable"] is True
        assert result["assignment"]["X"] in [1, 2, 3]

    def test_nodes_explored_tracked(self) -> None:
        from problemsolving.csp.backtracking import csp_solve

        variables = ["A", "B"]
        domains = {v: [1, 2] for v in variables}
        constraints = [("A", "B", lambda a, b: a != b)]
        result = csp_solve(variables=variables, domains=domains, constraints=constraints)
        assert result is not None
        assert "nodes_explored" in result
