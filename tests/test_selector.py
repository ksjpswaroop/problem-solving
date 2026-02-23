"""Phase 7: Algorithm selector and unified solve() API tests."""


class TestAlgorithmSelector:
    """Algorithm selector tests."""

    def test_selects_bfs_for_unweighted_pathfinding(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="pathfinding",
            features={"weighted": False, "needs_optimal": True},
        )
        assert result["algorithm"] == "bfs"

    def test_selects_astar_for_weighted_with_heuristic(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="pathfinding",
            features={"weighted": True, "has_heuristic": True, "needs_optimal": True},
        )
        assert result["algorithm"] == "astar"

    def test_selects_ucs_for_weighted_no_heuristic(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="pathfinding",
            features={"weighted": True, "has_heuristic": False, "needs_optimal": True},
        )
        assert result["algorithm"] == "ucs"

    def test_selects_greedy_for_speed(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="pathfinding",
            features={"weighted": True, "has_heuristic": True, "needs_optimal": False},
        )
        assert result["algorithm"] == "greedy"

    def test_selects_sat_for_boolean(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="satisfiability",
            features={"domain": "boolean"},
        )
        assert result["algorithm"] == "dpll_sat"

    def test_selects_csp_for_constraint(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="constraint_satisfaction",
            features={},
        )
        assert result["algorithm"] == "csp_backtracking"

    def test_selects_gradient_descent_for_differentiable(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="optimization",
            features={"differentiable": True},
        )
        assert result["algorithm"] == "gradient_descent"

    def test_selects_cas_for_symbolic(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="symbolic_math",
            features={},
        )
        assert result["algorithm"] == "cas"

    def test_returns_reasoning(self) -> None:
        from problemsolving.selector import select_algorithm

        result = select_algorithm(
            problem_type="pathfinding",
            features={"weighted": False},
        )
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0


class TestUnifiedSolveAPI:
    """Unified solve() function tests."""

    def test_solve_bfs_pathfinding(self) -> None:
        from problemsolving import solve

        result = solve(
            engine="bfs",
            input_data={
                "graph": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
                "start": "A",
                "goal": "D",
            },
        )
        assert result.status == "success"
        assert result.result is not None
        assert result.result["path"][-1] == "D"

    def test_solve_sat(self) -> None:
        from problemsolving import solve

        result = solve(
            engine="dpll_sat",
            input_data={"clauses": [[1, 2], [-1, 3]], "num_vars": 3},
        )
        assert result.status == "success"
        assert result.result is not None
        assert result.result["satisfiable"] is True

    def test_solve_unknown_engine(self) -> None:
        from problemsolving import solve

        result = solve(engine="nonexistent", input_data={})
        assert result.status == "error"
        assert result.error is not None

    def test_solve_returns_protocol_response(self) -> None:
        from problemsolving import solve
        from problemsolving.protocol.response import SolverResponse

        result = solve(
            engine="bfs",
            input_data={
                "graph": {"A": ["B"], "B": []},
                "start": "A",
                "goal": "B",
            },
        )
        assert isinstance(result, SolverResponse)

    def test_solve_with_auto_select(self) -> None:
        from problemsolving import solve

        # When engine="auto", selector picks the right algorithm
        result = solve(
            engine="auto",
            input_data={
                "graph": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
                "start": "A",
                "goal": "D",
            },
            problem_type="pathfinding",
            features={"weighted": False},
        )
        assert result.status == "success"
        assert result.result is not None

    def test_list_engines(self) -> None:
        from problemsolving import list_engines

        engines = list_engines()
        assert "bfs" in engines
        assert "astar" in engines
        assert "dpll_sat" in engines
        assert "cas" in engines
