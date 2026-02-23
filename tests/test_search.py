"""Phase 2: Search algorithm family tests (DFS, UCS, A*, Greedy)."""

import time


class TestDFS:
    """DFS correctness tests."""

    def test_finds_a_path(self) -> None:
        from problemsolving.search.dfs import dfs_solve

        graph = {"A": ["B", "C"], "B": ["D"], "C": ["D", "E"], "D": ["E"], "E": []}
        result = dfs_solve(
            start="A", goal_test=lambda n: n == "E", neighbors=lambda n: graph.get(n, [])
        )
        assert result is not None
        assert result["path"][0] == "A"
        assert result["path"][-1] == "E"

    def test_no_path_returns_none(self) -> None:
        from problemsolving.search.dfs import dfs_solve

        graph = {"A": ["B"], "B": [], "C": []}
        result = dfs_solve(
            start="A", goal_test=lambda n: n == "C", neighbors=lambda n: graph.get(n, [])
        )
        assert result is None

    def test_cycle_handling(self) -> None:
        from problemsolving.search.dfs import dfs_solve

        graph = {"A": ["B"], "B": ["C", "A"], "C": ["A", "D"], "D": []}
        result = dfs_solve(
            start="A", goal_test=lambda n: n == "D", neighbors=lambda n: graph.get(n, [])
        )
        assert result is not None
        assert result["path"][-1] == "D"

    def test_depth_limited(self) -> None:
        from problemsolving.search.dfs import dfs_solve

        # Linear chain: A→B→C→D→E, depth limit 2 won't find E
        graph = {"A": ["B"], "B": ["C"], "C": ["D"], "D": ["E"], "E": []}
        result = dfs_solve(
            start="A",
            goal_test=lambda n: n == "E",
            neighbors=lambda n: graph.get(n, []),
            max_depth=2,
        )
        assert result is None

    def test_start_is_goal(self) -> None:
        from problemsolving.search.dfs import dfs_solve

        result = dfs_solve(start="A", goal_test=lambda n: n == "A", neighbors=lambda n: [])
        assert result is not None
        assert result["path"] == ["A"]


class TestUCS:
    """Uniform Cost Search tests."""

    def test_cheapest_path_not_shortest_hops(self) -> None:
        from problemsolving.search.ucs import ucs_solve

        # A→B costs 1, A→C costs 10, B→D costs 1, C→D costs 1
        # Cheapest: A→B→D (cost 2), not A→C→D (cost 11)
        graph = {"A": [("B", 1), ("C", 10)], "B": [("D", 1)], "C": [("D", 1)], "D": []}
        result = ucs_solve(
            start="A",
            goal_test=lambda n: n == "D",
            neighbors=lambda n: graph.get(n, []),
        )
        assert result is not None
        assert result["path"] == ["A", "B", "D"]
        assert result["cost"] == 2

    def test_no_path_returns_none(self) -> None:
        from problemsolving.search.ucs import ucs_solve

        graph = {"A": [("B", 1)], "B": [], "C": []}
        result = ucs_solve(
            start="A", goal_test=lambda n: n == "C", neighbors=lambda n: graph.get(n, [])
        )
        assert result is None

    def test_zero_cost_edges(self) -> None:
        from problemsolving.search.ucs import ucs_solve

        graph = {"A": [("B", 0), ("C", 1)], "B": [("D", 0)], "C": [("D", 0)], "D": []}
        result = ucs_solve(
            start="A", goal_test=lambda n: n == "D", neighbors=lambda n: graph.get(n, [])
        )
        assert result is not None
        assert result["cost"] == 0

    def test_start_is_goal(self) -> None:
        from problemsolving.search.ucs import ucs_solve

        result = ucs_solve(start="A", goal_test=lambda n: n == "A", neighbors=lambda n: [])
        assert result is not None
        assert result["cost"] == 0


class TestAStar:
    """A* Search tests."""

    def test_optimal_with_admissible_heuristic(self) -> None:
        from problemsolving.search.astar import astar_solve

        # Weighted graph
        graph = {
            "A": [("B", 4), ("C", 2)],
            "B": [("D", 3)],
            "C": [("D", 1), ("E", 5)],
            "D": [("E", 2)],
            "E": [],
        }
        # Admissible heuristic (underestimates)
        h = {"A": 5, "B": 3, "C": 3, "D": 2, "E": 0}
        result = astar_solve(
            start="A",
            goal_test=lambda n: n == "E",
            neighbors=lambda n: graph.get(n, []),
            heuristic=lambda n: h.get(n, 0),
        )
        assert result is not None
        # Optimal: A→C→D→E (cost 2+1+2=5)
        assert result["cost"] == 5
        assert result["path"] == ["A", "C", "D", "E"]

    def test_no_path(self) -> None:
        from problemsolving.search.astar import astar_solve

        graph = {"A": [("B", 1)], "B": [], "C": []}
        result = astar_solve(
            start="A",
            goal_test=lambda n: n == "C",
            neighbors=lambda n: graph.get(n, []),
            heuristic=lambda n: 0,
        )
        assert result is None

    def test_grid_manhattan_heuristic(self) -> None:
        from problemsolving.search.astar import astar_solve

        size = 20

        def neighbors(node: tuple) -> list:
            r, c = node
            result = []
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    result.append(((nr, nc), 1))
            return result

        def manhattan(node: tuple) -> int:
            return abs(node[0] - (size - 1)) + abs(node[1] - (size - 1))

        result = astar_solve(
            start=(0, 0),
            goal_test=lambda n: n == (size - 1, size - 1),
            neighbors=neighbors,
            heuristic=manhattan,
        )
        assert result is not None
        assert result["cost"] == 2 * (size - 1)  # Manhattan distance

    def test_large_graph_performance(self) -> None:
        from problemsolving.search.astar import astar_solve

        size = 100

        def neighbors(node: tuple) -> list:
            r, c = node
            result = []
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    result.append(((nr, nc), 1))
            return result

        def manhattan(node: tuple) -> int:
            return abs(node[0] - (size - 1)) + abs(node[1] - (size - 1))

        start_time = time.time()
        result = astar_solve(
            start=(0, 0),
            goal_test=lambda n: n == (size - 1, size - 1),
            neighbors=neighbors,
            heuristic=manhattan,
        )
        elapsed = time.time() - start_time
        assert result is not None
        assert elapsed < 2.0


class TestGreedy:
    """Greedy Best-First Search tests."""

    def test_finds_path_fast(self) -> None:
        from problemsolving.search.greedy import greedy_solve

        graph = {
            "A": [("B", 4), ("C", 2)],
            "B": [("D", 3)],
            "C": [("D", 1), ("E", 5)],
            "D": [("E", 2)],
            "E": [],
        }
        h = {"A": 5, "B": 4, "C": 2, "D": 1, "E": 0}
        result = greedy_solve(
            start="A",
            goal_test=lambda n: n == "E",
            neighbors=lambda n: graph.get(n, []),
            heuristic=lambda n: h.get(n, 0),
        )
        assert result is not None
        assert result["path"][0] == "A"
        assert result["path"][-1] == "E"

    def test_no_path(self) -> None:
        from problemsolving.search.greedy import greedy_solve

        graph = {"A": [("B", 1)], "B": [], "C": []}
        result = greedy_solve(
            start="A",
            goal_test=lambda n: n == "C",
            neighbors=lambda n: graph.get(n, []),
            heuristic=lambda n: 0,
        )
        assert result is None

    def test_not_necessarily_optimal(self) -> None:
        """Greedy may find suboptimal paths — that's expected."""
        from problemsolving.search.greedy import greedy_solve

        # Heuristic misleads: direct path looks good but costs more
        graph = {"A": [("B", 1), ("C", 100)], "B": [("D", 100)], "C": [("D", 1)], "D": []}
        # Heuristic says C is closer to D
        h = {"A": 10, "B": 50, "C": 1, "D": 0}
        result = greedy_solve(
            start="A",
            goal_test=lambda n: n == "D",
            neighbors=lambda n: graph.get(n, []),
            heuristic=lambda n: h.get(n, 0),
        )
        assert result is not None
        # Greedy follows heuristic → A→C→D (cost 101) instead of A→B→D (cost 101)
        # Both happen to cost same here, just testing it finds a path
        assert result["path"][-1] == "D"


class TestSearchComparison:
    """Cross-algorithm comparison tests."""

    def test_bfs_finds_shortest_hops_ucs_finds_cheapest(self) -> None:
        from problemsolving.search.bfs import bfs_solve
        from problemsolving.search.ucs import ucs_solve

        # Graph where shortest hops ≠ cheapest cost
        # A→B→E: 2 hops, cost 1+100=101
        # A→C→D→E: 3 hops, cost 1+1+1=3
        weighted = {
            "A": [("B", 1), ("C", 1)],
            "B": [("E", 100)],
            "C": [("D", 1)],
            "D": [("E", 1)],
            "E": [],
        }
        unweighted = {k: [n for n, _ in v] for k, v in weighted.items()}

        bfs_result = bfs_solve(
            start="A", goal_test=lambda n: n == "E", neighbors=lambda n: unweighted.get(n, [])
        )
        ucs_result = ucs_solve(
            start="A", goal_test=lambda n: n == "E", neighbors=lambda n: weighted.get(n, [])
        )

        assert bfs_result is not None
        assert ucs_result is not None
        # BFS: fewest hops (A→B→E = 2 hops)
        assert len(bfs_result["path"]) <= 3
        # UCS: cheapest (A→C→D→E = cost 3)
        assert ucs_result["cost"] == 3
