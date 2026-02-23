"""Phase 1: BFS algorithm tests."""



class TestBFS:
    """BFS correctness tests."""

    def test_simple_graph_shortest_path(self) -> None:
        """BFS finds shortest path (fewest hops) in simple graph."""
        from problemsolving.search.bfs import bfs_solve

        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D", "E"],
            "D": ["E"],
            "E": [],
        }
        result = bfs_solve(
            start="A",
            goal_test=lambda n: n == "E",
            neighbors=lambda n: graph.get(n, []),
        )
        assert result is not None
        assert result["path"][0] == "A"
        assert result["path"][-1] == "E"
        # Shortest path is A→C→E (2 hops), not A→B→D→E (3 hops)
        assert len(result["path"]) == 3

    def test_no_path_returns_none(self) -> None:
        """BFS returns None when no path exists."""
        from problemsolving.search.bfs import bfs_solve

        graph = {"A": ["B"], "B": [], "C": []}
        result = bfs_solve(
            start="A",
            goal_test=lambda n: n == "C",
            neighbors=lambda n: graph.get(n, []),
        )
        assert result is None

    def test_start_is_goal(self) -> None:
        """BFS returns immediately when start == goal."""
        from problemsolving.search.bfs import bfs_solve

        result = bfs_solve(
            start="A",
            goal_test=lambda n: n == "A",
            neighbors=lambda n: [],
        )
        assert result is not None
        assert result["path"] == ["A"]
        assert result["nodes_explored"] == 1

    def test_cycle_does_not_infinite_loop(self) -> None:
        """BFS handles cycles without infinite loop."""
        from problemsolving.search.bfs import bfs_solve

        graph = {"A": ["B"], "B": ["C", "A"], "C": ["A", "D"], "D": []}
        result = bfs_solve(
            start="A",
            goal_test=lambda n: n == "D",
            neighbors=lambda n: graph.get(n, []),
        )
        assert result is not None
        assert result["path"][-1] == "D"

    def test_large_grid_performance(self) -> None:
        """BFS solves 100×100 grid in reasonable time."""
        import time

        from problemsolving.search.bfs import bfs_solve

        size = 100

        def neighbors(node: tuple) -> list:
            r, c = node
            result = []
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    result.append((nr, nc))
            return result

        start_time = time.time()
        result = bfs_solve(
            start=(0, 0),
            goal_test=lambda n: n == (size - 1, size - 1),
            neighbors=neighbors,
        )
        elapsed = time.time() - start_time

        assert result is not None
        assert result["path"][0] == (0, 0)
        assert result["path"][-1] == (99, 99)
        # Manhattan distance = 198 moves
        assert len(result["path"]) == 199
        assert elapsed < 2.0  # should be well under 1s

    def test_result_includes_nodes_explored(self) -> None:
        """BFS result includes count of explored nodes."""
        from problemsolving.search.bfs import bfs_solve

        graph = {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
        result = bfs_solve(
            start="A",
            goal_test=lambda n: n == "D",
            neighbors=lambda n: graph.get(n, []),
        )
        assert result is not None
        assert "nodes_explored" in result
        assert result["nodes_explored"] >= 3  # at least A, B, D

    def test_multiple_shortest_paths(self) -> None:
        """BFS finds one of the shortest paths when multiple exist."""
        from problemsolving.search.bfs import bfs_solve

        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }
        result = bfs_solve(
            start="A",
            goal_test=lambda n: n == "D",
            neighbors=lambda n: graph.get(n, []),
        )
        assert result is not None
        assert len(result["path"]) == 3  # A→B→D or A→C→D


class TestBFSProtocolIntegration:
    """BFS through the solver protocol layer."""

    def test_bfs_via_protocol(self) -> None:
        """Submit BFS problem via protocol, get protocol response."""
        from problemsolving.protocol.registry import get_default_registry
        from problemsolving.protocol.request import SolverRequest

        registry = get_default_registry()
        engine = registry.get("bfs")
        assert engine is not None, "BFS engine not registered"

        req = SolverRequest(
            engine="bfs",
            operation="solve",
            input_data={
                "start": "A",
                "goal": "E",
                "graph": {
                    "A": ["B", "C"],
                    "B": ["D"],
                    "C": ["D", "E"],
                    "D": ["E"],
                    "E": [],
                },
            },
        )
        resp = engine.handle(req)
        assert resp.status == "success"
        assert resp.result["path"][-1] == "E"
        assert resp.engine == "bfs"
