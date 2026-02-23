"""Phase 1: Solver protocol request/response tests."""



class TestSolverRequest:
    """Tests for SolverRequest model."""

    def test_valid_request_parses(self) -> None:
        from problemsolving.protocol.request import SolverRequest

        req = SolverRequest(
            engine="bfs",
            operation="solve",
            input_data={"start": "A", "goal": "E", "graph": {"A": ["B", "C"]}},
        )
        assert req.engine == "bfs"
        assert req.operation == "solve"
        assert req.input_data["start"] == "A"

    def test_request_has_auto_generated_id(self) -> None:
        from problemsolving.protocol.request import SolverRequest

        req = SolverRequest(engine="bfs", operation="solve", input_data={})
        assert req.id is not None
        assert isinstance(req.id, str)
        assert len(req.id) > 0

    def test_request_with_custom_id(self) -> None:
        from problemsolving.protocol.request import SolverRequest

        req = SolverRequest(id="custom_123", engine="bfs", operation="solve", input_data={})
        assert req.id == "custom_123"

    def test_request_with_config(self) -> None:
        from problemsolving.protocol.request import SolverRequest

        req = SolverRequest(
            engine="sat",
            operation="solve",
            input_data={},
            config={"timeout_ms": 5000, "produce_proof": True},
        )
        assert req.config["timeout_ms"] == 5000

    def test_request_default_config_is_empty(self) -> None:
        from problemsolving.protocol.request import SolverRequest

        req = SolverRequest(engine="bfs", operation="solve", input_data={})
        assert req.config == {}

    def test_request_serializes_to_dict(self) -> None:
        from problemsolving.protocol.request import SolverRequest

        req = SolverRequest(engine="bfs", operation="solve", input_data={"start": "A"})
        d = req.to_dict()
        assert d["engine"] == "bfs"
        assert d["operation"] == "solve"
        assert d["input"]["start"] == "A"


class TestSolverResponse:
    """Tests for SolverResponse model."""

    def test_success_response(self) -> None:
        from problemsolving.protocol.response import SolverResponse

        resp = SolverResponse.success(
            request_id="req_1",
            engine="bfs",
            result={"path": ["A", "C", "E"], "cost": 2},
        )
        assert resp.status == "success"
        assert resp.result["path"] == ["A", "C", "E"]
        assert resp.error is None

    def test_error_response(self) -> None:
        from problemsolving.protocol.response import SolverResponse

        resp = SolverResponse.make_error(
            request_id="req_1",
            engine="bfs",
            code="NO_PATH",
            message="No path found from start to goal",
        )
        assert resp.status == "error"
        assert resp.error is not None
        assert resp.error["code"] == "NO_PATH"
        assert resp.result is None

    def test_response_with_proof_trace(self) -> None:
        from problemsolving.protocol.response import SolverResponse

        resp = SolverResponse.success(
            request_id="req_1",
            engine="bfs",
            result={"path": ["A", "E"]},
            proof_trace=[{"step": 1, "action": "expand", "node": "A"}],
        )
        assert len(resp.proof_trace) == 1
        assert resp.proof_trace[0]["action"] == "expand"

    def test_response_with_metadata(self) -> None:
        from problemsolving.protocol.response import SolverResponse

        resp = SolverResponse.success(
            request_id="req_1",
            engine="bfs",
            result={},
            metadata={"time_ms": 12, "nodes_explored": 47},
        )
        assert resp.metadata["time_ms"] == 12

    def test_response_serializes_to_dict(self) -> None:
        from problemsolving.protocol.response import SolverResponse

        resp = SolverResponse.success(
            request_id="req_1", engine="bfs", result={"path": ["A"]}
        )
        d = resp.to_dict()
        assert d["status"] == "success"
        assert "result" in d
        assert "id" in d


class TestEngineRegistry:
    """Tests for engine registry."""

    def test_register_and_retrieve_engine(self) -> None:
        from problemsolving.protocol.registry import EngineRegistry

        registry = EngineRegistry()

        def dummy_solve(req: dict) -> dict:
            return {"result": "ok"}

        registry.register("test_engine", solve_fn=dummy_solve, tags=["test"])
        engine = registry.get("test_engine")
        assert engine is not None
        assert engine.name == "test_engine"

    def test_get_unknown_engine_returns_none(self) -> None:
        from problemsolving.protocol.registry import EngineRegistry

        registry = EngineRegistry()
        assert registry.get("nonexistent") is None

    def test_list_engines(self) -> None:
        from problemsolving.protocol.registry import EngineRegistry

        registry = EngineRegistry()
        registry.register("engine_a", solve_fn=lambda r: {}, tags=["search"])
        registry.register("engine_b", solve_fn=lambda r: {}, tags=["symbolic"])
        names = registry.list_engines()
        assert "engine_a" in names
        assert "engine_b" in names

    def test_list_engines_by_tag(self) -> None:
        from problemsolving.protocol.registry import EngineRegistry

        registry = EngineRegistry()
        registry.register("engine_a", solve_fn=lambda r: {}, tags=["search"])
        registry.register("engine_b", solve_fn=lambda r: {}, tags=["symbolic"])
        search_engines = registry.list_engines(tag="search")
        assert "engine_a" in search_engines
        assert "engine_b" not in search_engines
