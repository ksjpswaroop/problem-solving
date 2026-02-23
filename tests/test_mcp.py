"""Phase 8: MCP server tool handler tests."""



class TestMCPToolHandlers:
    """Test MCP tool handlers (without running actual server)."""

    def test_solve_tool(self) -> None:
        from problemsolving.mcp.handlers import handle_solve

        result = handle_solve({
            "engine": "bfs",
            "input": {
                "graph": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
                "start": "A",
                "goal": "D",
            },
        })
        assert result["status"] == "success"
        assert result["result"]["path"][-1] == "D"

    def test_solve_tool_error(self) -> None:
        from problemsolving.mcp.handlers import handle_solve

        result = handle_solve({"engine": "nonexistent", "input": {}})
        assert result["status"] == "error"

    def test_select_algorithm_tool(self) -> None:
        from problemsolving.mcp.handlers import handle_select_algorithm

        result = handle_select_algorithm({
            "problem_type": "pathfinding",
            "features": {"weighted": False},
        })
        assert result["algorithm"] == "bfs"
        assert "reasoning" in result

    def test_list_engines_tool(self) -> None:
        from problemsolving.mcp.handlers import handle_list_engines

        result = handle_list_engines({})
        assert "engines" in result
        assert "bfs" in result["engines"]
        assert "astar" in result["engines"]

    def test_list_engines_with_tag(self) -> None:
        from problemsolving.mcp.handlers import handle_list_engines

        result = handle_list_engines({"tag": "optimization"})
        engines = result["engines"]
        assert "gradient_descent" in engines
        assert "bfs" not in engines

    def test_verify_tool(self) -> None:
        from problemsolving.mcp.handlers import handle_verify

        result = handle_verify({
            "engine": "bfs",
            "input": {
                "graph": {"A": ["B"], "B": ["C"], "C": []},
                "start": "A",
                "goal": "C",
            },
            "expected": {"path": ["A", "B", "C"]},
        })
        assert result["verified"] is True

    def test_verify_tool_mismatch(self) -> None:
        from problemsolving.mcp.handlers import handle_verify

        result = handle_verify({
            "engine": "bfs",
            "input": {
                "graph": {"A": ["B"], "B": ["C"], "C": []},
                "start": "A",
                "goal": "C",
            },
            "expected": {"path": ["A", "C"]},
        })
        assert result["verified"] is False

    def test_explain_algorithm_tool(self) -> None:
        from problemsolving.mcp.handlers import handle_explain_algorithm

        result = handle_explain_algorithm({"algorithm": "bfs"})
        assert result["found"] is True
        assert result["name"] == "Breadth-First Search"
        assert "when_to_use" in result

    def test_explain_unknown_algorithm(self) -> None:
        from problemsolving.mcp.handlers import handle_explain_algorithm

        result = handle_explain_algorithm({"algorithm": "nonexistent"})
        assert result["found"] is False


class TestMCPServerManifest:
    """Test MCP server manifest/schema."""

    def test_manifest_has_all_tools(self) -> None:
        from problemsolving.mcp.server import get_manifest

        manifest = get_manifest()
        tool_names = {t["name"] for t in manifest["tools"]}
        assert "solve" in tool_names
        assert "select_algorithm" in tool_names
        assert "list_engines" in tool_names
        assert "verify" in tool_names
        assert "explain_algorithm" in tool_names

    def test_tools_have_schemas(self) -> None:
        from problemsolving.mcp.server import get_manifest

        manifest = get_manifest()
        for tool in manifest["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_dispatch_routes_correctly(self) -> None:
        from problemsolving.mcp.server import dispatch_tool

        result = dispatch_tool("list_engines", {})
        assert "engines" in result

    def test_dispatch_unknown_tool_errors(self) -> None:
        from problemsolving.mcp.server import dispatch_tool

        result = dispatch_tool("unknown_tool", {})
        assert "error" in result
