"""
File Purpose Summary:
This example demonstrates how to use the library's MCP-facing interface directly
from Python. It showcases manifest discovery and dispatching tool calls that mirror
agent integrations built on the Model Context Protocol tool contract.

Responsibilities & Behavior:
- Reads the MCP manifest to inspect available tools.
- Calls list_engines and select_algorithm via dispatch_tool(...).
- Solves a pathfinding task through the MCP "solve" tool.
- Verifies expected output through the MCP "verify" tool.
- Fetches algorithm metadata through "explain_algorithm".

Context & Architectural Notes:
- Uses problemsolving.mcp.server as an in-process adapter for MCP semantics.
- Demonstrates tool payload structure exactly as MCP clients would send.
- Useful as a baseline for wiring external JSON-RPC transport layers.

Created At (ISO): 2026-02-23T18:27:40Z
Created At (PT): 2026-02-23 10:27:40 PST
Updated At (ISO): 2026-02-23T18:27:40Z
Updated At (PT): 2026-02-23 10:27:40 PST
Updated By: AI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow direct execution from repository root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from problemsolving.mcp.server import dispatch_tool, get_manifest


def run() -> None:
    """Run a small sequence of MCP tool calls."""
    manifest = get_manifest()
    tool_names = [tool["name"] for tool in manifest["tools"]]
    print("Manifest tools:", tool_names)

    print("\nlist_engines(search):")
    print(json.dumps(dispatch_tool("list_engines", {"tag": "search"}), indent=2))

    print("\nselect_algorithm(pathfinding):")
    selection = dispatch_tool(
        "select_algorithm",
        {
            "problem_type": "pathfinding",
            "features": {"weighted": True, "has_heuristic": True, "needs_optimal": True},
        },
    )
    print(json.dumps(selection, indent=2))

    solve_input = {
        "graph": {"A": [["B", 1], ["C", 4]], "B": [["D", 5]], "C": [["D", 1]], "D": []},
        "start": "A",
        "goal": "D",
        "heuristic": {"A": 5, "B": 4, "C": 1, "D": 0},
    }

    print("\nsolve(astar):")
    solved = dispatch_tool("solve", {"engine": "astar", "input": solve_input})
    print(json.dumps(solved, indent=2))

    print("\nverify(expected path and cost):")
    verified = dispatch_tool(
        "verify",
        {
            "engine": "astar",
            "input": solve_input,
            "expected": {"path": ["A", "C", "D"], "cost": 5},
        },
    )
    print(json.dumps(verified, indent=2))

    print("\nexplain_algorithm(bfs):")
    explanation = dispatch_tool("explain_algorithm", {"algorithm": "bfs"})
    print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    run()
