"""MCP server: manifest, tool dispatch, and JSON-RPC handler."""

from __future__ import annotations

from typing import Any

from problemsolving.mcp.handlers import (
    handle_explain_algorithm,
    handle_list_engines,
    handle_select_algorithm,
    handle_solve,
    handle_verify,
)

_TOOLS = [
    {
        "name": "solve",
        "description": (
            "Solve a problem using a specified engine or auto-select. "
            "Supports pathfinding, optimization, SAT, CSP, symbolic math, "
            "logic programming, and more."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "engine": {
                    "type": "string",
                    "description": "Engine name or 'auto' for automatic selection",
                },
                "input": {
                    "type": "object",
                    "description": "Engine-specific input data",
                },
                "problem_type": {
                    "type": "string",
                    "description": "Problem type for auto-selection",
                },
                "features": {
                    "type": "object",
                    "description": "Problem features for auto-selection",
                },
            },
            "required": ["engine", "input"],
        },
    },
    {
        "name": "select_algorithm",
        "description": (
            "Select the best algorithm for a given problem type and features. "
            "Returns the recommended algorithm with reasoning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problem_type": {
                    "type": "string",
                    "description": "Type of problem (pathfinding, optimization, etc.)",
                },
                "features": {
                    "type": "object",
                    "description": "Problem features (weighted, differentiable, etc.)",
                },
            },
            "required": ["problem_type"],
        },
    },
    {
        "name": "list_engines",
        "description": "List all available solver engines, optionally filtered by tag.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Optional tag to filter engines",
                },
            },
        },
    },
    {
        "name": "verify",
        "description": (
            "Verify a solution by running an engine and comparing output "
            "against expected values."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "engine": {"type": "string"},
                "input": {"type": "object"},
                "expected": {
                    "type": "object",
                    "description": "Expected result fields to verify",
                },
            },
            "required": ["engine", "input", "expected"],
        },
    },
    {
        "name": "explain_algorithm",
        "description": (
            "Get detailed metadata about an algorithm: description, "
            "when to use, complexity, etc."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm ID (e.g. 'bfs', 'astar', 'dpll_sat')",
                },
            },
            "required": ["algorithm"],
        },
    },
]

_DISPATCH: dict[str, Any] = {
    "solve": handle_solve,
    "select_algorithm": handle_select_algorithm,
    "list_engines": handle_list_engines,
    "verify": handle_verify,
    "explain_algorithm": handle_explain_algorithm,
}


def get_manifest() -> dict[str, Any]:
    """Return the MCP server manifest with all tool definitions."""
    return {
        "name": "problemsolving",
        "version": "0.1.0",
        "description": "Universal AI reasoning toolkit — classical algorithms as MCP tools",
        "tools": _TOOLS,
    }


def dispatch_tool(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler."""
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return handler(params)  # type: ignore[no-any-return]
