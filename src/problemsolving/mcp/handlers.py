"""MCP tool handlers — business logic for each MCP tool."""

from __future__ import annotations

import json
import os
from typing import Any


def handle_solve(params: dict[str, Any]) -> dict[str, Any]:
    """Handle the 'solve' MCP tool call."""
    from problemsolving import solve

    engine = params.get("engine", "auto")
    input_data = params.get("input", {})
    problem_type = params.get("problem_type")
    features = params.get("features")

    response = solve(
        engine=engine,
        input_data=input_data,
        problem_type=problem_type,
        features=features,
    )
    return response.to_dict()


def handle_select_algorithm(params: dict[str, Any]) -> dict[str, Any]:
    """Handle the 'select_algorithm' MCP tool call."""
    from problemsolving.selector import select_algorithm

    return select_algorithm(
        problem_type=params.get("problem_type", ""),
        features=params.get("features"),
    )


def handle_list_engines(params: dict[str, Any]) -> dict[str, Any]:
    """Handle the 'list_engines' MCP tool call."""
    from problemsolving import list_engines

    tag = params.get("tag")
    return {"engines": list_engines(tag=tag)}


def handle_verify(params: dict[str, Any]) -> dict[str, Any]:
    """Handle the 'verify' MCP tool call.

    Runs the engine and compares output to expected values.
    """
    from problemsolving import solve

    engine = params.get("engine", "auto")
    input_data = params.get("input", {})
    expected = params.get("expected", {})

    response = solve(engine=engine, input_data=input_data)

    if response.status != "success" or response.result is None:
        return {
            "verified": False,
            "reason": "Engine returned error or no result",
            "actual": response.to_dict(),
        }

    # Compare expected fields against actual
    mismatches: list[str] = []
    for key, expected_val in expected.items():
        actual_val = response.result.get(key)
        if actual_val != expected_val:
            mismatches.append(
                f"{key}: expected {expected_val}, got {actual_val}"
            )

    return {
        "verified": len(mismatches) == 0,
        "mismatches": mismatches,
        "actual": response.result,
    }


def handle_explain_algorithm(params: dict[str, Any]) -> dict[str, Any]:
    """Handle the 'explain_algorithm' MCP tool call.

    Returns metadata about an algorithm from the knowledge base.
    """
    algo_id = params.get("algorithm", "")

    # Try to load from knowledge base
    knowledge_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "knowledge",
        "algorithms",
    )
    json_path = os.path.join(knowledge_dir, f"{algo_id}.json")

    if os.path.exists(json_path):
        with open(json_path) as f:
            metadata = json.load(f)
        return {
            "found": True,
            "id": metadata.get("id", algo_id),
            "name": metadata.get("name", algo_id),
            "description": metadata.get("description", ""),
            "when_to_use": metadata.get("when_to_use", []),
            "when_not_to_use": metadata.get("when_not_to_use", []),
            "complexity": metadata.get("complexity", {}),
            "tags": metadata.get("tags", []),
        }

    return {"found": False, "id": algo_id, "message": f"No metadata found for '{algo_id}'"}
