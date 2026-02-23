"""
File Purpose Summary:
This example demonstrates LLM connectivity for planning plus deterministic
verification using ProblemSolving engines. The LLM proposes a structured route plan,
and the solver executes/validates it so final decisions are grounded in algorithmic
results rather than unverified text output.

Responsibilities & Behavior:
- Connects to an OpenAI-compatible chat completion endpoint (optional).
- Requests a JSON route-planning proposal from the LLM.
- Falls back to a local deterministic proposal when API credentials are absent.
- Executes the proposed plan through solve(...).
- Performs a simple post-check on destination and cost budget.

Context & Architectural Notes:
- Illustrates a common "LLM propose, symbolic verify" production pattern.
- Uses only standard library networking and JSON handling (no extra dependencies).
- Expects OPENAI-compatible env vars but remains runnable without external access.

Created At (ISO): 2026-02-23T18:27:40Z
Created At (PT): 2026-02-23 10:27:40 PST
Updated At (ISO): 2026-02-23T18:27:40Z
Updated At (PT): 2026-02-23 10:27:40 PST
Updated By: AI
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Allow direct execution from repository root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from problemsolving import solve


GRAPH = {
    "warehouse": [("north_hub", 6), ("east_hub", 4), ("south_hub", 7)],
    "north_hub": [("customer_a", 2), ("customer_b", 5)],
    "east_hub": [("customer_b", 3), ("customer_c", 6)],
    "south_hub": [("customer_c", 2), ("customer_d", 5)],
    "customer_a": [("destination", 8)],
    "customer_b": [("destination", 4)],
    "customer_c": [("destination", 3)],
    "customer_d": [("destination", 1)],
    "destination": [],
}

HEURISTIC = {
    "warehouse": 6,
    "north_hub": 5,
    "east_hub": 4,
    "south_hub": 3,
    "customer_a": 8,
    "customer_b": 3,
    "customer_c": 2,
    "customer_d": 1,
    "destination": 0,
}

FALLBACK_PLAN = {
    "engine": "auto",
    "problem_type": "pathfinding",
    "features": {"weighted": True, "has_heuristic": True, "needs_optimal": True},
    "input": {"graph": GRAPH, "start": "warehouse", "goal": "destination", "heuristic": HEURISTIC},
}


def query_openai_compatible(prompt: str) -> str | None:
    """Return model text response from an OpenAI-compatible endpoint, or None."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON. Choose a robust pathfinding plan with an engine, "
                    "problem_type, features, and input payload."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    choices = body.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    return content if isinstance(content, str) else None


def parse_plan(model_output: str | None) -> dict[str, Any]:
    """Parse model output into a plan dict. Falls back on parse failures."""
    if not model_output:
        return FALLBACK_PLAN

    candidate = model_output.strip()
    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return FALLBACK_PLAN

    candidate = candidate[first_brace : last_brace + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return FALLBACK_PLAN

    if not isinstance(parsed, dict):
        return FALLBACK_PLAN

    engine = parsed.get("engine")
    input_payload = parsed.get("input")
    if not isinstance(engine, str) or not isinstance(input_payload, dict):
        return FALLBACK_PLAN

    return {
        "engine": parsed.get("engine", "auto"),
        "problem_type": parsed.get("problem_type", "pathfinding"),
        "features": parsed.get(
            "features",
            {"weighted": True, "has_heuristic": True, "needs_optimal": True},
        ),
        "input": parsed["input"],
    }


def run() -> None:
    """Run LLM proposal + deterministic verification workflow."""
    prompt = (
        "Given this weighted graph route-planning problem, return JSON with keys "
        "engine, problem_type, features, input. "
        f"Graph={GRAPH}; heuristic={HEURISTIC}; start='warehouse'; goal='destination'."
    )

    model_output = query_openai_compatible(prompt)
    plan = parse_plan(model_output)

    response = solve(
        engine=plan["engine"],
        input_data=plan["input"],
        problem_type=plan.get("problem_type"),
        features=plan.get("features"),
    )

    if response.status != "success" or response.result is None:
        raise RuntimeError(f"Solver verification failed: {response.error}")

    path = response.result.get("path", [])
    cost = response.result.get("cost")
    verified_goal = bool(path) and path[-1] == "destination"
    verified_cost = isinstance(cost, (int, float)) and cost <= 15

    print("Plan source:", "llm" if model_output else "fallback")
    print("Selected engine:", response.engine)
    print("Path:", path)
    print("Cost:", cost)
    print("Verified destination:", verified_goal)
    print("Verified budget (<= 15):", verified_cost)


if __name__ == "__main__":
    run()
