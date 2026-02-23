"""ProblemSolving: Universal AI reasoning toolkit.

Classical algorithms + symbolic engines as Chain-of-Thought replacement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from problemsolving.protocol.response import SolverResponse

__version__ = "0.1.0"


def solve(
    engine: str,
    input_data: dict[str, Any],
    operation: str = "solve",
    config: dict[str, Any] | None = None,
    problem_type: str | None = None,
    features: dict[str, Any] | None = None,
) -> SolverResponse:
    """Unified solve API: route to any registered engine.

    If engine="auto", uses the algorithm selector to pick the best engine.
    """
    from problemsolving.protocol.registry import get_default_registry
    from problemsolving.protocol.request import SolverRequest
    from problemsolving.protocol.response import SolverResponse

    if engine == "auto":
        from problemsolving.selector import select_algorithm

        selection = select_algorithm(
            problem_type=problem_type or "pathfinding",
            features=features,
        )
        engine = selection["algorithm"]

    registry = get_default_registry()
    registered = registry.get(engine)

    if registered is None:
        return SolverResponse.make_error(
            request_id="",
            engine=engine,
            code="UNKNOWN_ENGINE",
            message=f"Engine '{engine}' not found. Available: {registry.list_engines()}",
        )

    request = SolverRequest(
        engine=engine,
        operation=operation,
        input_data=input_data,
        config=config or {},
    )
    return registered.handle(request)


def list_engines(tag: str | None = None) -> list[str]:
    """List all registered engine names, optionally filtered by tag."""
    from problemsolving.protocol.registry import get_default_registry

    return get_default_registry().list_engines(tag=tag)
