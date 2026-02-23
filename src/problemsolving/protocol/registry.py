"""Engine registry — register and discover solver engines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from problemsolving.protocol.request import SolverRequest
from problemsolving.protocol.response import SolverResponse


@dataclass
class RegisteredEngine:
    """A registered solver engine."""

    name: str
    solve_fn: Callable[..., dict[str, Any]]
    tags: list[str] = field(default_factory=list)

    def handle(self, request: SolverRequest) -> SolverResponse:
        """Handle a solver request and return a protocol response."""
        try:
            result = self.solve_fn(request.input_data)
            return SolverResponse.success(
                request_id=request.id or "",
                engine=self.name,
                result=result,
                metadata=result.pop("_metadata", {}),
                proof_trace=result.pop("_proof_trace", []),
            )
        except Exception as e:
            return SolverResponse.make_error(
                request_id=request.id or "",
                engine=self.name,
                code="SOLVER_ERROR",
                message=str(e),
            )


class EngineRegistry:
    """Registry of available solver engines."""

    def __init__(self) -> None:
        self._engines: dict[str, RegisteredEngine] = {}

    def register(
        self,
        name: str,
        solve_fn: Callable[..., dict[str, Any]],
        tags: list[str] | None = None,
    ) -> None:
        """Register a solver engine."""
        self._engines[name] = RegisteredEngine(
            name=name, solve_fn=solve_fn, tags=tags or []
        )

    def get(self, name: str) -> RegisteredEngine | None:
        """Get engine by name, or None if not found."""
        return self._engines.get(name)

    def list_engines(self, tag: str | None = None) -> list[str]:
        """List registered engine names, optionally filtered by tag."""
        if tag is None:
            return list(self._engines.keys())
        return [
            name
            for name, engine in self._engines.items()
            if tag in engine.tags
        ]


# Global default registry, populated as engines are imported
_default_registry: EngineRegistry | None = None


def get_default_registry() -> EngineRegistry:
    """Get the default global engine registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = EngineRegistry()
        _register_builtin_engines(_default_registry)
    return _default_registry


def _register_builtin_engines(registry: EngineRegistry) -> None:
    """Register all built-in engines."""
    from problemsolving.search.astar import astar_solve_from_dict
    from problemsolving.search.bfs import bfs_solve_from_dict
    from problemsolving.search.dfs import dfs_solve_from_dict
    from problemsolving.search.greedy import greedy_solve_from_dict
    from problemsolving.search.ucs import ucs_solve_from_dict

    registry.register("bfs", solve_fn=bfs_solve_from_dict, tags=["search", "pathfinding"])
    registry.register("dfs", solve_fn=dfs_solve_from_dict, tags=["search", "pathfinding"])
    registry.register(
        "ucs", solve_fn=ucs_solve_from_dict, tags=["search", "pathfinding", "optimal"]
    )
    registry.register(
        "astar",
        solve_fn=astar_solve_from_dict,
        tags=["search", "pathfinding", "optimal", "heuristic"],
    )
    registry.register(
        "greedy", solve_fn=greedy_solve_from_dict, tags=["search", "pathfinding", "heuristic"]
    )

    # Optimization engines
    from problemsolving.optimization.genetic import ga_solve_from_dict
    from problemsolving.optimization.gradient_descent import gd_solve_from_dict
    from problemsolving.optimization.simulated_annealing import sa_solve_from_dict

    registry.register(
        "gradient_descent", solve_fn=gd_solve_from_dict, tags=["optimization", "gradient"]
    )
    registry.register(
        "genetic_algorithm", solve_fn=ga_solve_from_dict, tags=["optimization", "evolutionary"]
    )
    registry.register(
        "simulated_annealing",
        solve_fn=sa_solve_from_dict,
        tags=["optimization", "metaheuristic"],
    )

    # CSP / SAT engines
    from problemsolving.csp.backtracking import csp_solve_from_dict
    from problemsolving.csp.sat import dpll_solve_from_dict

    registry.register("dpll_sat", solve_fn=dpll_solve_from_dict, tags=["sat", "logic", "csp"])
    registry.register(
        "csp_backtracking", solve_fn=csp_solve_from_dict, tags=["csp", "constraint"]
    )

    # Symbolic engines
    from problemsolving.symbolic.cas import cas_solve_from_dict
    from problemsolving.symbolic.smt_lite import smt_solve_from_dict

    registry.register(
        "cas", solve_fn=cas_solve_from_dict, tags=["symbolic", "algebra", "calculus"]
    )
    registry.register(
        "smt_lite",
        solve_fn=smt_solve_from_dict,
        tags=["smt", "arithmetic", "constraint"],
    )

    # Logic engines
    from problemsolving.logic.prolog_lite import prolog_solve_from_dict
    from problemsolving.logic.rule_engine import rule_engine_solve_from_dict

    registry.register(
        "prolog_lite",
        solve_fn=prolog_solve_from_dict,
        tags=["logic", "symbolic"],
    )
    registry.register(
        "rule_engine",
        solve_fn=rule_engine_solve_from_dict,
        tags=["logic", "rule_based"],
    )
