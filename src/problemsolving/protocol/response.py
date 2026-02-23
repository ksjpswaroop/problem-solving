"""Solver protocol response model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolverResponse:
    """Response from a solver engine."""

    id: str
    status: str  # "success" or "error"
    engine: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    proof_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        request_id: str,
        engine: str,
        result: dict[str, Any],
        proof_trace: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SolverResponse:
        """Create a success response."""
        return cls(
            id=request_id,
            status="success",
            engine=engine,
            result=result,
            error=None,
            proof_trace=proof_trace or [],
            metadata=metadata or {},
        )

    @classmethod
    def make_error(
        cls,
        request_id: str,
        engine: str,
        code: str,
        message: str,
        suggestion: str | None = None,
    ) -> SolverResponse:
        """Create an error response."""
        err: dict[str, Any] = {"code": code, "message": message}
        if suggestion:
            err["suggestion"] = suggestion
        return cls(
            id=request_id,
            status="error",
            engine=engine,
            error=err,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to protocol-format dictionary."""
        d: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "engine": self.engine,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        if self.proof_trace:
            d["proof_trace"] = self.proof_trace
        if self.metadata:
            d["metadata"] = self.metadata
        return d
