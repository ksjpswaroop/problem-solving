"""Solver protocol request model."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolverRequest:
    """A request to solve a problem via a specific engine."""

    engine: str
    operation: str
    input_data: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = f"req_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to protocol-format dictionary."""
        return {
            "id": self.id,
            "engine": self.engine,
            "operation": self.operation,
            "input": self.input_data,
            "config": self.config,
        }
