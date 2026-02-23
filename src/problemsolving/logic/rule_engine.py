"""Forward-chaining rule engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Rule:
    """A production rule: if condition then action."""

    name: str
    condition: Callable[[dict[str, Any]], bool]
    action: Callable[[dict[str, Any]], None]
    fired: bool = False


class RuleEngine:
    """Forward-chaining rule engine.

    Repeatedly evaluates rules against working memory (facts)
    until no new rules fire (fixed-point).
    """

    def __init__(self) -> None:
        self.facts: dict[str, Any] = {}
        self._rules: list[Rule] = []

    def add_fact(self, key: str, value: Any) -> None:
        """Add a fact to working memory."""
        self.facts[key] = value

    def add_rule(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        action: Callable[[dict[str, Any]], None],
    ) -> None:
        """Add a production rule."""
        self._rules.append(Rule(name=name, condition=condition, action=action))

    def run(self, max_iterations: int = 100) -> list[dict[str, Any]]:
        """Run the engine until quiescence. Returns trace of fired rules."""
        trace: list[dict[str, Any]] = []

        for _ in range(max_iterations):
            fired_any = False

            for rule in self._rules:
                if rule.fired:
                    continue

                try:
                    if rule.condition(self.facts):
                        snapshot_before = dict(self.facts)
                        rule.action(self.facts)
                        rule.fired = True
                        fired_any = True

                        trace.append({
                            "rule": rule.name,
                            "facts_before": snapshot_before,
                            "facts_after": dict(self.facts),
                        })
                except Exception:
                    continue

            if not fired_any:
                break

        return trace


def rule_engine_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for rule engine.

    For protocol use, rules are defined as simple threshold checks.
    """
    engine = RuleEngine()

    for key, value in input_data.get("facts", {}).items():
        engine.add_fact(key, value)

    for rule_def in input_data.get("rules", []):
        name = rule_def["name"]
        # Simple threshold rules: if fact > threshold then set conclusion
        fact_key = rule_def.get("if_fact", "")
        threshold = rule_def.get("threshold", 0)
        conclusion_key = rule_def.get("then_set", "")
        conclusion_val = rule_def.get("then_value", True)

        def _make_condition(
            fk: str, t: Any
        ) -> Callable[[dict[str, Any]], bool]:
            return lambda f: f.get(fk, 0) > t

        def _make_action(
            ck: str, cv: Any
        ) -> Callable[[dict[str, Any]], None]:
            return lambda f: f.update({ck: cv})

        engine.add_rule(
            name=name,
            condition=_make_condition(fact_key, threshold),
            action=_make_action(conclusion_key, conclusion_val),
        )

    trace = engine.run()
    return {
        "facts": engine.facts,
        "rules_fired": [t["rule"] for t in trace],
        "trace": trace,
    }
