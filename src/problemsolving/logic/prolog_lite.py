"""Prolog-lite: simple logic programming with facts, rules, and queries."""

from __future__ import annotations

from typing import Any


class PrologEngine:
    """A minimal Prolog-style logic engine supporting facts, rules, and queries."""

    def __init__(self) -> None:
        self._facts: list[tuple[str, tuple[str, ...]]] = []
        self._rules: list[tuple[str, tuple[str, ...], list[tuple[str, tuple[str, ...]]]]] = []

    def add_fact(self, predicate: str, args: tuple[str, ...]) -> None:
        """Add a ground fact. e.g. add_fact("parent", ("tom", "bob"))."""
        self._facts.append((predicate, args))

    def add_rule(
        self,
        head_predicate: str,
        head_args: tuple[str, ...],
        body: list[tuple[str, tuple[str, ...]]],
    ) -> None:
        """Add a rule. Variables start with '?'.

        e.g. add_rule("grandparent", ("?X", "?Z"),
                       [("parent", ("?X", "?Y")), ("parent", ("?Y", "?Z"))])
        """
        self._rules.append((head_predicate, head_args, body))

    def query(self, predicate: str, args: tuple[str, ...]) -> list[dict[str, str]]:
        """Query for matching bindings. Variables start with '?'.

        Returns list of variable binding dicts (without '?' prefix in keys).
        """
        results: list[dict[str, str]] = []
        bindings: dict[str, str] = {}

        self._solve_goal(predicate, args, bindings, results)
        return results

    def _solve_goal(
        self,
        predicate: str,
        args: tuple[str, ...],
        bindings: dict[str, str],
        results: list[dict[str, str]],
    ) -> None:
        """Try to satisfy a single goal."""
        # Try matching against facts
        for fact_pred, fact_args in self._facts:
            if fact_pred != predicate:
                continue
            if len(fact_args) != len(args):
                continue

            new_bindings = dict(bindings)
            if self._unify_args(args, fact_args, new_bindings):
                # Extract variable bindings for result
                result = {
                    k.lstrip("?"): v for k, v in new_bindings.items() if k.startswith("?")
                }
                if result not in results:
                    results.append(result)

        # Try matching against rules
        for rule_pred, rule_head_args, rule_body in self._rules:
            if rule_pred != predicate:
                continue
            if len(rule_head_args) != len(args):
                continue

            new_bindings = dict(bindings)
            if self._unify_args(args, rule_head_args, new_bindings):
                # Try to satisfy all body goals
                self._solve_body(rule_body, new_bindings, results)

    def _solve_body(
        self,
        body: list[tuple[str, tuple[str, ...]]],
        bindings: dict[str, str],
        results: list[dict[str, str]],
    ) -> None:
        """Satisfy all goals in a rule body."""
        if not body:
            result = {
                k.lstrip("?"): v for k, v in bindings.items() if k.startswith("?")
            }
            if result not in results:
                results.append(result)
            return

        pred, args = body[0]
        rest = body[1:]

        # Substitute current bindings into args
        resolved_args = tuple(self._resolve(a, bindings) for a in args)

        # Find all matching facts for this subgoal
        for fact_pred, fact_args in self._facts:
            if fact_pred != pred or len(fact_args) != len(resolved_args):
                continue

            new_bindings = dict(bindings)
            if self._unify_args(resolved_args, fact_args, new_bindings):
                self._solve_body(rest, new_bindings, results)

    def _unify_args(
        self,
        query_args: tuple[str, ...],
        fact_args: tuple[str, ...],
        bindings: dict[str, str],
    ) -> bool:
        """Try to unify query args with fact args, updating bindings."""
        for q, f in zip(query_args, fact_args):
            q_resolved = self._resolve(q, bindings)
            f_resolved = self._resolve(f, bindings)

            if q_resolved.startswith("?"):
                bindings[q_resolved] = f_resolved
            elif f_resolved.startswith("?"):
                bindings[f_resolved] = q_resolved
            elif q_resolved != f_resolved:
                return False
        return True

    def _resolve(self, term: str, bindings: dict[str, str]) -> str:
        """Resolve a term through bindings chain."""
        seen: set[str] = set()
        while term.startswith("?") and term in bindings and term not in seen:
            seen.add(term)
            term = bindings[term]
        return term


def prolog_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Protocol adapter for Prolog-lite engine."""
    engine = PrologEngine()

    for fact in input_data.get("facts", []):
        engine.add_fact(fact["predicate"], tuple(fact["args"]))

    for rule in input_data.get("rules", []):
        body = [(g["predicate"], tuple(g["args"])) for g in rule["body"]]
        engine.add_rule(rule["head_predicate"], tuple(rule["head_args"]), body)

    query = input_data["query"]
    results = engine.query(query["predicate"], tuple(query["args"]))
    return {"results": results, "count": len(results)}
