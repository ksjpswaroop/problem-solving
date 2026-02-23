"""Phase 6: Logic programming and rule engine tests."""


class TestPrologLite:
    """Prolog-lite logic programming tests."""

    def test_simple_fact_query(self) -> None:
        from problemsolving.logic.prolog_lite import PrologEngine

        engine = PrologEngine()
        engine.add_fact("parent", ("tom", "bob"))
        engine.add_fact("parent", ("tom", "liz"))

        results = engine.query("parent", ("tom", "?X"))
        assert len(results) == 2
        values = {r["X"] for r in results}
        assert values == {"bob", "liz"}

    def test_rule_based_query(self) -> None:
        from problemsolving.logic.prolog_lite import PrologEngine

        engine = PrologEngine()
        engine.add_fact("parent", ("tom", "bob"))
        engine.add_fact("parent", ("bob", "ann"))

        # grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        engine.add_rule(
            "grandparent",
            ("?X", "?Z"),
            [("parent", ("?X", "?Y")), ("parent", ("?Y", "?Z"))],
        )

        results = engine.query("grandparent", ("tom", "?Z"))
        assert len(results) == 1
        assert results[0]["Z"] == "ann"

    def test_no_match_returns_empty(self) -> None:
        from problemsolving.logic.prolog_lite import PrologEngine

        engine = PrologEngine()
        engine.add_fact("parent", ("tom", "bob"))

        results = engine.query("parent", ("bob", "?X"))
        assert results == []

    def test_multiple_variable_binding(self) -> None:
        from problemsolving.logic.prolog_lite import PrologEngine

        engine = PrologEngine()
        engine.add_fact("edge", ("a", "b"))
        engine.add_fact("edge", ("b", "c"))
        engine.add_fact("edge", ("c", "d"))

        results = engine.query("edge", ("?X", "?Y"))
        assert len(results) == 3

    def test_ground_query(self) -> None:
        from problemsolving.logic.prolog_lite import PrologEngine

        engine = PrologEngine()
        engine.add_fact("likes", ("alice", "python"))

        assert engine.query("likes", ("alice", "python")) != []
        assert engine.query("likes", ("alice", "java")) == []


class TestRuleEngine:
    """Forward-chaining rule engine tests."""

    def test_simple_rule_fires(self) -> None:
        from problemsolving.logic.rule_engine import RuleEngine

        engine = RuleEngine()
        engine.add_fact("temperature", 105)

        engine.add_rule(
            name="fever_check",
            condition=lambda facts: facts.get("temperature", 0) > 100,
            action=lambda facts: facts.update({"has_fever": True}),
        )

        engine.run()
        assert engine.facts["has_fever"] is True

    def test_chained_rules(self) -> None:
        from problemsolving.logic.rule_engine import RuleEngine

        engine = RuleEngine()
        engine.add_fact("temperature", 105)
        engine.add_fact("cough", True)

        engine.add_rule(
            name="fever_check",
            condition=lambda facts: facts.get("temperature", 0) > 100,
            action=lambda facts: facts.update({"has_fever": True}),
        )
        engine.add_rule(
            name="flu_diagnosis",
            condition=lambda facts: facts.get("has_fever") and facts.get("cough"),
            action=lambda facts: facts.update({"diagnosis": "flu"}),
        )

        engine.run()
        assert engine.facts["diagnosis"] == "flu"

    def test_no_matching_rule(self) -> None:
        from problemsolving.logic.rule_engine import RuleEngine

        engine = RuleEngine()
        engine.add_fact("temperature", 98)

        engine.add_rule(
            name="fever_check",
            condition=lambda facts: facts.get("temperature", 0) > 100,
            action=lambda facts: facts.update({"has_fever": True}),
        )

        engine.run()
        assert "has_fever" not in engine.facts

    def test_rule_fires_only_once(self) -> None:
        from problemsolving.logic.rule_engine import RuleEngine

        counter = {"count": 0}
        engine = RuleEngine()
        engine.add_fact("x", 1)

        def increment(facts: dict) -> None:
            counter["count"] += 1
            facts["incremented"] = True

        engine.add_rule(
            name="increment_once",
            condition=lambda facts: facts.get("x") == 1,
            action=increment,
        )

        engine.run()
        assert counter["count"] == 1

    def test_returns_trace(self) -> None:
        from problemsolving.logic.rule_engine import RuleEngine

        engine = RuleEngine()
        engine.add_fact("a", True)

        engine.add_rule(
            name="rule1",
            condition=lambda facts: facts.get("a"),
            action=lambda facts: facts.update({"b": True}),
        )

        trace = engine.run()
        assert len(trace) == 1
        assert trace[0]["rule"] == "rule1"
