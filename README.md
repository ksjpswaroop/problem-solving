# ProblemSolving: Universal AI Reasoning Toolkit

ProblemSolving is a Python toolkit that combines classical AI algorithms, symbolic reasoning, and MCP-compatible tool interfaces so agents and applications can use deterministic solvers instead of relying on free-form reasoning alone.

This repository includes:

- A unified `solve(...)` API across all engines
- Auto-selection of algorithms with `select_algorithm(...)`
- An MCP tool surface for agent/tool integrations
- Runnable examples (real-world scenarios, LLM connectivity, MCP usage)

---

## Current Implementation Status

The library currently ships **14 implemented engines**:

### Search / Pathfinding

- `bfs` — Breadth-First Search
- `dfs` — Depth-First Search
- `ucs` — Uniform Cost Search
- `astar` — A* Search
- `greedy` — Greedy Best-First Search

### Optimization

- `gradient_descent` — Gradient-based minimization
- `genetic_algorithm` — Evolutionary optimization
- `simulated_annealing` — Metaheuristic optimization

### Constraints and Satisfiability

- `dpll_sat` — SAT solving with DPLL
- `csp_backtracking` — CSP backtracking with MRV

### Symbolic Reasoning

- `cas` — Computer Algebra System (solve, solve_system, differentiate, integrate, simplify)
- `smt_lite` — SMT-lite linear integer constraint solving

### Logic

- `prolog_lite` — Facts/rules/query logic programming
- `rule_engine` — Forward-chaining production rules

---

## Quick Start

### 1) Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[cas,smt,sat]"
```

### 2) Solve a problem

```python
from problemsolving import solve

response = solve(
    engine="bfs",
    input_data={
        "graph": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
        "start": "A",
        "goal": "D",
    },
)

print(response.status)          # success
print(response.result["path"])  # ['A', 'B', 'D'] or ['A', 'C', 'D']
```

### 3) Use auto-selection

```python
from problemsolving import solve

response = solve(
    engine="auto",
    input_data={
        "graph": {
            "A": [("B", 1), ("C", 4)],
            "B": [("D", 5)],
            "C": [("D", 1)],
            "D": [],
        },
        "start": "A",
        "goal": "D",
        "heuristic": {"A": 5, "B": 4, "C": 1, "D": 0},
    },
    problem_type="pathfinding",
    features={"weighted": True, "has_heuristic": True, "needs_optimal": True},
)

print(response.engine)  # astar
print(response.result)  # {'path': [...], 'cost': ..., 'nodes_explored': ...}
```

---

## Documentation

- `LIBRARY_FEATURES.md` — Feature-by-feature documentation with examples for all implemented engines
- `spec/API_REFERENCE.md` — Detailed API signatures and payload schemas
- `spec/PROTOCOL_SPEC.md` — Request/response protocol contract
- `ARCHITECTURE.md` — System architecture and MCP design
- `algorithms.md` — Broader algorithm catalog and pseudocode
- `symbolic_reasoning.md` — Symbolic reasoning strategy and integration patterns

---

## Examples

The `examples/` folder now includes practical scripts:

- `examples/python/real_world_logistics_planning.py`
  - Route planning for delivery logistics using auto-selection and search engines.
- `examples/python/constraint_based_scheduling.py`
  - Staff scheduling with CSP and a SAT consistency check.
- `examples/python/llm_verified_reasoning.py`
  - OpenAI-compatible LLM proposal + deterministic solver verification loop.
- `examples/python/mcp_usage.py`
  - MCP manifest/tool usage with `dispatch_tool(...)` (`solve`, `select_algorithm`, `verify`, etc.).

Run examples from repo root:

```bash
PYTHONPATH=src python examples/python/real_world_logistics_planning.py
PYTHONPATH=src python examples/python/constraint_based_scheduling.py
PYTHONPATH=src python examples/python/llm_verified_reasoning.py
PYTHONPATH=src python examples/python/mcp_usage.py
```

See `examples/README.md` for more details.

---

## LLM Connectivity

`examples/python/llm_verified_reasoning.py` supports OpenAI-compatible endpoints using:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)

If no API key is set, the example falls back to a deterministic local candidate plan so the flow is still runnable.

---

## MCP Integration

The library exposes MCP-style tools through `problemsolving.mcp.server`:

- `solve`
- `select_algorithm`
- `list_engines`
- `verify`
- `explain_algorithm`

Use:

```python
from problemsolving.mcp.server import get_manifest, dispatch_tool
```

to discover tools and execute calls.

---

## Next Steps TODO

- [ ] Add knowledge metadata files for `prolog_lite` and `rule_engine` in `knowledge/algorithms/`
- [ ] Add notebook examples for interactive workflows (`examples/notebooks/`)
- [ ] Add benchmark scripts for search and optimization scalability
- [ ] Expand MCP examples to include full JSON-RPC transport adapters
- [ ] Add additional real-world solver recipes (vehicle routing, timetable generation, compliance checks)
- [ ] Publish CI artifacts for example outputs and reproducibility checks

---

## Development

```bash
pip install -e ".[dev,cas,smt,sat]"
pytest
```
