# ProblemSolving Library Features (Implemented) with Examples

This document covers the **currently implemented** capabilities in the library and shows practical usage snippets for each feature.

> For full API signatures, see `spec/API_REFERENCE.md`.

---

## 1. Unified APIs

### `solve(...)`

Single entry point for all engines.

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

assert response.status == "success"
print(response.engine)  # bfs
print(response.result)  # {'path': [...], 'nodes_explored': ...}
```

### `list_engines(...)`

Discover available engines globally or by tag.

```python
from problemsolving import list_engines

print(list_engines())
print(list_engines(tag="search"))
print(list_engines(tag="optimization"))
```

### `engine="auto"` with selector

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
```

---

## 2. Search and Pathfinding Engines

### 2.1 BFS (`bfs`)
Use for shortest-hop paths on unweighted graphs.

```python
from problemsolving import solve

solve(
    engine="bfs",
    input_data={
        "graph": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
        "start": "A",
        "goal": "D",
    },
)
```

### 2.2 DFS (`dfs`)
Use for depth-first exploration and quick path discovery.

```python
from problemsolving import solve

solve(
    engine="dfs",
    input_data={
        "graph": {"A": ["B", "C"], "B": ["D"], "C": [], "D": []},
        "start": "A",
        "goal": "D",
        "max_depth": 10,
    },
)
```

### 2.3 Uniform Cost Search (`ucs`)
Use for cheapest path in weighted graphs.

```python
from problemsolving import solve

solve(
    engine="ucs",
    input_data={
        "graph": {"A": [("B", 1), ("C", 5)], "B": [("D", 3)], "C": [("D", 1)], "D": []},
        "start": "A",
        "goal": "D",
    },
)
```

### 2.4 A* (`astar`)
Use for weighted pathfinding when heuristic is available and optimality matters.

```python
from problemsolving import solve

solve(
    engine="astar",
    input_data={
        "graph": {"A": [("B", 1), ("C", 4)], "B": [("D", 5)], "C": [("D", 1)], "D": []},
        "start": "A",
        "goal": "D",
        "heuristic": {"A": 5, "B": 4, "C": 1, "D": 0},
    },
)
```

### 2.5 Greedy Best-First (`greedy`)
Use when speed is preferred over guaranteed optimality.

```python
from problemsolving import solve

solve(
    engine="greedy",
    input_data={
        "graph": {"A": [("B", 1), ("C", 4)], "B": [("D", 5)], "C": [("D", 1)], "D": []},
        "start": "A",
        "goal": "D",
        "heuristic": {"A": 5, "B": 4, "C": 1, "D": 0},
    },
)
```

---

## 3. Optimization Engines

### 3.1 Gradient Descent (`gradient_descent`)
Protocol adapter minimizes quadratic objective based on coefficients and offsets.

```python
from problemsolving import solve

solve(
    engine="gradient_descent",
    input_data={
        "coefficients": [1, 1],
        "offsets": [3, 4],
        "initial": [0.0, 0.0],
        "learning_rate": 0.1,
        "max_iterations": 200,
    },
)
```

### 3.2 Genetic Algorithm (`genetic_algorithm`)
Protocol adapter maximizes fitness based on closeness to `target`.

```python
from problemsolving import solve

solve(
    engine="genetic_algorithm",
    input_data={
        "bounds": [[-10, 10], [-10, 10]],
        "target": [3, 4],
        "population_size": 80,
        "generations": 120,
    },
)
```

### 3.3 Simulated Annealing (`simulated_annealing`)
Protocol adapter minimizes squared distance to `target`.

```python
from problemsolving import solve

solve(
    engine="simulated_annealing",
    input_data={
        "initial": [0.0, 0.0],
        "target": [7, -3],
        "step_size": 0.5,
        "temperature": 10.0,
        "cooling_rate": 0.995,
        "max_iterations": 5000,
    },
)
```

---

## 4. SAT and CSP

### 4.1 DPLL SAT (`dpll_sat`)

```python
from problemsolving import solve

solve(
    engine="dpll_sat",
    input_data={
        "clauses": [[1, 2], [-1, 3]],
        "num_vars": 3,
    },
)
```

### 4.2 CSP Backtracking (`csp_backtracking`)

```python
from problemsolving import solve

solve(
    engine="csp_backtracking",
    input_data={
        "variables": ["WA", "NT", "SA"],
        "domains": {
            "WA": ["red", "green", "blue"],
            "NT": ["red", "green", "blue"],
            "SA": ["red", "green", "blue"],
        },
        "not_equal_constraints": [["WA", "NT"], ["WA", "SA"], ["NT", "SA"]],
    },
)
```

---

## 5. Symbolic Engines

### 5.1 CAS (`cas`)
Supports multiple operations via `operation` in `input_data`.

#### Solve equation

```python
from problemsolving import solve

solve(
    engine="cas",
    input_data={
        "operation": "solve",
        "equation": "x**2 - 5*x + 6",
        "variable": "x",
    },
)
```

#### Solve system

```python
from problemsolving import solve

solve(
    engine="cas",
    input_data={
        "operation": "solve_system",
        "equations": ["x + y - 10", "x - y - 4"],
        "variables": ["x", "y"],
    },
)
```

#### Differentiate / Integrate / Simplify

```python
from problemsolving import solve

solve(engine="cas", input_data={"operation": "differentiate", "expression": "x**3 + 2*x", "variable": "x"})
solve(engine="cas", input_data={"operation": "integrate", "expression": "2*x", "variable": "x"})
solve(engine="cas", input_data={"operation": "simplify", "expression": "(x**2 - 1)/(x - 1)"})
```

### 5.2 SMT-lite (`smt_lite`)
Linear integer arithmetic satisfiability.

```python
from problemsolving import solve

solve(
    engine="smt_lite",
    input_data={
        "variables": ["x", "y"],
        "constraints": ["x + y == 10", "x > y"],
    },
)
```

---

## 6. Logic Engines

### 6.1 Prolog-lite (`prolog_lite`)

```python
from problemsolving import solve

solve(
    engine="prolog_lite",
    input_data={
        "facts": [
            {"predicate": "parent", "args": ["tom", "bob"]},
            {"predicate": "parent", "args": ["bob", "ann"]},
        ],
        "rules": [
            {
                "head_predicate": "grandparent",
                "head_args": ["?X", "?Z"],
                "body": [
                    {"predicate": "parent", "args": ["?X", "?Y"]},
                    {"predicate": "parent", "args": ["?Y", "?Z"]},
                ],
            }
        ],
        "query": {"predicate": "grandparent", "args": ["tom", "?Z"]},
    },
)
```

### 6.2 Rule Engine (`rule_engine`)

```python
from problemsolving import solve

solve(
    engine="rule_engine",
    input_data={
        "facts": {"temperature": 105, "cough": True},
        "rules": [
            {
                "name": "fever_check",
                "if_fact": "temperature",
                "threshold": 100,
                "then_set": "has_fever",
                "then_value": True,
            }
        ],
    },
)
```

---

## 7. Algorithm Selection (`select_algorithm`)

```python
from problemsolving.selector import select_algorithm

decision = select_algorithm(
    problem_type="optimization",
    features={"differentiable": True},
)

print(decision)
# {'algorithm': 'gradient_descent', 'reasoning': '...', 'alternatives': [...]}
```

Supported problem types currently include:

- `pathfinding`
- `satisfiability`
- `constraint_satisfaction`
- `optimization`
- `symbolic_math`
- `arithmetic_constraints`
- `logic_programming`
- `rule_based`

---

## 8. MCP Features

MCP interface is exposed through:

```python
from problemsolving.mcp.server import get_manifest, dispatch_tool
```

### Tool: `list_engines`

```python
dispatch_tool("list_engines", {"tag": "search"})
```

### Tool: `select_algorithm`

```python
dispatch_tool(
    "select_algorithm",
    {"problem_type": "pathfinding", "features": {"weighted": False}},
)
```

### Tool: `solve`

```python
dispatch_tool(
    "solve",
    {
        "engine": "bfs",
        "input": {
            "graph": {"A": ["B"], "B": ["C"], "C": []},
            "start": "A",
            "goal": "C",
        },
    },
)
```

### Tool: `verify`

```python
dispatch_tool(
    "verify",
    {
        "engine": "bfs",
        "input": {
            "graph": {"A": ["B"], "B": ["C"], "C": []},
            "start": "A",
            "goal": "C",
        },
        "expected": {"path": ["A", "B", "C"]},
    },
)
```

### Tool: `explain_algorithm`

```python
dispatch_tool("explain_algorithm", {"algorithm": "bfs"})
```

---

## 9. Practical Example Paths

- `examples/python/real_world_logistics_planning.py`
- `examples/python/constraint_based_scheduling.py`
- `examples/python/llm_verified_reasoning.py`
- `examples/python/mcp_usage.py`

Run from repository root:

```bash
PYTHONPATH=src python examples/python/real_world_logistics_planning.py
```
