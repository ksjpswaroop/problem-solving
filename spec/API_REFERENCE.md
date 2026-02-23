# ProblemSolving API Reference

Version 0.1.0

## Table of Contents

1. [Unified API](#1-unified-api)
2. [Search Engines](#2-search-engines)
3. [Optimization Engines](#3-optimization-engines)
4. [SAT & CSP Engines](#4-sat--csp-engines)
5. [Symbolic Engines](#5-symbolic-engines)
6. [Logic Engines](#6-logic-engines)
7. [Algorithm Selector](#7-algorithm-selector)
8. [MCP Server Tools](#8-mcp-server-tools)
9. [Protocol Layer](#9-protocol-layer)

---

## 1. Unified API

### `solve(engine, input_data, ...)`

The top-level entry point. Routes to any registered engine or auto-selects.

```python
from problemsolving import solve

response = solve(
    engine: str,               # Engine name or "auto"
    input_data: dict,          # Engine-specific input
    operation: str = "solve",  # Operation name
    config: dict | None = None,# Engine config overrides
    problem_type: str | None = None,  # For auto-selection
    features: dict | None = None,     # For auto-selection
) -> SolverResponse
```

**Returns:** `SolverResponse` with fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Request ID (auto-generated if not provided) |
| `status` | `str` | `"success"` or `"error"` |
| `engine` | `str` | Engine that handled the request |
| `result` | `dict \| None` | Engine output on success |
| `error` | `dict \| None` | Error info on failure (`code`, `message`, `suggestion`) |
| `proof_trace` | `list[dict]` | Step-by-step reasoning trace |
| `metadata` | `dict` | Timing, stats, etc. |

**Error codes:**

| Code | Meaning |
|------|---------|
| `UNKNOWN_ENGINE` | Engine name not found in registry |
| `SOLVER_ERROR` | Engine raised an exception during solve |

**Example:**

```python
from problemsolving import solve

result = solve(
    engine="bfs",
    input_data={
        "graph": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
        "start": "A",
        "goal": "D",
    },
)
assert result.status == "success"
print(result.result["path"])  # ["A", "B", "D"] or ["A", "C", "D"]
```

### `list_engines(tag=None)`

```python
from problemsolving import list_engines

engines = list_engines(tag: str | None = None) -> list[str]
```

Returns all registered engine names. Pass `tag` to filter (e.g. `"search"`, `"optimization"`).

---

## 2. Search Engines

All search engines share a common interface pattern: a generic function accepting callables, plus a protocol adapter accepting dict input.

### 2.1 BFS — Breadth-First Search

**Engine name:** `bfs`
**Tags:** `search`, `pathfinding`

**Generic API:**

```python
from problemsolving.search.bfs import bfs_solve

result = bfs_solve(
    start: T,                           # Initial state
    goal_test: Callable[[T], bool],     # Returns True at goal
    neighbors: Callable[[T], list[T]],  # Returns adjacent states
) -> dict | None
```

**Returns (success):** `{"path": list[T], "nodes_explored": int}`
**Returns (no path):** `None`

**Protocol input:**

```json
{
  "graph": {"A": ["B", "C"], "B": ["D"], "C": [], "D": []},
  "start": "A",
  "goal": "D"
}
```

**Guarantees:** Shortest path by hop count in unweighted graphs.

### 2.2 DFS — Depth-First Search

**Engine name:** `dfs`
**Tags:** `search`, `pathfinding`

**Generic API:**

```python
from problemsolving.search.dfs import dfs_solve

result = dfs_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[T]],
    max_depth: int | None = None,       # Optional depth limit
) -> dict | None
```

**Returns (success):** `{"path": list[T], "nodes_explored": int}`
**Returns (no path):** `None`

**Protocol input:**

```json
{
  "graph": {"A": ["B", "C"], "B": ["D"], "C": [], "D": []},
  "start": "A",
  "goal": "D",
  "max_depth": 5
}
```

**Guarantees:** Finds *a* path (not necessarily shortest). Handles cycles. Respects depth limit.

### 2.3 UCS — Uniform Cost Search

**Engine name:** `ucs`
**Tags:** `search`, `pathfinding`, `optimal`

**Generic API:**

```python
from problemsolving.search.ucs import ucs_solve

result = ucs_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[tuple[T, float]]],  # (node, cost) pairs
) -> dict | None
```

**Returns (success):** `{"path": list[T], "cost": float, "nodes_explored": int}`
**Returns (no path):** `None`

**Protocol input:**

```json
{
  "graph": {
    "A": [["B", 1], ["C", 5]],
    "B": [["D", 3]],
    "C": [["D", 1]],
    "D": []
  },
  "start": "A",
  "goal": "D"
}
```

**Guarantees:** Optimal (cheapest cost) path with non-negative edge weights.

### 2.4 A* Search

**Engine name:** `astar`
**Tags:** `search`, `pathfinding`, `optimal`, `heuristic`

**Generic API:**

```python
from problemsolving.search.astar import astar_solve

result = astar_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[tuple[T, float]]],
    heuristic: Callable[[T], float],   # Admissible estimate to goal
) -> dict | None
```

**Returns (success):** `{"path": list[T], "cost": float, "nodes_explored": int}`
**Returns (no path):** `None`

**Protocol input:**

```json
{
  "graph": {
    "A": [["B", 1], ["C", 4]],
    "B": [["D", 5]],
    "C": [["D", 1]],
    "D": []
  },
  "start": "A",
  "goal": "D",
  "heuristic": {"A": 5, "B": 4, "C": 1, "D": 0}
}
```

**Guarantees:** Optimal with admissible heuristic. Evaluates `f(n) = g(n) + h(n)`.

### 2.5 Greedy Best-First Search

**Engine name:** `greedy`
**Tags:** `search`, `pathfinding`, `heuristic`

**Generic API:**

```python
from problemsolving.search.greedy import greedy_solve

result = greedy_solve(
    start: T,
    goal_test: Callable[[T], bool],
    neighbors: Callable[[T], list[tuple[T, float]]],
    heuristic: Callable[[T], float],
) -> dict | None
```

**Returns (success):** `{"path": list[T], "nodes_explored": int}`
**Returns (no path):** `None`

**Protocol input:** Same as A* (graph + heuristic dict).

**Guarantees:** Fast but NOT optimal. Expands closest-to-goal first by heuristic alone.

---

## 3. Optimization Engines

### 3.1 Gradient Descent

**Engine name:** `gradient_descent`
**Tags:** `optimization`, `gradient`

**Generic API:**

```python
from problemsolving.optimization.gradient_descent import gd_solve

result = gd_solve(
    objective: Callable[[list[float]], float],    # Function to minimize
    gradient: Callable[[list[float]], list[float]], # Gradient of objective
    initial: list[float],                         # Starting point
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,                      # Gradient norm threshold
) -> dict
```

**Returns:** `{"solution": list[float], "objective_value": float, "iterations": int}`

**Protocol input:**

```json
{
  "coefficients": [1, 1],
  "offsets": [3, 4],
  "initial": [0.0, 0.0],
  "learning_rate": 0.1,
  "max_iterations": 200
}
```

Protocol adapter minimizes `sum(c_i * (x_i - offset_i)^2)`.

### 3.2 Genetic Algorithm

**Engine name:** `genetic_algorithm`
**Tags:** `optimization`, `evolutionary`

**Generic API:**

```python
from problemsolving.optimization.genetic import ga_solve

result = ga_solve(
    fitness: Callable[[list[float]], float],  # Maximize this
    bounds: list[tuple[float, float]],        # (min, max) per dimension
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    elite_fraction: float = 0.1,
) -> dict
```

**Returns:** `{"solution": list[float], "fitness": float, "generations": int}`

**Protocol input:**

```json
{
  "bounds": [[-10, 10], [-10, 10]],
  "target": [3, 4],
  "population_size": 50,
  "generations": 100
}
```

### 3.3 Simulated Annealing

**Engine name:** `simulated_annealing`
**Tags:** `optimization`, `metaheuristic`

**Generic API:**

```python
from problemsolving.optimization.simulated_annealing import sa_solve

result = sa_solve(
    objective: Callable[[list[float]], float],     # Minimize this
    initial: list[float],
    neighbor_fn: Callable[[list[float]], list[float]],  # Generate neighbor
    temperature: float = 100.0,
    cooling_rate: float = 0.995,
    max_iterations: int = 10000,
    min_temperature: float = 1e-10,
) -> dict
```

**Returns:** `{"solution": list[float], "objective_value": float, "iterations": int}`

**Protocol input:**

```json
{
  "initial": [0.0, 0.0],
  "target": [7, -3],
  "step_size": 0.5,
  "temperature": 10.0,
  "cooling_rate": 0.995,
  "max_iterations": 5000
}
```

---

## 4. SAT & CSP Engines

### 4.1 DPLL SAT Solver

**Engine name:** `dpll_sat`
**Tags:** `sat`, `logic`, `csp`

**Generic API:**

```python
from problemsolving.csp.sat import dpll_solve

result = dpll_solve(
    clauses: list[list[int]],  # CNF: positive=True, negative=False
    num_vars: int,             # Number of variables (1-indexed)
) -> dict
```

**Returns:** `{"satisfiable": bool, "assignment": dict[int, bool]}`

Clause format: `[[1, 2], [-1, 3]]` means `(x₁ ∨ x₂) ∧ (¬x₁ ∨ x₃)`.

**Protocol input:**

```json
{
  "clauses": [[1, 2], [-1, 3]],
  "num_vars": 3
}
```

**Features:** Unit propagation, pure literal elimination, backtracking.

### 4.2 CSP Backtracking

**Engine name:** `csp_backtracking`
**Tags:** `csp`, `constraint`

**Generic API:**

```python
from problemsolving.csp.backtracking import csp_solve

result = csp_solve(
    variables: list[str],
    domains: dict[str, list[Any]],
    constraints: list[tuple[str, str, Callable[..., bool]]],
) -> dict
```

**Returns:** `{"satisfiable": bool, "assignment": dict[str, Any], "nodes_explored": int}`

**Protocol input:**

```json
{
  "variables": ["WA", "NT", "SA"],
  "domains": {"WA": ["red", "green", "blue"], "NT": ["red", "green", "blue"], "SA": ["red", "green", "blue"]},
  "not_equal_constraints": [["WA", "NT"], ["WA", "SA"], ["NT", "SA"]]
}
```

**Features:** Backtracking with MRV (Minimum Remaining Values) heuristic.

---

## 5. Symbolic Engines

### 5.1 CAS — Computer Algebra System

**Engine name:** `cas`
**Tags:** `symbolic`, `algebra`, `calculus`

CAS supports multiple operations through a single engine.

**Solve equation:**

```python
from problemsolving.symbolic.cas import cas_solve

result = cas_solve(
    equation: str,         # Expression equal to zero
    variable: str,         # Variable to solve for
    domain: str = "complex",  # "real" or "complex"
) -> dict
# Returns: {"solutions": list[int|float], "variable": str}
```

**Solve system:**

```python
from problemsolving.symbolic.cas import cas_solve_system

result = cas_solve_system(
    equations: list[str],   # Expressions equal to zero
    variables: list[str],
) -> dict
# Returns: {"solution": dict[str, int|float], "solvable": bool}
```

**Differentiate:**

```python
from problemsolving.symbolic.cas import cas_differentiate

result = cas_differentiate(
    expression: str,
    variable: str,
) -> dict
# Returns: {"derivative": str, "variable": str}
```

**Integrate:**

```python
from problemsolving.symbolic.cas import cas_integrate

result = cas_integrate(
    expression: str,
    variable: str,
) -> dict
# Returns: {"integral": str, "variable": str}
```

**Simplify:**

```python
from problemsolving.symbolic.cas import cas_simplify

result = cas_simplify(expression: str) -> dict
# Returns: {"simplified": str}
```

**Protocol input (via `operation` field):**

```json
{"operation": "solve", "equation": "x**2 - 5*x + 6", "variable": "x"}
{"operation": "solve_system", "equations": ["x+y-10", "x-y-4"], "variables": ["x","y"]}
{"operation": "differentiate", "expression": "x**3 + 2*x", "variable": "x"}
{"operation": "integrate", "expression": "2*x", "variable": "x"}
{"operation": "simplify", "expression": "(x**2 - 1) / (x - 1)"}
```

### 5.2 SMT-Lite Solver

**Engine name:** `smt_lite`
**Tags:** `smt`, `arithmetic`, `constraint`

**Generic API:**

```python
from problemsolving.symbolic.smt_lite import smt_solve

result = smt_solve(
    variables: list[str],
    constraints: list[str],  # e.g. ["x > 0", "x + y == 10"]
) -> dict
```

**Returns:** `{"satisfiable": bool, "assignment": dict[str, int]}`

**Protocol input:**

```json
{
  "variables": ["x", "y"],
  "constraints": ["x + y == 10", "x > y"]
}
```

**Note:** `==` in constraint strings is automatically converted to SymPy `Eq()`. Supports `>`, `<`, `>=`, `<=`, `==` operators over integers.

---

## 6. Logic Engines

### 6.1 Prolog-Lite

**Engine name:** `prolog_lite`

**Class API:**

```python
from problemsolving.logic.prolog_lite import PrologEngine

engine = PrologEngine()

engine.add_fact(predicate: str, args: tuple[str, ...]) -> None
engine.add_rule(
    head_predicate: str,
    head_args: tuple[str, ...],       # Variables start with "?"
    body: list[tuple[str, tuple[str, ...]]],
) -> None
results = engine.query(predicate: str, args: tuple[str, ...]) -> list[dict[str, str]]
```

**Protocol input:**

```json
{
  "facts": [
    {"predicate": "parent", "args": ["tom", "bob"]},
    {"predicate": "parent", "args": ["bob", "ann"]}
  ],
  "rules": [
    {
      "head_predicate": "grandparent",
      "head_args": ["?X", "?Z"],
      "body": [
        {"predicate": "parent", "args": ["?X", "?Y"]},
        {"predicate": "parent", "args": ["?Y", "?Z"]}
      ]
    }
  ],
  "query": {"predicate": "grandparent", "args": ["tom", "?Z"]}
}
```

**Returns:** `{"results": [{"Z": "ann"}], "count": 1}`

### 6.2 Rule Engine

**Engine name:** `rule_engine`

**Class API:**

```python
from problemsolving.logic.rule_engine import RuleEngine

engine = RuleEngine()

engine.add_fact(key: str, value: Any) -> None
engine.add_rule(
    name: str,
    condition: Callable[[dict[str, Any]], bool],
    action: Callable[[dict[str, Any]], None],
) -> None
trace = engine.run(max_iterations: int = 100) -> list[dict[str, Any]]
```

**Protocol input:**

```json
{
  "facts": {"temperature": 105, "cough": true},
  "rules": [
    {"name": "fever", "if_fact": "temperature", "threshold": 100, "then_set": "has_fever", "then_value": true}
  ]
}
```

**Returns:** `{"facts": dict, "rules_fired": list[str], "trace": list[dict]}`

**Trace entry:** `{"rule": str, "facts_before": dict, "facts_after": dict}`

---

## 7. Algorithm Selector

```python
from problemsolving.selector import select_algorithm

result = select_algorithm(
    problem_type: str,                   # See table below
    features: dict[str, Any] | None = None,
) -> dict
```

**Returns:** `{"algorithm": str, "reasoning": str, "alternatives": list[str]}`

**Problem types and features:**

| Problem Type | Feature Keys | Selected Algorithm |
|---|---|---|
| `pathfinding` | `weighted=False` | `bfs` |
| `pathfinding` | `weighted=True, has_heuristic=True, needs_optimal=True` | `astar` |
| `pathfinding` | `weighted=True, has_heuristic=True, needs_optimal=False` | `greedy` |
| `pathfinding` | `weighted=True, has_heuristic=False, needs_optimal=True` | `ucs` |
| `satisfiability` | `domain=boolean` | `dpll_sat` |
| `constraint_satisfaction` | *(any)* | `csp_backtracking` |
| `optimization` | `differentiable=True` | `gradient_descent` |
| `optimization` | `discrete=True` | `genetic_algorithm` |
| `optimization` | *(default)* | `simulated_annealing` |
| `symbolic_math` | *(any)* | `cas` |
| `arithmetic_constraints` | *(any)* | `smt_lite` |
| `logic_programming` | *(any)* | `prolog_lite` |
| `rule_based` | *(any)* | `rule_engine` |

---

## 8. MCP Server Tools

The MCP server exposes 5 tools following the Model Context Protocol specification.

### 8.1 `solve`

Solve a problem using a specified engine or auto-select.

**Input schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `engine` | string | yes | Engine name or `"auto"` |
| `input` | object | yes | Engine-specific input data |
| `problem_type` | string | no | For auto-selection |
| `features` | object | no | For auto-selection |

**Output:** Full `SolverResponse` as dict.

### 8.2 `select_algorithm`

Select the best algorithm for a problem type.

**Input schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `problem_type` | string | yes | Problem category |
| `features` | object | no | Problem characteristics |

**Output:** `{"algorithm": str, "reasoning": str, "alternatives": list[str]}`

### 8.3 `list_engines`

List available engines.

**Input schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tag` | string | no | Filter by tag |

**Output:** `{"engines": list[str]}`

### 8.4 `verify`

Run an engine and compare output to expected values.

**Input schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `engine` | string | yes | Engine to run |
| `input` | object | yes | Engine input |
| `expected` | object | yes | Expected result fields |

**Output:** `{"verified": bool, "mismatches": list[str], "actual": dict}`

### 8.5 `explain_algorithm`

Get metadata about an algorithm from the knowledge base.

**Input schema:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `algorithm` | string | yes | Algorithm ID |

**Output:** `{"found": bool, "id": str, "name": str, "description": str, "when_to_use": list, "when_not_to_use": list, "complexity": dict, "tags": list}`

---

## 9. Protocol Layer

### SolverRequest

```python
@dataclass
class SolverRequest:
    engine: str                        # Target engine name
    operation: str                     # Operation name (default: "solve")
    input_data: dict[str, Any]         # Engine-specific input
    config: dict[str, Any] = {}        # Optional config overrides
    id: str | None = None              # Auto-generated if not provided
```

**Serialized form (`.to_dict()`):**

```json
{
  "id": "req_a1b2c3d4e5f6",
  "engine": "bfs",
  "operation": "solve",
  "input": { ... },
  "config": { }
}
```

### SolverResponse

```python
@dataclass
class SolverResponse:
    id: str
    status: str                                # "success" | "error"
    engine: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    proof_trace: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
```

**Factory methods:**

- `SolverResponse.success(request_id, engine, result, proof_trace=None, metadata=None)`
- `SolverResponse.make_error(request_id, engine, code, message, suggestion=None)`

### EngineRegistry

```python
class EngineRegistry:
    def register(name: str, solve_fn: Callable, tags: list[str] | None) -> None
    def get(name: str) -> RegisteredEngine | None
    def list_engines(tag: str | None = None) -> list[str]

def get_default_registry() -> EngineRegistry  # Lazily creates + registers builtins
```

**Registered engines (default):**

`bfs`, `dfs`, `ucs`, `astar`, `greedy`, `gradient_descent`, `genetic_algorithm`, `simulated_annealing`, `dpll_sat`, `csp_backtracking`, `cas`, `smt_lite`, `prolog_lite`, `rule_engine`
