# ProblemSolving Functional Specification

Version 0.1.0

## 1. Overview

ProblemSolving is a universal AI reasoning toolkit providing classical algorithms and symbolic engines as a programmatic alternative to LLM Chain-of-Thought reasoning. It provides a unified interface for agents, applications, and LLMs to delegate exact computation to verified solvers.

### 1.1 Design Principles

- **Correctness over speed:** Every engine must produce provably correct results for its problem class.
- **Pluggable engines:** New engines can be registered at runtime without modifying core code.
- **Protocol-first:** All engines communicate through a standardized request/response envelope.
- **TDD methodology:** Every feature is test-driven; no code ships without passing tests.
- **Fail-safe:** Engines return structured errors, never crash the host process.

### 1.2 Supported Problem Classes

| Class | Engines | Use Case |
|-------|---------|----------|
| Pathfinding | BFS, DFS, UCS, A*, Greedy | Graph traversal, route planning |
| Optimization | Gradient Descent, Genetic Algorithm, Simulated Annealing | Function minimization/maximization |
| Satisfiability | DPLL SAT | Boolean constraint solving |
| Constraint Satisfaction | CSP Backtracking | Variable assignment under constraints |
| Symbolic Math | CAS (SymPy) | Equation solving, calculus, simplification |
| Arithmetic Constraints | SMT-Lite | Integer inequality/equality solving |
| Logic Programming | Prolog-Lite | Fact/rule-based queries |
| Rule-Based Reasoning | Rule Engine | Forward-chaining inference |

---

## 2. Functional Requirements

### 2.1 Unified API

**FR-001:** The `solve()` function MUST accept any registered engine name and route to that engine.

**FR-002:** When `engine="auto"`, the selector MUST choose an appropriate engine based on `problem_type` and `features`.

**FR-003:** If the requested engine is not registered, `solve()` MUST return a `SolverResponse` with `status="error"` and `code="UNKNOWN_ENGINE"`.

**FR-004:** `solve()` MUST never raise an exception to the caller. All errors are wrapped in `SolverResponse.make_error()`.

**FR-005:** `list_engines()` MUST return all registered engine names as a list of strings.

**FR-006:** `list_engines(tag="X")` MUST return only engines whose tags include `"X"`.

### 2.2 Search Engines

**FR-010:** BFS MUST return the shortest path by hop count in unweighted graphs.

**FR-011:** BFS MUST handle cycles without infinite loops.

**FR-012:** BFS MUST return `None` when no path exists.

**FR-013:** DFS MUST find a path to the goal if one exists (not necessarily shortest).

**FR-014:** DFS MUST respect `max_depth` parameter when provided, not exploring beyond that depth.

**FR-015:** DFS MUST handle cycles without infinite loops.

**FR-016:** UCS MUST return the cheapest-cost path (not necessarily shortest by hops).

**FR-017:** UCS MUST handle zero-cost edges correctly.

**FR-018:** A* MUST return optimal paths when the heuristic is admissible (never overestimates).

**FR-019:** A* MUST use `f(n) = g(n) + h(n)` for node evaluation.

**FR-020:** Greedy Best-First Search MUST expand nodes by heuristic value alone (h only, no g).

**FR-021:** Greedy MAY return suboptimal paths — this is not a defect.

**FR-022:** All search engines MUST return `nodes_explored` count in results.

**FR-023:** All search engines MUST handle `start == goal` by returning a single-element path.

### 2.3 Optimization Engines

**FR-030:** Gradient Descent MUST converge toward the minimum for convex functions with appropriate learning rate.

**FR-031:** Gradient Descent MUST stop early when gradient norm falls below `tolerance`.

**FR-032:** Gradient Descent MUST track iteration count in results.

**FR-033:** Genetic Algorithm MUST maximize the fitness function.

**FR-034:** Genetic Algorithm MUST keep solutions within specified bounds.

**FR-035:** Genetic Algorithm MUST implement elitism (preserve top individuals across generations).

**FR-036:** Simulated Annealing MUST minimize the objective function.

**FR-037:** Simulated Annealing MUST be capable of escaping local minima (probabilistic uphill moves at high temperature).

**FR-038:** Simulated Annealing MUST reduce temperature by `cooling_rate` each iteration.

### 2.4 SAT & CSP Engines

**FR-040:** DPLL MUST correctly identify satisfiable formulas and return a satisfying assignment.

**FR-041:** DPLL MUST correctly identify unsatisfiable formulas.

**FR-042:** DPLL MUST implement unit propagation (unit clauses force variable assignment).

**FR-043:** DPLL MUST implement pure literal elimination.

**FR-044:** DPLL MUST handle empty clause sets as trivially satisfiable.

**FR-045:** DPLL assignments MUST satisfy all clauses (verifiable post-hoc).

**FR-046:** CSP solver MUST find an assignment satisfying all constraints if one exists.

**FR-047:** CSP solver MUST report `satisfiable=False` when no valid assignment exists.

**FR-048:** CSP solver MUST use MRV (Minimum Remaining Values) heuristic for variable ordering.

**FR-049:** CSP solver MUST track `nodes_explored` count.

### 2.5 Symbolic Engines

**FR-050:** CAS `solve` MUST find all solutions of polynomial equations.

**FR-051:** CAS `solve` with `domain="real"` MUST filter out complex solutions.

**FR-052:** CAS `solve_system` MUST solve systems of simultaneous equations.

**FR-053:** CAS `differentiate` MUST return the correct symbolic derivative.

**FR-054:** CAS `integrate` MUST return the correct indefinite integral (without +C).

**FR-055:** CAS `simplify` MUST reduce expressions to simpler equivalent forms.

**FR-056:** SMT-Lite MUST find integer assignments satisfying all constraints when they exist.

**FR-057:** SMT-Lite MUST correctly identify unsatisfiable constraint sets.

**FR-058:** SMT-Lite MUST handle `==` by converting to SymPy `Eq()` (not Python equality).

**FR-059:** SMT-Lite MUST handle `>`, `<`, `>=`, `<=` relational operators.

### 2.6 Logic Engines

**FR-060:** Prolog-Lite MUST unify variables (prefixed with `?`) against ground terms.

**FR-061:** Prolog-Lite MUST support rules with multiple body goals (conjunction).

**FR-062:** Prolog-Lite MUST return all matching bindings for a query.

**FR-063:** Prolog-Lite MUST handle ground queries (no variables) as boolean checks.

**FR-064:** Rule Engine MUST fire rules whose conditions match current facts.

**FR-065:** Rule Engine MUST support chained rules (rule A produces fact that triggers rule B).

**FR-066:** Rule Engine MUST prevent the same rule from firing more than once.

**FR-067:** Rule Engine MUST reach quiescence (stop when no new rules can fire).

**FR-068:** Rule Engine MUST return a trace of fired rules with before/after fact snapshots.

### 2.7 Algorithm Selector

**FR-070:** Selector MUST match by `problem_type` first, then rank by feature specificity.

**FR-071:** Selector MUST return a `reasoning` string explaining the selection.

**FR-072:** Selector MUST return up to 3 `alternatives` when multiple engines match.

**FR-073:** When no rule matches, selector MUST default to `bfs`.

### 2.8 MCP Server

**FR-080:** MCP manifest MUST include all 5 tools: `solve`, `select_algorithm`, `list_engines`, `verify`, `explain_algorithm`.

**FR-081:** Every MCP tool MUST have a valid JSON Schema for its `inputSchema`.

**FR-082:** `dispatch_tool` MUST route to the correct handler by tool name.

**FR-083:** `dispatch_tool` for unknown tools MUST return `{"error": "Unknown tool: ..."}`.

**FR-084:** `verify` tool MUST compare each key in `expected` against the engine's actual output.

**FR-085:** `explain_algorithm` MUST load metadata from `knowledge/algorithms/<id>.json`.

**FR-086:** `explain_algorithm` for unknown algorithms MUST return `{"found": false}`.

---

## 3. Edge Cases & Error Handling

### 3.1 Search

| Scenario | Expected Behavior |
|----------|-------------------|
| Empty graph | Return `None` (no path) |
| Start == Goal | Return single-element path |
| Disconnected graph | Return `None` for unreachable goal |
| Self-loop in graph | Handle without infinite loop |
| Very large graph (10k+ nodes) | Complete within reasonable time (< 5s) |

### 3.2 Optimization

| Scenario | Expected Behavior |
|----------|-------------------|
| Flat objective (gradient = 0) | Stop immediately at initial point |
| Non-convex function | SA may find global min; GD may find local min |
| Bounds violation | GA clamps solutions to specified bounds |
| Zero learning rate | GD returns initial point unchanged |

### 3.3 SAT / CSP

| Scenario | Expected Behavior |
|----------|-------------------|
| Empty clause set | Return satisfiable = True |
| Single unit clause | Propagate immediately |
| Contradictory clauses `[1], [-1]` | Return satisfiable = False |
| Over-constrained CSP | Return satisfiable = False |
| Large CSP (8-queens) | Complete within reasonable time |

### 3.4 Symbolic

| Scenario | Expected Behavior |
|----------|-------------------|
| No real solutions (x²+1=0, domain=real) | Return empty solutions list |
| Division by zero in expression | SymPy handles symbolically |
| Malformed constraint string | May raise exception (wrapped by protocol) |

### 3.5 Logic

| Scenario | Expected Behavior |
|----------|-------------------|
| Query on empty fact base | Return empty results |
| Circular rules | Rule engine stops at `max_iterations` |
| Variable in fact position | Treated as ground term |

---

## 4. Non-Functional Requirements

**NFR-001:** All engines MUST be importable without optional dependencies (graceful degradation).

**NFR-002:** CAS and SMT-Lite require `sympy` (optional dependency group `[cas]`).

**NFR-003:** The library MUST support Python 3.10+.

**NFR-004:** All public functions MUST pass mypy strict type checking.

**NFR-005:** All code MUST pass ruff linting with configured rules.

**NFR-006:** Test suite MUST maintain > 90% line coverage for all engine modules.

**NFR-007:** No engine solve call should take more than 30 seconds for reasonable inputs.

---

## 5. Acceptance Criteria

### 5.1 Per-Engine Acceptance

Each engine is accepted when:

1. All tests pass (RED → GREEN verified)
2. Lint clean (`ruff check`)
3. Type check clean (`mypy`)
4. Registered in engine registry
5. Protocol adapter (`*_solve_from_dict`) exists and works through `solve()`
6. Knowledge base metadata JSON exists (where applicable)

### 5.2 System Acceptance

The system is accepted when:

1. All 115+ tests pass
2. `solve(engine="auto", ...)` correctly routes for all problem types
3. MCP manifest is valid and all 5 tools dispatch correctly
4. `verify` tool can validate any engine's output
5. All specs are consistent with implementation
