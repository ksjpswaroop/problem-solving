# Implementation Plan: ProblemSolving Toolkit

**Methodology:** Test-Driven Development (TDD) — every task starts with writing tests, then implementation to pass them.
**Principle:** Small, measurable progress. Every phase ends with something that works, ships, and is independently useful.
**Language:** Python-first (Phase 1-8). Rust core added in Phase 9-10. TypeScript SDK in Phase 11.

---

## How to Read This Plan

Each phase follows this rhythm:

```
1. Write failing tests (RED)
2. Implement minimum code to pass (GREEN)
3. Refactor (REFACTOR)
4. Measure: run test suite, check acceptance criteria
5. Ship: tag release, update changelog
```

Every phase has:
- **Goal:** One sentence, what this phase achieves
- **Depends on:** Which prior phases must be complete
- **Deliverable:** The tangible output
- **Acceptance criteria:** Binary pass/fail checks
- **Tasks:** Ordered by TDD sequence (tests first)
- **Duration:** Estimated calendar time (solo developer)
- **Exit gate:** What must be true before moving to next phase

---

## Phase 0: Project Skeleton

**Goal:** Repository structure, CI, and development environment — zero features, but everything needed to write the first test.

**Depends on:** Nothing
**Duration:** 1 day
**Deliverable:** Empty package that installs and runs an empty test suite

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| 0.1 | Create monorepo structure matching ARCHITECTURE.md section 8 | Setup | 1h |
| 0.2 | Write `pyproject.toml` with maturin build, pytest, ruff, mypy configured | Setup | 1h |
| 0.3 | Create `src/problemsolving/__init__.py` with version string | Setup | 15m |
| 0.4 | Write first test: `test_import.py` — `import problemsolving` succeeds, version is a string | RED→GREEN | 15m |
| 0.5 | Set up GitHub Actions: run pytest + ruff + mypy on every push | CI | 1h |
| 0.6 | Create `spec/` directory with empty `protocol.json` schema | Setup | 15m |
| 0.7 | Create `knowledge/` directory with `schema.json` (algorithm metadata JSON Schema) | Setup | 30m |
| 0.8 | Add `Makefile` with targets: `test`, `lint`, `typecheck`, `all` | Setup | 15m |

### Acceptance Criteria
- [ ] `pip install -e .` works
- [ ] `make test` runs and passes (1 test)
- [ ] `make lint` passes (ruff)
- [ ] `make typecheck` passes (mypy)
- [ ] CI green on GitHub

### Exit Gate
`make all` passes. Repository is on GitHub with green CI badge.

---

## Phase 1: Solver Protocol + First Algorithm (BFS)

**Goal:** Establish the solver protocol (request/response format) and prove it works end-to-end with one algorithm.

**Depends on:** Phase 0
**Duration:** 3 days
**Deliverable:** `problemsolving.search.bfs` works, protocol layer handles requests/responses

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **Protocol** | | | |
| 1.1 | Write `spec/protocol.json` — JSON Schema for SolverRequest and SolverResponse | Spec | 2h |
| 1.2 | Write tests for protocol models: valid request parses, invalid request rejects, response serializes | RED | 1h |
| 1.3 | Implement `src/problemsolving/protocol/request.py` and `response.py` (Pydantic models) | GREEN | 2h |
| 1.4 | Write tests for engine registry: register engine, list engines, get engine by name | RED | 30m |
| 1.5 | Implement `src/problemsolving/protocol/registry.py` | GREEN | 1h |
| **BFS** | | | |
| 1.6 | Write `spec/engines/bfs.json` — input/output schema for BFS | Spec | 30m |
| 1.7 | Write `knowledge/algorithms/bfs.json` — full metadata (when-to-use, complexity, tags) | Data | 30m |
| 1.8 | Write tests for BFS: shortest path in simple graph, no-path case, single node, cycle handling, large grid (100×100) | RED | 1h |
| 1.9 | Implement `src/problemsolving/search/bfs.py` — generic BFS with state interface | GREEN | 2h |
| 1.10 | Write integration test: submit BFS problem through protocol layer, get protocol response | RED→GREEN | 1h |
| 1.11 | Refactor: extract shared `SearchResult` type, clean up protocol wiring | REFACTOR | 1h |

### Acceptance Criteria
- [ ] 10+ tests pass covering BFS correctness
- [ ] Protocol request → BFS → protocol response works end-to-end
- [ ] `bfs.json` metadata validates against `schema.json`
- [ ] BFS solves 100×100 grid in <100ms

### Exit Gate
`make test` passes with 15+ tests. First algorithm is complete across all layers (code + spec + metadata).

---

## Phase 2: Search Algorithm Family

**Goal:** Complete all 5 search algorithms. Prove the architecture scales to multiple algorithms without protocol changes.

**Depends on:** Phase 1
**Duration:** 4 days
**Deliverable:** BFS, DFS, UCS, A*, Greedy — all with tests, specs, and metadata

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **DFS** | | | |
| 2.1 | Write tests: finds a path, depth-limited, cycle detection, no-path | RED | 30m |
| 2.2 | Implement `search/dfs.py` | GREEN | 1h |
| 2.3 | Write `spec/engines/dfs.json` + `knowledge/algorithms/dfs.json` | Spec+Data | 30m |
| **UCS** | | | |
| 2.4 | Write tests: cheapest path (not shortest hops), tie-breaking, zero-cost edges | RED | 30m |
| 2.5 | Implement `search/ucs.py` | GREEN | 1.5h |
| 2.6 | Write `spec/engines/ucs.json` + `knowledge/algorithms/ucs.json` | Spec+Data | 30m |
| **A*** | | | |
| 2.7 | Write tests: optimal with admissible heuristic, suboptimal with inadmissible, Manhattan on grid, large graph perf | RED | 1h |
| 2.8 | Implement `search/astar.py` | GREEN | 2h |
| 2.9 | Write `spec/engines/astar.json` + `knowledge/algorithms/astar.json` | Spec+Data | 30m |
| **Greedy** | | | |
| 2.10 | Write tests: fast but suboptimal, correct with perfect heuristic | RED | 30m |
| 2.11 | Implement `search/greedy.py` | GREEN | 1h |
| 2.12 | Write `spec/engines/greedy.json` + `knowledge/algorithms/greedy.json` | Spec+Data | 30m |
| **Shared** | | | |
| 2.13 | Refactor: extract `SearchProblem` protocol (interface), shared `Graph` type | REFACTOR | 1h |
| 2.14 | Write comparison test: same graph, all 5 algorithms, verify BFS=shortest hops, UCS/A*=cheapest, Greedy=fast | RED→GREEN | 1h |
| 2.15 | Write performance benchmark: 10K node graph, all algorithms, assert time bounds | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] 40+ tests pass
- [ ] All 5 search algorithms pass correctness tests
- [ ] All 5 have spec + metadata JSON
- [ ] Comparison test proves each algorithm's unique property
- [ ] 10K node graph completes in <1s for all algorithms

### Exit Gate
`make test` passes with 40+ tests. All search algorithms interchangeable via protocol.

---

## Phase 3: Optimization Algorithms

**Goal:** Gradient descent, genetic algorithms, simulated annealing — all with TDD.

**Depends on:** Phase 1 (protocol only, not Phase 2)
**Duration:** 4 days
**Deliverable:** 3 optimization algorithms with tests, specs, metadata

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **Gradient Descent** | | | |
| 3.1 | Write tests: minimize x², minimize Rosenbrock, converges with good LR, diverges with bad LR, custom stopping | RED | 1h |
| 3.2 | Implement `optimization/gradient_descent.py` (numerical gradients + optional analytical) | GREEN | 2h |
| 3.3 | Spec + metadata JSON | Spec+Data | 30m |
| **Genetic Algorithm** | | | |
| 3.4 | Write tests: maximize OneMax, solve simple TSP (5 cities, known optimal), population diversity, convergence | RED | 1h |
| 3.5 | Implement `optimization/genetic.py` (configurable selection, crossover, mutation) | GREEN | 3h |
| 3.6 | Spec + metadata JSON | Spec+Data | 30m |
| **Simulated Annealing** | | | |
| 3.7 | Write tests: escapes local minima (multi-modal function), temperature schedule, deterministic with fixed seed | RED | 1h |
| 3.8 | Implement `optimization/simulated_annealing.py` | GREEN | 2h |
| 3.9 | Spec + metadata JSON | Spec+Data | 30m |
| **Shared** | | | |
| 3.10 | Extract `OptimizationProblem` protocol, shared `OptimizationResult` type | REFACTOR | 1h |
| 3.11 | Comparison test: all 3 on same benchmark function, verify all find near-optimal | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] 25+ new tests pass
- [ ] GD converges on convex functions in <100 iterations
- [ ] GA solves 5-city TSP optimally
- [ ] SA escapes local minima (verified on multi-modal function)
- [ ] All 3 have spec + metadata JSON

### Exit Gate
`make test` passes with 65+ total tests. Optimization module complete.

---

## Phase 4: SAT Solver + CSP

**Goal:** First symbolic engine. SAT solver wrapping pysat, plus CSP backtracking. This is the first CoT-replacement capability.

**Depends on:** Phase 1
**Duration:** 4 days
**Deliverable:** SAT solver, CSP backtracker, both via solver protocol

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **SAT Solver** | | | |
| 4.1 | Write tests: simple SAT (3 vars), simple UNSAT, pigeonhole (known UNSAT), Sudoku encoding (9×9), model extraction, 1K variable random instance | RED | 2h |
| 4.2 | Implement `symbolic/sat.py` wrapping pysat (Glucose3 backend) | GREEN | 2h |
| 4.3 | Write tests: proof trace is returned, UNSAT produces unsat core | RED→GREEN | 1h |
| 4.4 | Spec + metadata JSON (include `replaces_cot_for` field) | Spec+Data | 30m |
| **CSP Backtracking** | | | |
| 4.5 | Write tests: 4-queens, graph coloring (3 colors, 5 nodes), map coloring (Australia), no-solution case | RED | 1h |
| 4.6 | Implement `csp/backtracking.py` with arc consistency (AC-3) + MRV heuristic | GREEN | 3h |
| 4.7 | Spec + metadata JSON | Spec+Data | 30m |
| **SAT-backed CSP** | | | |
| 4.8 | Write test: same CSP problems solved via SAT encoding (encode CSP as CNF, solve with SAT) | RED | 1h |
| 4.9 | Implement CSP-to-SAT encoder in `csp/sat_encoder.py` | GREEN | 2h |
| 4.10 | Comparison test: backtracking vs SAT on same problems, both produce valid solutions | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] SAT solves 9×9 Sudoku in <100ms
- [ ] SAT correctly returns UNSAT for pigeonhole
- [ ] CSP solves 8-queens in <1s
- [ ] CSP-to-SAT encoder produces same solutions as native backtracking
- [ ] 20+ new tests pass

### Exit Gate
`make test` passes with 85+ total tests. First symbolic engine operational.

---

## Phase 5: SMT + CAS (Symbolic Math)

**Goal:** SMT solver (Z3) and Computer Algebra System (SymPy). These handle the "LLM can't do math" problem.

**Depends on:** Phase 1
**Duration:** 4 days
**Deliverable:** SMT and CAS engines, both via solver protocol

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **SMT Solver** | | | |
| 5.1 | Write tests: linear integer (x+2y≤10), linear real (0.5x+0.3y=1), nonlinear (x²+y²≤100), UNSAT with unsat core, optimization (minimize x subject to constraints) | RED | 1.5h |
| 5.2 | Implement `symbolic/smt.py` wrapping z3-solver | GREEN | 3h |
| 5.3 | Write tests: SMT-LIB string input (parse and solve), model extraction as dict | RED→GREEN | 1h |
| 5.4 | Spec + metadata JSON | Spec+Data | 30m |
| **CAS (SymPy)** | | | |
| 5.5 | Write tests: solve quadratic, solve system of equations, simplify trig identity, differentiate, integrate, series expansion | RED | 1.5h |
| 5.6 | Implement `symbolic/cas.py` wrapping sympy | GREEN | 2h |
| 5.7 | Write tests: step-by-step mode returns intermediate steps, verify each step is algebraically correct | RED→GREEN | 1h |
| 5.8 | Spec + metadata JSON | Spec+Data | 30m |
| **Cross-engine** | | | |
| 5.9 | Write test: "find integer x,y where x²+y²=25" — solve via both SMT and CAS, results agree | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] SMT solves mixed integer/real constraints
- [ ] SMT optimization (minimize/maximize) works
- [ ] CAS solves quadratic with exact symbolic roots (not floats)
- [ ] CAS step-by-step produces verifiable intermediate steps
- [ ] 20+ new tests pass

### Exit Gate
`make test` passes with 105+ total tests. Math reasoning is symbolic, not hallucinated.

---

## Phase 6: Logic Programming + Rule Engine

**Goal:** Prolog-style reasoning and forward-chaining rule engine. Covers relational queries, policy checking, expert systems.

**Depends on:** Phase 1
**Duration:** 4 days
**Deliverable:** Prolog wrapper, rule engine, both via solver protocol

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **Prolog** | | | |
| 6.1 | Write tests: family tree queries (grandparent, ancestor), transitive closure, multiple solutions, no-solution | RED | 1h |
| 6.2 | Implement `symbolic/prolog.py` wrapping pyswip (SWI-Prolog) | GREEN | 2h |
| 6.3 | Write tests: returns all bindings, proof tree traceable | RED→GREEN | 1h |
| 6.4 | Spec + metadata JSON | Spec+Data | 30m |
| **Rule Engine** | | | |
| 6.5 | Write tests: medical diagnosis (fever+cough→flu), chain of rules (A→B, B→C, verify C derived), conflict resolution (priority), audit trail | RED | 1.5h |
| 6.6 | Implement `symbolic/rule_engine.py` — forward chaining with priority-based conflict resolution | GREEN | 3h |
| 6.7 | Write tests: audit trail records which rules fired in what order | RED→GREEN | 1h |
| 6.8 | Spec + metadata JSON | Spec+Data | 30m |
| **Cross-engine** | | | |
| 6.9 | Write test: same access-control policy solved by both Prolog (backward chaining) and rule engine (forward chaining), same conclusion | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] Prolog solves recursive ancestor queries
- [ ] Prolog returns all solutions (not just first)
- [ ] Rule engine fires chained rules correctly
- [ ] Audit trail is complete and ordered
- [ ] 20+ new tests pass

### Exit Gate
`make test` passes with 125+ total tests. Logic reasoning engines complete.

---

## Phase 7: Algorithm Selector + `solve()` API

**Goal:** The auto-selector that makes `solve("any problem description")` work. This is the "magic" layer.

**Depends on:** Phase 2, 3, 4, 5, 6 (needs all engines available)
**Duration:** 3 days
**Deliverable:** `from problemsolving import solve` auto-selects engine and returns result

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **Selector** | | | |
| 7.1 | Write `spec/decision_tree.json` — the algorithm selection decision tree | Spec | 1h |
| 7.2 | Write tests: 20 problem descriptions, each with expected engine selection (from decision tree) | RED | 2h |
| 7.3 | Implement `selector.py` — rule-based matcher using keyword extraction + decision tree traversal | GREEN | 3h |
| 7.4 | Write tests: selector returns confidence score, alternatives list, and reasoning string | RED→GREEN | 1h |
| **Top-level API** | | | |
| 7.5 | Write tests: `solve(problem_string)` returns correct result for BFS problem, SAT problem, math problem, Prolog problem | RED | 1h |
| 7.6 | Implement `__init__.py` top-level `solve()`, `verify()`, `select()` functions | GREEN | 2h |
| 7.7 | Write tests: `solve(problem, engine="astar")` forces engine override | RED→GREEN | 30m |
| 7.8 | Write tests: `verify(problem, solution)` checks correctness via symbolic engine | RED→GREEN | 1h |
| **Structured input** | | | |
| 7.9 | Write tests: `solve(problem, structured_input={...})` skips NL parsing, uses structured data directly | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] Selector correctly picks engine for 18/20 test problems (90%+ accuracy)
- [ ] `solve()` works end-to-end for all engine types
- [ ] `verify()` catches invalid solutions
- [ ] Engine override works
- [ ] 15+ new tests pass

### Exit Gate
`make test` passes with 140+ total tests. The library is usable as `pip install` + `from problemsolving import solve`.

---

## Phase 8: MCP Server

**Goal:** Expose everything as MCP tools. Any MCP client (Claude, GPT agents, custom) can call the toolkit.

**Depends on:** Phase 7
**Duration:** 5 days
**Deliverable:** `problemsolving serve --stdio` works with Claude Desktop

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **MCP Protocol** | | | |
| 8.1 | Write tests: server initializes, returns capabilities (tools + resources + prompts) | RED | 1h |
| 8.2 | Implement `mcp/server.py` — MCP server using `mcp` Python SDK, stdio transport | GREEN | 3h |
| **Tools** | | | |
| 8.3 | Write tests: `tools/list` returns all 5 tools (solve, verify, select_algorithm, translate, explain_algorithm) with correct schemas | RED | 1h |
| 8.4 | Implement `mcp/tools.py` — tool handlers that delegate to solver protocol | GREEN | 2h |
| 8.5 | Write tests: `tools/call` with `solve` tool — BFS problem → correct result in MCP response format | RED→GREEN | 1h |
| 8.6 | Write tests: `tools/call` with `solve` tool — SAT problem, math problem, Prolog problem | RED→GREEN | 1h |
| 8.7 | Write tests: `tools/call` with `verify` tool — valid and invalid solutions | RED→GREEN | 1h |
| 8.8 | Write tests: `tools/call` with `select_algorithm` tool | RED→GREEN | 30m |
| 8.9 | Write tests: `tools/call` with `translate` tool — NL → CNF, NL → SymPy expression | RED→GREEN | 1h |
| 8.10 | Write tests: `tools/call` with `explain_algorithm` tool — brief and detailed modes | RED→GREEN | 1h |
| **Resources** | | | |
| 8.11 | Write tests: `resources/list` returns algorithm registry and decision tree | RED | 30m |
| 8.12 | Implement `mcp/resources.py` — serve algorithm metadata as MCP resources | GREEN | 1h |
| 8.13 | Write tests: `resources/read` with `algorithm://astar/metadata` returns correct JSON | RED→GREEN | 30m |
| **Prompts** | | | |
| 8.14 | Write tests: `prompts/list` returns prompt templates | RED | 30m |
| 8.15 | Implement `mcp/prompts.py` — prompt templates for solve_problem, compare_approaches | GREEN | 1h |
| 8.16 | Write tests: `prompts/get` with arguments returns formatted prompt | RED→GREEN | 30m |
| **CLI** | | | |
| 8.17 | Implement `cli.py` — `problemsolving serve --stdio` and `problemsolving serve --http --port 3847` | GREEN | 1h |
| 8.18 | Write test: CLI spawns server, send JSON-RPC initialize, get capabilities back | RED→GREEN | 1h |
| **Integration** | | | |
| 8.19 | Write Claude Desktop config JSON for `claude_desktop_config.json` | Docs | 15m |
| 8.20 | Manual smoke test: connect Claude Desktop to MCP server, run 3 different problem types | Manual | 1h |

### Acceptance Criteria
- [ ] `problemsolving serve --stdio` starts and responds to MCP initialize
- [ ] All 5 tools callable and return correct results
- [ ] Resources serve algorithm metadata
- [ ] Prompts return formatted templates
- [ ] Claude Desktop can connect and use the `solve` tool
- [ ] 25+ new tests pass

### Exit Gate
`make test` passes with 165+ total tests. MCP server works with Claude Desktop.

---

## Phase 9: Game Theory + KG Reasoning + Remaining Algorithms

**Goal:** Fill in the remaining algorithm families. Smaller modules, same TDD pattern.

**Depends on:** Phase 1
**Duration:** 5 days
**Deliverable:** Game theory (minimax, alpha-beta), KG reasoning, expert systems, CBR

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **Game Theory** | | | |
| 9.1 | Write tests: minimax on tic-tac-toe (known optimal play), alpha-beta prunes correctly (count nodes), alpha-beta same result as minimax | RED | 1.5h |
| 9.2 | Implement `game_theory/minimax.py` and `game_theory/alpha_beta.py` | GREEN | 3h |
| 9.3 | Spec + metadata JSON for both | Spec+Data | 30m |
| **Knowledge Graph Reasoning** | | | |
| 9.4 | Write tests: RDFS subclass inference, SPARQL query on inferred graph, OWL consistency check | RED | 1.5h |
| 9.5 | Implement `symbolic/kg_reasoner.py` wrapping rdflib + owlready2 | GREEN | 3h |
| 9.6 | Spec + metadata JSON | Spec+Data | 30m |
| **Expert Systems** | | | |
| 9.7 | Write tests: forward chain medical diagnosis, fire priority rules, no infinite loops | RED | 1h |
| 9.8 | Implement `expert_systems/forward_chain.py` | GREEN | 2h |
| 9.9 | Spec + metadata JSON | Spec+Data | 30m |
| **CBR** | | | |
| 9.10 | Write tests: retrieve similar cases, adapt solution, retain new case | RED | 1h |
| 9.11 | Implement `cbr/case_based.py` | GREEN | 2h |
| 9.12 | Spec + metadata JSON | Spec+Data | 30m |
| **Theorem Prover (basic)** | | | |
| 9.13 | Write tests: prove simple propositional theorems via Z3, return proof | RED | 1h |
| 9.14 | Implement `symbolic/theorem_prover.py` wrapping Z3's proof mode | GREEN | 2h |
| 9.15 | Spec + metadata JSON | Spec+Data | 30m |
| **Integration** | | | |
| 9.16 | Update selector with new engines, write 10 more selection tests | RED→GREEN | 1h |
| 9.17 | Update MCP tools to support new engines (should just work via registry) | GREEN | 30m |

### Acceptance Criteria
- [ ] Minimax plays tic-tac-toe optimally (never loses)
- [ ] Alpha-beta explores fewer nodes than minimax, same result
- [ ] KG reasoner infers subclass relationships
- [ ] Expert system forward chains correctly with audit trail
- [ ] CBR retrieves and adapts from case base
- [ ] 30+ new tests pass

### Exit Gate
`make test` passes with 195+ total tests. All algorithm families implemented.

---

## Phase 10: Training Data Pipeline

**Goal:** Auto-capture training data from every solver invocation. Generate datasets for finetuning.

**Depends on:** Phase 7
**Duration:** 3 days
**Deliverable:** TrainingLogger, export to HuggingFace formats

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| 10.1 | Write tests: TrainingLogger captures problem + result as JSON, context manager works | RED | 1h |
| 10.2 | Implement `training/logger.py` | GREEN | 2h |
| 10.3 | Write tests: export to ShareGPT format, Alpaca format, OpenAI format | RED | 1h |
| 10.4 | Implement `training/formats.py` | GREEN | 2h |
| 10.5 | Write tests: generate synthetic problems for each engine type (at least 5 per engine) | RED | 1h |
| 10.6 | Implement `training/generator.py` — template-based problem generator | GREEN | 3h |
| 10.7 | Write test: generate 50 training examples, validate all are well-formed | RED→GREEN | 1h |
| 10.8 | Create `training_data/dataset_card.md` for HuggingFace | Docs | 30m |
| 10.9 | Write test: MCP server with `enable_training_log=True` captures all tool calls | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] Logger captures complete problem/solution/engine/trace tuples
- [ ] All 3 export formats produce valid JSON/JSONL
- [ ] Generator creates diverse problems across all engine types
- [ ] MCP training log works transparently
- [ ] 15+ new tests pass

### Exit Gate
`make test` passes with 210+ total tests. Can generate training datasets on demand.

---

## Phase 11: Agent Framework Integrations

**Goal:** First-class integrations with LangChain, OpenAI, Anthropic tool formats.

**Depends on:** Phase 7
**Duration:** 3 days
**Deliverable:** `from problemsolving.integrations.langchain import get_tools` works

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| 11.1 | Write tests: `get_tools()` returns list of LangChain `Tool` objects, each with name + description + schema | RED | 1h |
| 11.2 | Implement `integrations/langchain.py` | GREEN | 2h |
| 11.3 | Write tests: OpenAI function-calling schema validates against OpenAI spec | RED | 1h |
| 11.4 | Implement `integrations/openai.py` | GREEN | 1.5h |
| 11.5 | Write tests: Anthropic tool-use schema validates against Anthropic spec | RED | 1h |
| 11.6 | Implement `integrations/anthropic.py` | GREEN | 1.5h |
| 11.7 | Write tests: LlamaIndex tool wrappers | RED→GREEN | 1.5h |
| 11.8 | Write example: `examples/python/langchain_agent.py` — agent uses tools to solve problem | Example | 1h |
| 11.9 | Write example: `examples/python/verified_reasoning.py` — solve + verify loop | Example | 1h |

### Acceptance Criteria
- [ ] LangChain tools work in a LangChain agent
- [ ] OpenAI schemas valid per OpenAI function calling spec
- [ ] Anthropic schemas valid per Anthropic tool use spec
- [ ] Examples run end-to-end
- [ ] 10+ new tests pass

### Exit Gate
`make test` passes with 220+ total tests. Library integrates with all major agent frameworks.

---

## Phase 12: PyPI Release + Documentation

**Goal:** Ship v0.1.0 to PyPI. Comprehensive docs.

**Depends on:** Phase 8, 9, 10, 11
**Duration:** 4 days
**Deliverable:** `pip install problemsolving` works for everyone. Docs site live.

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| 12.1 | Set up MkDocs with material theme | Setup | 1h |
| 12.2 | Write `docs/getting-started.md` — install → first solve in 3 lines | Docs | 1h |
| 12.3 | Write `docs/mcp-integration.md` — Claude Desktop setup, full tool reference | Docs | 1.5h |
| 12.4 | Write `docs/api-reference/` — auto-generated from docstrings (mkdocstrings) | Docs | 1h |
| 12.5 | Write `docs/cookbook/sudoku.md` — SAT solver walkthrough | Docs | 1h |
| 12.6 | Write `docs/cookbook/math-proofs.md` — CAS walkthrough | Docs | 1h |
| 12.7 | Write `docs/cookbook/scheduling.md` — SMT walkthrough | Docs | 1h |
| 12.8 | Write `docs/cookbook/agent-reasoning.md` — solve+verify loop walkthrough | Docs | 1h |
| 12.9 | Create 4 Jupyter notebooks in `examples/notebooks/` | Docs | 3h |
| 12.10 | Set up PyPI publishing via GitHub Actions (on tag) | CI | 1h |
| 12.11 | Write comprehensive `README.md` for PyPI (badges, quick start, links) | Docs | 1h |
| 12.12 | Tag v0.1.0, publish to PyPI | Release | 30m |
| 12.13 | Deploy docs to GitHub Pages | Release | 30m |
| 12.14 | Smoke test: fresh virtualenv, `pip install problemsolving`, run getting-started code | Manual | 30m |

### Acceptance Criteria
- [ ] `pip install problemsolving` works from PyPI
- [ ] Docs site live with getting started + API reference + cookbook
- [ ] 4 Jupyter notebooks run cleanly
- [ ] MCP server launchable via `problemsolving serve`
- [ ] Fresh install smoke test passes

### Exit Gate
v0.1.0 on PyPI. Docs live. Library is publicly usable.

---

## Phase 13: Rust Core (Performance Layer)

**Goal:** Port performance-critical algorithms to Rust. Python SDK transparently uses Rust via PyO3.

**Depends on:** Phase 12 (stable Python API to preserve)
**Duration:** 8 days
**Deliverable:** Rust core crate, Python bindings via maturin, 5-50x speedup on core algorithms

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| **Setup** | | | |
| 13.1 | Create `core/Cargo.toml` with varisat, petgraph, pyo3 dependencies | Setup | 1h |
| 13.2 | Set up maturin build in `python/pyproject.toml` — Rust extension compiles on `pip install` | Setup | 2h |
| **Search (Rust)** | | | |
| 13.3 | Write Rust tests: BFS, A* on petgraph — same test cases as Python | RED | 2h |
| 13.4 | Implement `core/src/search/` — BFS, DFS, UCS, A*, Greedy using petgraph | GREEN | 4h |
| 13.5 | Write PyO3 bindings: Python calls Rust search, returns Python objects | GREEN | 2h |
| 13.6 | Write Python tests: existing search tests still pass, now backed by Rust | GREEN | 1h |
| 13.7 | Benchmark: Python vs Rust on 100K node graph. Assert Rust is 5x+ faster. | RED→GREEN | 1h |
| **SAT (Rust)** | | | |
| 13.8 | Write Rust tests: SAT solve, UNSAT detect, using varisat | RED | 1h |
| 13.9 | Implement `core/src/symbolic/sat.rs` wrapping varisat | GREEN | 2h |
| 13.10 | Write PyO3 bindings | GREEN | 1h |
| 13.11 | Python tests: SAT tests still pass, now backed by Rust | GREEN | 30m |
| 13.12 | Benchmark: Python (pysat) vs Rust (varisat) on 10K variable instance | RED→GREEN | 1h |
| **Optimization (Rust)** | | | |
| 13.13 | Implement GA + SA in Rust, PyO3 bindings | GREEN | 4h |
| 13.14 | Python tests still pass with Rust backend | GREEN | 30m |
| **Game Theory (Rust)** | | | |
| 13.15 | Implement minimax + alpha-beta in Rust, PyO3 bindings | GREEN | 3h |
| 13.16 | Python tests still pass with Rust backend | GREEN | 30m |
| **Fallback** | | | |
| 13.17 | Implement auto-detection: if Rust extension available, use it; else fall back to pure Python | GREEN | 1h |
| 13.18 | Test: `pip install problemsolving` (no Rust toolchain) still works (pure Python mode) | RED→GREEN | 30m |

### Acceptance Criteria
- [ ] All existing Python tests still pass (zero regressions)
- [ ] Rust backend 5x+ faster on search (100K nodes)
- [ ] Rust SAT backend 2x+ faster on large instances
- [ ] Pure Python fallback works when Rust extension not available
- [ ] maturin builds cross-platform wheels (Linux, macOS, Windows)

### Exit Gate
`make test` passes with same 220+ tests. Performance benchmarks show measurable speedup. v0.2.0 tagged.

---

## Phase 14: TypeScript SDK + WASM

**Goal:** TypeScript SDK backed by WASM-compiled Rust core.

**Depends on:** Phase 13
**Duration:** 5 days
**Deliverable:** `npm install problemsolving` works, WASM core for browser

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| 14.1 | Set up `typescript/package.json` with wasm-bindgen, vitest | Setup | 1h |
| 14.2 | Compile Rust core to WASM via wasm-pack | Setup | 2h |
| 14.3 | Write tests: BFS, A*, SAT via WASM — same test cases as Python/Rust | RED | 2h |
| 14.4 | Implement TypeScript wrappers over WASM bindings | GREEN | 3h |
| 14.5 | Write tests: `solve()`, `verify()`, `select()` top-level API | RED→GREEN | 1.5h |
| 14.6 | Implement MCP client helper (`mcp/client.ts`) for calling Python MCP server | GREEN | 2h |
| 14.7 | Write tests: MCP client connects to Python server, calls tools | RED→GREEN | 1.5h |
| 14.8 | Implement MCP server in TypeScript (Streamable HTTP transport) | GREEN | 3h |
| 14.9 | Write tests: TS MCP server responds to initialize + tools/call | RED→GREEN | 1h |
| 14.10 | Publish to npm | Release | 1h |

### Acceptance Criteria
- [ ] `npm install problemsolving` works
- [ ] WASM core runs in Node.js and browser
- [ ] TypeScript SDK passes equivalent tests to Python SDK
- [ ] MCP server (TS) works over Streamable HTTP
- [ ] Published to npm

### Exit Gate
npm package published. TypeScript SDK functional.

---

## Phase 15: Evaluation Benchmarks + Hardening

**Goal:** Comprehensive benchmark suite proving the library works. Performance baselines. Edge case coverage.

**Depends on:** Phase 12
**Duration:** 3 days
**Deliverable:** Benchmark suite, edge case tests, performance regression CI

### Tasks

| # | Task | Type | Est |
|---|------|------|-----|
| 15.1 | Write 50-problem evaluation benchmark (10 per category: search, optimization, SAT/CSP, math, logic) | Data | 3h |
| 15.2 | Write benchmark runner: solve all 50, score accuracy, log latency | GREEN | 2h |
| 15.3 | Write edge case tests: empty graph, single variable SAT, trivial equation, empty rule base | RED→GREEN | 2h |
| 15.4 | Write stress tests: 1M node graph, 100K variable SAT, 1000-equation system | RED→GREEN | 2h |
| 15.5 | Write selector accuracy evaluation: 100 problems, measure correct engine selection rate | RED→GREEN | 2h |
| 15.6 | Add benchmark to CI: fail if accuracy <90% or latency regresses >20% | CI | 1h |
| 15.7 | Write security tests: malformed input doesn't crash, timeouts work | RED→GREEN | 1h |

### Acceptance Criteria
- [ ] 50-problem benchmark: 95%+ solve rate
- [ ] Selector accuracy: 90%+ on 100-problem eval
- [ ] No crashes on malformed input
- [ ] Timeouts trigger cleanly
- [ ] CI prevents performance regressions

### Exit Gate
Benchmark suite in CI. All quality gates pass. Library is production-hardened.

---

## Summary: Milestone Timeline

```
Phase  │ Duration │ Tests │ Capability Unlocked
───────┼──────────┼───────┼──────────────────────────────────────
  0    │  1 day   │   1   │ Empty project, CI green
  1    │  3 days  │  15   │ Protocol + BFS (first vertical slice)
  2    │  4 days  │  40   │ All search algorithms
  3    │  4 days  │  65   │ Optimization algorithms
  4    │  4 days  │  85   │ SAT solver + CSP (first CoT replacement)
  5    │  4 days  │ 105   │ SMT + symbolic math
  6    │  4 days  │ 125   │ Prolog + rule engine
  7    │  3 days  │ 140   │ Auto-selector + solve() API ← usable library
  8    │  5 days  │ 165   │ MCP server ← agents can use it
  9    │  5 days  │ 195   │ All remaining algorithms
 10    │  3 days  │ 210   │ Training data pipeline
 11    │  3 days  │ 220   │ LangChain/OpenAI/Anthropic integrations
 12    │  4 days  │ 220   │ PyPI v0.1.0 + docs ← public launch
 13    │  8 days  │ 220   │ Rust core (5-50x speedup)
 14    │  5 days  │ 250+  │ TypeScript SDK + npm
 15    │  3 days  │ 300+  │ Benchmarks + hardening ← production ready
───────┼──────────┼───────┼──────────────────────────────────────
Total  │ ~63 days │ 300+  │ Full toolkit, 3 languages, MCP, training data
```

### Key Milestones

| Week | Milestone | You Can... |
|------|-----------|------------|
| 1 | Phase 0-1 done | Run BFS through the solver protocol |
| 2 | Phase 2-3 done | Solve search + optimization problems |
| 3 | Phase 4-5 done | Replace CoT with SAT/SMT/CAS |
| 4 | Phase 6-7 done | `pip install` and `solve("any problem")` |
| 5-6 | Phase 8-9 done | Claude/GPT agents use it via MCP |
| 7 | Phase 10-11 done | Generate training data, integrate with agent frameworks |
| 8 | Phase 12 done | **Public launch on PyPI** |
| 10 | Phase 13 done | 5-50x faster with Rust core |
| 12 | Phase 14-15 done | TypeScript SDK, benchmarks, production-ready |

---

## TDD Workflow Reminder

For every task in every phase:

```
1. Write the test FIRST
   - Test describes WHAT, not HOW
   - Test uses the public API (not internals)
   - Test covers: happy path, edge cases, error cases

2. Run the test — it MUST FAIL (RED)
   - If it passes, your test is wrong or the feature already exists

3. Write MINIMUM code to pass (GREEN)
   - No extra features
   - No premature optimization
   - No "while I'm here" additions

4. Refactor (REFACTOR)
   - Clean up duplication
   - Improve naming
   - Extract shared utilities
   - Tests must still pass after refactoring

5. Commit
   - One commit per RED→GREEN→REFACTOR cycle
   - Commit message: "feat(search): BFS finds shortest path [#1.8]"
```

### Test Naming Convention

```python
# test_{module}/test_{function}_{scenario}_{expected}

def test_bfs_simple_graph_finds_shortest_path():
def test_bfs_no_path_returns_none():
def test_bfs_cycle_does_not_infinite_loop():
def test_sat_pigeonhole_returns_unsat():
def test_selector_weighted_path_recommends_astar():
def test_mcp_solve_tool_returns_valid_response():
```

### Coverage Targets

| Phase | Line Coverage | Branch Coverage |
|-------|--------------|-----------------|
| 1-7 | 90%+ | 80%+ |
| 8 (MCP) | 85%+ | 75%+ |
| 9-11 | 90%+ | 80%+ |
| 12+ | 90%+ | 85%+ |
