# Architecture: ProblemSolving — Universal AI Reasoning Toolkit

## 1. Vision

A multi-language reasoning engine that any LLM, agent, or AI application can call to solve problems with formal guarantees. Instead of relying on chain-of-thought (fragile, hallucination-prone), applications delegate reasoning to symbolic engines that return provably correct results.

The library is consumed in three ways:

1. **As an import** — `from problemsolving import solve` / `import { solve } from "problemsolving"`
2. **As an MCP server** — any MCP-compatible client (Claude, GPT, custom agents) calls tools over JSON-RPC
3. **As training data** — structured problem/solution pairs for finetuning LLMs on algorithm selection

---

## 2. Requirements

### Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | Solve search problems (BFS, DFS, UCS, A*, Greedy) given a graph/state-space definition |
| FR2 | Solve optimization problems (gradient descent, genetic, simulated annealing) given an objective |
| FR3 | Solve constraint satisfaction problems via SAT/SMT solvers and backtracking |
| FR4 | Perform logical reasoning via Prolog-style resolution and rule engines |
| FR5 | Perform exact symbolic math via CAS (solve equations, simplify, differentiate, integrate) |
| FR6 | Reason over knowledge graphs (RDFS/OWL inference, SPARQL) |
| FR7 | Verify properties formally via theorem proving |
| FR8 | Auto-select the best algorithm/engine for a given problem description |
| FR9 | Expose all capabilities as MCP tools callable by any MCP client |
| FR10 | Provide multi-language SDKs: Python (primary), TypeScript, Rust (native) |
| FR11 | Generate structured training data from solver invocations |

### Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | Latency: p50 < 100ms for algorithm selection, p99 < 5s for most solver runs |
| NFR2 | SAT/SMT: handle 100K+ variable problems |
| NFR3 | Zero required external services — runs fully local |
| NFR4 | Python SDK installable via single `pip install problemsolving` |
| NFR5 | MCP server launchable via single command: `problemsolving serve` |
| NFR6 | Every solver result includes a verification certificate or proof trace |
| NFR7 | Pluggable backends — swap SAT solver, CAS engine, etc. without API changes |

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONSUMERS                                     │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Claude   │  │ GPT /    │  │ Custom   │  │ Direct SDK Import  │  │
│  │ Desktop  │  │ OpenAI   │  │ Agents   │  │ (Python/TS/Rust)   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬───────────┘  │
│       │              │              │                  │              │
│       │     MCP (JSON-RPC)          │           Native function call │
│       │              │              │                  │              │
└───────┼──────────────┼──────────────┼──────────────────┼─────────────┘
        │              │              │                  │
        ▼              ▼              ▼                  │
┌──────────────────────────────────────────┐            │
│           MCP SERVER LAYER               │            │
│                                          │            │
│  Transport: stdio | Streamable HTTP      │            │
│  Protocol:  JSON-RPC 2.0                 │            │
│                                          │            │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐  │            │
│  │  Tools  │ │Resources │ │ Prompts  │  │            │
│  │ (solve, │ │(algorithm│ │(problem  │  │            │
│  │ select, │ │ metadata,│ │ templates│  │            │
│  │ verify) │ │ schemas) │ │ & guides)│  │            │
│  └────┬────┘ └────┬─────┘ └────┬─────┘  │            │
│       │           │             │         │            │
└───────┼───────────┼─────────────┼─────────┘            │
        │           │             │                      │
        ▼           ▼             ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SOLVER PROTOCOL LAYER                          │
│                                                                   │
│  Standardized JSON request/response envelope for all operations   │
│  Every call produces: result + proof_trace + metadata              │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Router / Selector                        │  │
│  │  Input: problem description (NL or structured)             │  │
│  │  Output: recommended engine + formatted input              │  │
│  └──────────┬────────────┬────────────┬───────────┬───────────┘  │
│             │            │            │           │               │
│             ▼            ▼            ▼           ▼               │
│  ┌──────────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐    │
│  │   Classical  │ │ Symbolic  │ │  Math    │ │ Knowledge  │    │
│  │  Algorithms  │ │  Logic    │ │  Engine  │ │  Reasoning │    │
│  │             │ │           │ │          │ │            │    │
│  │ Search      │ │ SAT/CDCL  │ │ CAS      │ │ KG/OWL     │    │
│  │ Optimization│ │ SMT/Z3    │ │ Equation │ │ SPARQL     │    │
│  │ CSP         │ │ Prolog    │ │ Calculus │ │ RDFS       │    │
│  │ Game Theory │ │ Rules/Rete│ │ Simplify │ │ Ontology   │    │
│  │ ML basics   │ │ Theorem   │ │ Verify   │ │ Graph Walk │    │
│  └──────────────┘ └───────────┘ └──────────┘ └────────────┘    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
        │            │            │           │
        ▼            ▼            ▼           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    BACKEND LAYER (Pluggable)                      │
│                                                                   │
│  Rust native:  varisat (SAT), egg (rewriting), petgraph (graphs) │
│  C/C++ binds:  Z3 (SMT), CaDiCaL (SAT)                          │
│  Python:       sympy (CAS), pyswip (Prolog), rdflib (KG)         │
│  WASM:         browser-compatible builds of core solvers          │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Layer Responsibilities

**MCP Server Layer** — Protocol adapter. Translates between MCP's JSON-RPC transport and the internal solver protocol. Exposes tools, resources, and prompts. Handles MCP lifecycle (initialize, capabilities negotiation, tool listing). Manages long-running solver tasks via MCP Tasks primitive.

**Solver Protocol Layer** — The core brain. Contains the router/selector that decides which engine to use, formats inputs for each engine, collects results, and produces unified response envelopes. This layer is the single source of truth — both the MCP server and direct SDK imports call into it.

**Backend Layer** — Pluggable solver implementations. Each engine has a defined interface; backends are swappable. The default backends ship with the package, but users can register custom backends (e.g., a commercial SAT solver, a specialized theorem prover).

---

## 4. The Solver Protocol

The solver protocol is the core abstraction. Every operation — whether called via MCP, Python import, or TypeScript — goes through this protocol. It's a simple JSON envelope.

### 4.1 Request Format

```json
{
  "id": "req_abc123",
  "engine": "sat",
  "operation": "solve",
  "input": {
    "clauses": [[1, -2, 3], [-1, 2], [-3, 1]],
    "num_variables": 3
  },
  "config": {
    "timeout_ms": 5000,
    "backend": "auto",
    "produce_proof": true
  }
}
```

### 4.2 Response Format

```json
{
  "id": "req_abc123",
  "status": "success",
  "engine": "sat",
  "backend": "varisat",
  "result": {
    "satisfiable": true,
    "model": { "1": true, "2": false, "3": true }
  },
  "proof_trace": [
    { "step": 1, "action": "unit_propagate", "literal": 1 },
    { "step": 2, "action": "decide", "literal": -2 },
    { "step": 3, "action": "unit_propagate", "literal": 3 }
  ],
  "metadata": {
    "time_ms": 12,
    "nodes_explored": 47,
    "backend_version": "0.2.6"
  }
}
```

### 4.3 Error Format

```json
{
  "id": "req_abc123",
  "status": "error",
  "error": {
    "code": "TIMEOUT",
    "message": "Solver exceeded 5000ms time limit",
    "partial_result": null,
    "suggestion": "Try increasing timeout or reducing problem size"
  }
}
```

### 4.4 Engine Registry

Every engine registers itself with the protocol layer:

```python
@register_engine
class SATEngine:
    name = "sat"
    version = "1.0.0"
    operations = ["solve", "check", "enumerate"]
    input_schema = SATInputSchema       # JSON Schema
    output_schema = SATOutputSchema
    backends = ["varisat", "z3_sat", "cadical"]
    default_backend = "varisat"

    tags = ["boolean", "satisfiability", "constraint", "feasibility"]
    replaces_cot_for = [
        "boolean puzzles",
        "scheduling feasibility",
        "configuration checking",
        "combinatorial problems"
    ]
```

### 4.5 Protocol Operations (All Engines)

| Operation | Description |
|-----------|-------------|
| `solve` | Find a solution (model, path, assignment, proof) |
| `check` | Verify a proposed solution is correct |
| `enumerate` | Find all solutions (or up to N) |
| `optimize` | Find the best solution given an objective |
| `explain` | Return human-readable explanation of algorithm/engine behavior |
| `translate` | Convert natural language problem to formal representation |

---

## 5. MCP Server Design

### 5.1 Server Identity & Capabilities

```json
{
  "name": "problemsolving",
  "version": "1.0.0",
  "capabilities": {
    "tools": {},
    "resources": { "subscribe": true },
    "prompts": {},
    "tasks": {}
  }
}
```

### 5.2 MCP Tools

Tools are the primary interface. Each tool maps to a solver protocol operation.

#### Tool: `solve`

The main tool. Accepts a problem and returns a solution.

```json
{
  "name": "solve",
  "description": "Solve a problem using the optimal algorithm or symbolic engine. Supports search, optimization, constraint satisfaction, logical reasoning, symbolic math, knowledge graph queries, and formal verification. Automatically selects the best engine, or accepts an explicit engine choice.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "problem": {
        "type": "string",
        "description": "Natural language problem description. The system will parse this and route to the appropriate engine."
      },
      "engine": {
        "type": "string",
        "enum": ["auto", "bfs", "dfs", "ucs", "astar", "greedy", "gradient_descent", "genetic", "simulated_annealing", "backtracking", "sat", "smt", "prolog", "rule_engine", "cas", "kg_reasoner", "theorem_prover"],
        "description": "Explicit engine selection. Use 'auto' (default) to let the system choose."
      },
      "structured_input": {
        "type": "object",
        "description": "Optional structured problem data (graph, clauses, equations, etc.). If provided, skips natural language parsing."
      },
      "config": {
        "type": "object",
        "properties": {
          "timeout_ms": { "type": "integer", "default": 30000 },
          "produce_proof": { "type": "boolean", "default": true },
          "max_solutions": { "type": "integer", "default": 1 }
        }
      }
    },
    "required": ["problem"]
  },
  "annotations": {
    "title": "Problem Solver",
    "readOnlyHint": true,
    "openWorldHint": false
  }
}
```

#### Tool: `verify`

Verify a proposed solution against constraints.

```json
{
  "name": "verify",
  "description": "Verify that a proposed solution is correct for a given problem. Uses symbolic engines to formally check the solution rather than heuristic validation.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "problem": {
        "type": "string",
        "description": "The original problem statement"
      },
      "proposed_solution": {
        "type": "object",
        "description": "The solution to verify"
      },
      "verification_method": {
        "type": "string",
        "enum": ["auto", "sat_check", "smt_check", "symbolic_eval", "proof_check"],
        "default": "auto"
      }
    },
    "required": ["problem", "proposed_solution"]
  },
  "annotations": {
    "title": "Solution Verifier",
    "readOnlyHint": true,
    "openWorldHint": false
  }
}
```

#### Tool: `select_algorithm`

Recommend the best algorithm/engine without solving.

```json
{
  "name": "select_algorithm",
  "description": "Given a problem description, recommend the best algorithm or symbolic engine to use. Returns the recommendation with reasoning, complexity analysis, and alternative options.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "problem": { "type": "string" },
      "constraints": {
        "type": "object",
        "properties": {
          "needs_optimal": { "type": "boolean" },
          "needs_proof": { "type": "boolean" },
          "max_time_ms": { "type": "integer" },
          "explainability_required": { "type": "boolean" }
        }
      }
    },
    "required": ["problem"]
  },
  "annotations": {
    "title": "Algorithm Selector",
    "readOnlyHint": true,
    "openWorldHint": false
  }
}
```

#### Tool: `translate`

Convert natural language to formal representation (for advanced users who want to inspect/modify the formal input before solving).

```json
{
  "name": "translate",
  "description": "Translate a natural language problem description into a formal representation (SAT clauses, SMT-LIB, Prolog facts/rules, SymPy expressions, SPARQL query, etc.) without solving. Useful for inspection, debugging, or manual modification before solving.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "problem": { "type": "string" },
      "target_format": {
        "type": "string",
        "enum": ["cnf", "smt_lib", "prolog", "sympy", "sparql", "auto"]
      }
    },
    "required": ["problem"]
  },
  "annotations": {
    "title": "Problem Translator",
    "readOnlyHint": true,
    "openWorldHint": false
  }
}
```

#### Tool: `explain_algorithm`

Teach about an algorithm (for educational use and agent context-building).

```json
{
  "name": "explain_algorithm",
  "description": "Get a detailed explanation of a specific algorithm or symbolic engine, including when to use it, complexity, pseudocode, and examples.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "algorithm": {
        "type": "string",
        "description": "Algorithm name (e.g., 'astar', 'sat', 'minimax')"
      },
      "detail_level": {
        "type": "string",
        "enum": ["brief", "detailed", "with_code", "comparison"],
        "default": "detailed"
      },
      "compare_with": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Optional: list of algorithms to compare against"
      }
    },
    "required": ["algorithm"]
  },
  "annotations": {
    "title": "Algorithm Explainer",
    "readOnlyHint": true,
    "openWorldHint": false
  }
}
```

### 5.3 MCP Resources

Resources provide read access to algorithm metadata and knowledge base.

```json
[
  {
    "uri": "algorithm://registry",
    "name": "Algorithm Registry",
    "description": "List of all available algorithms and engines with metadata",
    "mimeType": "application/json"
  },
  {
    "uri": "algorithm://decision-tree",
    "name": "Algorithm Decision Tree",
    "description": "Decision tree for selecting the right algorithm given problem characteristics",
    "mimeType": "application/json"
  },
  {
    "uri": "algorithm://{id}/schema",
    "name": "Algorithm Input Schema",
    "description": "JSON Schema for a specific algorithm's input format",
    "mimeType": "application/json"
  },
  {
    "uri": "algorithm://{id}/metadata",
    "name": "Algorithm Metadata",
    "description": "Full metadata for a specific algorithm (complexity, use cases, tags, etc.)",
    "mimeType": "application/json"
  }
]
```

Resource templates allow dynamic lookup:

```json
{
  "uriTemplate": "algorithm://{algorithm_id}/metadata",
  "name": "Algorithm Metadata",
  "description": "Metadata for a specific algorithm. Use algorithm://registry to list available IDs."
}
```

### 5.4 MCP Prompts

Pre-built prompt templates that help LLMs use the tools effectively.

```json
[
  {
    "name": "solve_problem",
    "description": "Guide for solving a problem: translate → select engine → solve → verify → explain",
    "arguments": [
      { "name": "problem", "description": "The problem to solve", "required": true },
      { "name": "style", "description": "Output style: concise | detailed | educational", "required": false }
    ]
  },
  {
    "name": "compare_approaches",
    "description": "Compare CoT reasoning vs symbolic solving for a given problem",
    "arguments": [
      { "name": "problem", "description": "The problem to compare approaches on", "required": true }
    ]
  },
  {
    "name": "build_training_example",
    "description": "Generate an instruction-tuning example from a solved problem",
    "arguments": [
      { "name": "problem", "description": "The source problem", "required": true },
      { "name": "format", "description": "alpaca | sharegpt | openai", "required": false }
    ]
  }
]
```

### 5.5 MCP Tasks (Async / Long-Running)

For complex solver invocations that may take >30 seconds:

```
Client                          MCP Server
  │                                  │
  │  tools/call {solve, big_sat}     │
  │ ────────────────────────────►    │
  │                                  │  Start solver in background
  │  ◄──── result: task_id=T1       │
  │                                  │
  │  tasks/get {T1}                  │
  │ ────────────────────────────►    │
  │                                  │
  │  ◄──── status: running,          │
  │        progress: 45%,            │
  │        nodes_explored: 120000    │
  │                                  │
  │  ... (poll or subscribe) ...     │
  │                                  │
  │  ◄──── status: completed,        │
  │        result: {sat, model}      │
  │                                  │
```

Task lifecycle:
- `tools/call` with large problem → server returns task handle immediately
- Client polls `tasks/get` or subscribes to `tasks/subscribe` for updates
- Server publishes progress (percentage, intermediate results, nodes explored)
- On completion: full result with proof trace
- On timeout/cancel: partial result if available

### 5.6 Transport

| Transport | Use Case | Config |
|-----------|----------|--------|
| stdio | Local agent (Claude Code, CLI tools) | `problemsolving serve --stdio` |
| Streamable HTTP | Remote agents, web apps, multi-client | `problemsolving serve --http --port 3847` |

Both transports support the full protocol. stdio is preferred for single-client, local scenarios (lower latency, simpler setup). Streamable HTTP for multi-client or remote use.

---

## 6. Language SDK Architecture

### 6.1 Build Strategy

```
                    ┌──────────────────────────┐
                    │     Rust Core Crate       │
                    │   problemsolving-core     │
                    │                           │
                    │  • All classical algos    │
                    │  • SAT solver (varisat)   │
                    │  • Graph engine (petgraph)│
                    │  • Protocol layer         │
                    │  • Engine registry        │
                    │  • Router/selector        │
                    └─────┬──────────┬──────────┘
                          │          │
              ┌───────────┘          └────────────┐
              │                                    │
              ▼                                    ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│    Python SDK            │        │   TypeScript SDK          │
│  problemsolving (PyPI)   │        │  problemsolving (npm)     │
│                          │        │                           │
│  PyO3/maturin bindings   │        │  WASM (wasm-bindgen)      │
│  to Rust core            │        │  to Rust core             │
│                          │        │                           │
│  + Native Python for:    │        │  + Native TS for:         │
│    • sympy (CAS)         │        │    • MCP client helpers   │
│    • pyswip (Prolog)     │        │    • Agent framework glue │
│    • rdflib (KG)         │        │    • Browser demos        │
│    • z3-solver (SMT)     │        │                           │
│    • MCP server (stdio)  │        │  + MCP server (HTTP)      │
│    • Training data gen   │        │                           │
│                          │        │                           │
│  Hybrid: Rust speed for  │        │  WASM for core algos,     │
│  core algos, Python for  │        │  native JS for ecosystem  │
│  ecosystem integrations  │        │  integrations             │
└──────────────────────────┘        └──────────────────────────┘
```

### 6.2 What Lives Where

| Component | Language | Reason |
|-----------|----------|--------|
| Search algorithms (BFS, A*, etc.) | Rust | Performance on large graphs |
| Optimization (GA, SA) | Rust | Hot loops, parallelizable |
| SAT solver | Rust (varisat) | Performance-critical, existing Rust crate |
| CSP backtracking | Rust | Deep recursion + pruning benefits from Rust |
| Game theory (minimax, alpha-beta) | Rust | Deep tree search |
| SMT solver | Python wraps Z3 | Z3 has best Python bindings, no need to rewrite |
| CAS (symbolic math) | Python wraps SymPy | SymPy is the gold standard, no competing Rust CAS |
| Prolog engine | Python wraps SWI-Prolog | Mature engine, FFI is fine |
| KG reasoning | Python wraps rdflib/owlready2 | RDF ecosystem is Python-centric |
| Theorem proving | Python wraps Z3/Lean | Existing mature tools |
| Rule engine (Rete) | Rust | Pattern matching + memory efficiency |
| Router / Selector | Rust + Python | Rust for scoring, Python for NL understanding |
| MCP server | Python (primary) + TS | Python for stdio, TS for HTTP/browser |
| Protocol layer | Rust (shared) | Single source of truth for request/response |
| Training data gen | Python | HuggingFace ecosystem, data processing |

### 6.3 Python SDK API Design

```python
# ── Zero-Config Usage ────────────────────────────────────
from problemsolving import solve, verify, select

# Natural language (auto-selects engine)
result = solve("Find shortest path from A to E in this weighted graph",
               graph={"A-B": 4, "A-C": 2, "B-D": 3, "C-D": 1, "D-E": 5})
# → {engine: "astar", path: ["A","C","D","E"], cost: 8, proof_trace: [...]}

# Verify a proposed answer
ok = verify("Is this a valid 3-coloring?",
            proposed={"A": "red", "B": "blue", "C": "red"},
            constraints=graph)
# → {valid: True, proof: "SAT check passed"}

# Just get a recommendation
rec = select("Schedule 50 nurses across 3 shifts with constraints")
# → {engine: "smt", reason: "Large CSP with numeric constraints", alternatives: ["sat", "genetic"]}


# ── Explicit Engine Usage ────────────────────────────────
from problemsolving.search import astar, bfs
from problemsolving.symbolic import sat, smt, cas, prolog

# Direct SAT solving
result = sat.solve(clauses=[[1, -2], [-1, 3], [2, -3]],
                   num_vars=3)

# Direct symbolic math
from problemsolving.symbolic import cas
result = cas.solve("x**2 + 3*x - 10 = 0", variable="x")
# → {solutions: [2, -5], steps: [...]}

result = cas.differentiate("sin(x) * e^x", variable="x")
# → {result: "e^x*(sin(x) + cos(x))", steps: [...]}

# Direct Prolog
result = prolog.query(
    facts=["parent(tom, bob)", "parent(bob, ann)"],
    rules=["grandparent(X, Z) :- parent(X, Y), parent(Y, Z)"],
    query="grandparent(tom, Who)"
)
# → {bindings: [{"Who": "ann"}]}


# ── MCP Server ───────────────────────────────────────────
# Launch from CLI:
#   $ problemsolving serve --stdio
#   $ problemsolving serve --http --port 3847

# Or programmatically:
from problemsolving.mcp import create_server
server = create_server(
    engines=["all"],           # or subset: ["sat", "smt", "cas"]
    transport="stdio",         # or "http"
    enable_tasks=True,         # async long-running solves
    enable_training_log=True   # log all calls as training data
)
server.run()


# ── Agent Framework Integration ──────────────────────────
# LangChain
from problemsolving.integrations.langchain import get_tools
tools = get_tools()  # returns list of LangChain Tool objects

# OpenAI function calling
from problemsolving.integrations.openai import get_function_schemas
schemas = get_function_schemas()  # returns OpenAI-format tool definitions

# Anthropic tool use
from problemsolving.integrations.anthropic import get_tool_definitions
tools = get_tool_definitions()  # returns Anthropic-format tool definitions


# ── Verification Loop (Agent Pattern) ────────────────────
from problemsolving import solve, verify

def verified_solve(problem, max_attempts=3):
    """Solve with formal verification — the core CoT replacement pattern."""
    for attempt in range(max_attempts):
        result = solve(problem)
        check = verify(problem, result.solution)

        if check.valid:
            return result  # formally verified

        # Feed counterexample back for re-attempt
        problem = problem.with_feedback(check.counterexample)

    raise SolverError("Could not find verified solution", attempts=max_attempts)


# ── Training Data Generation ─────────────────────────────
from problemsolving.training import TrainingLogger

logger = TrainingLogger(format="sharegpt", output="training_data.jsonl")

# Wrap any solve call to auto-generate training examples
with logger:
    result = solve("Find all solutions to x^2 - 5x + 6 = 0")
    # Logger captures: {problem, engine_selected, formal_input,
    #                    formal_output, natural_explanation}
    # and writes as instruction-tuning pair


# ── Plugin System ─────────────────────────────────────────
from problemsolving import register_backend

@register_backend(engine="sat")
class MyCustomSATSolver:
    name = "my_solver"
    def solve(self, clauses, config):
        # custom implementation
        ...
```

### 6.4 TypeScript SDK API Design

```typescript
// ── Zero-Config Usage ────────────────────────────────────
import { solve, verify, select } from "problemsolving";

const result = await solve({
  problem: "Find shortest path from A to E",
  structuredInput: {
    graph: { "A-B": 4, "A-C": 2, "B-D": 3, "C-D": 1, "D-E": 5 }
  }
});

// ── Explicit Engine Usage ────────────────────────────────
import { sat, cas } from "problemsolving/symbolic";

const satResult = await sat.solve({
  clauses: [[1, -2], [-1, 3], [2, -3]],
  numVars: 3
});

const mathResult = await cas.solve("x**2 + 3*x - 10 = 0", "x");

// ── MCP Client (connect to running server) ───────────────
import { MCPClient } from "problemsolving/mcp";

const client = new MCPClient("http://localhost:3847");
const tools = await client.listTools();
const result = await client.callTool("solve", {
  problem: "Is this schedule feasible?",
  engine: "sat"
});

// ── MCP Server (TypeScript implementation) ───────────────
import { createServer } from "problemsolving/mcp-server";

const server = createServer({
  transport: "http",
  port: 3847,
  engines: ["all"]
});
server.start();
```

### 6.5 Rust SDK (Native)

```rust
use problemsolving::{solve, Engine, Config};
use problemsolving::search::astar;
use problemsolving::symbolic::sat;

// Direct algorithm use
let result = astar::solve(AStarInput {
    start: "A",
    goal: |s| s == "E",
    neighbors: &graph,
    heuristic: &h,
})?;

// SAT solving
let result = sat::solve(SATInput {
    clauses: vec![vec![1, -2], vec![-1, 3]],
    num_vars: 3,
    config: Config::default(),
})?;

// Auto-selection
let result = solve("Find shortest path...", &structured_input)?;
```

---

## 7. Router / Algorithm Selector

The selector is a key differentiator. It takes a problem description and returns the best engine.

### 7.1 Selection Logic

```
Input: problem description (NL or structured)
         │
         ├─── Feature Extraction ───┐
         │    • keywords             │
         │    • problem structure    │
         │    • constraint types     │
         │    • variable types       │
         │    • scale estimate       │
         │                           │
         ▼                           ▼
    ┌──────────┐              ┌──────────────┐
    │ Rule-    │              │ Embedding    │
    │ Based    │              │ Similarity   │
    │ Matcher  │              │ (fast, for   │
    │ (fast,   │              │ NL problems) │
    │ precise) │              │              │
    └────┬─────┘              └──────┬───────┘
         │                           │
         └────────┬──────────────────┘
                  │
                  ▼
         ┌───────────────┐
         │  Score &      │
         │  Rank         │
         │  Candidates   │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │  Return:      │
         │  • engine     │
         │  • confidence │
         │  • reasoning  │
         │  • alternates │
         └───────────────┘
```

### 7.2 Selection Rules (Subset)

```python
SELECTION_RULES = [
    # Highest priority: structural matches
    Rule("boolean variables + satisfiability",        → "sat",    priority=100),
    Rule("integer/real constraints + feasibility",    → "smt",    priority=100),
    Rule("equations + solve/simplify/differentiate",  → "cas",    priority=100),
    Rule("knowledge graph + query",                   → "kg",     priority=100),
    Rule("prove/verify/theorem",                      → "theorem",priority=100),

    # Medium priority: problem-type matches
    Rule("shortest path + unweighted",                → "bfs",    priority=80),
    Rule("shortest path + weighted + heuristic",      → "astar",  priority=80),
    Rule("shortest path + weighted + no heuristic",   → "ucs",    priority=80),
    Rule("constraint satisfaction + small",            → "backtrack", priority=80),
    Rule("constraint satisfaction + large",            → "sat",    priority=80),
    Rule("optimize + continuous + differentiable",     → "gradient", priority=80),
    Rule("optimize + discrete/black-box",             → "genetic", priority=80),
    Rule("two-player + zero-sum",                     → "minimax", priority=80),
    Rule("rules + facts + inference",                 → "prolog",  priority=80),
    Rule("if-then rules + forward reasoning",         → "rule_engine", priority=80),

    # Low priority: fallbacks
    Rule("optimization + any",                        → "simulated_annealing", priority=40),
    Rule("search + any",                              → "bfs",    priority=40),
]
```

---

## 8. Package Structure

### 8.1 Monorepo Layout

```
problemsolving/
├── README.md
├── LICENSE                          # Apache 2.0
├── ARCHITECTURE.md                  # This document
├── ROADMAP.md
│
├── spec/                            # THE SPECIFICATION (language-agnostic)
│   ├── protocol.json                # Solver protocol JSON Schema
│   ├── engines/                     # Per-engine schemas
│   │   ├── sat.json
│   │   ├── smt.json
│   │   ├── cas.json
│   │   ├── prolog.json
│   │   ├── kg.json
│   │   ├── astar.json
│   │   └── ...
│   ├── mcp/                         # MCP tool/resource/prompt definitions
│   │   ├── tools.json
│   │   ├── resources.json
│   │   └── prompts.json
│   └── decision_tree.json           # Algorithm selection logic
│
├── core/                            # RUST CORE
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── protocol/               # Solver protocol implementation
│   │   │   ├── mod.rs
│   │   │   ├── request.rs
│   │   │   ├── response.rs
│   │   │   └── registry.rs
│   │   ├── search/                  # Classical search algorithms
│   │   │   ├── mod.rs
│   │   │   ├── bfs.rs
│   │   │   ├── dfs.rs
│   │   │   ├── astar.rs
│   │   │   ├── ucs.rs
│   │   │   └── greedy.rs
│   │   ├── optimization/
│   │   │   ├── mod.rs
│   │   │   ├── gradient_descent.rs
│   │   │   ├── genetic.rs
│   │   │   └── simulated_annealing.rs
│   │   ├── csp/
│   │   │   ├── mod.rs
│   │   │   └── backtracking.rs
│   │   ├── game_theory/
│   │   │   ├── mod.rs
│   │   │   ├── minimax.rs
│   │   │   └── alpha_beta.rs
│   │   ├── symbolic/
│   │   │   ├── mod.rs
│   │   │   ├── sat.rs              # Wraps varisat
│   │   │   └── rule_engine.rs      # Rete implementation
│   │   ├── selector/
│   │   │   ├── mod.rs
│   │   │   ├── rules.rs
│   │   │   └── scorer.rs
│   │   └── traits.rs               # Shared Engine, Solver, Problem traits
│   └── tests/
│
├── python/                          # PYTHON SDK
│   ├── pyproject.toml               # maturin build config
│   ├── src/
│   │   └── problemsolving/
│   │       ├── __init__.py          # solve(), verify(), select()
│   │       ├── _core.pyi            # Type stubs for Rust bindings
│   │       ├── search/              # Thin wrappers over Rust core
│   │       ├── optimization/
│   │       ├── csp/
│   │       ├── game_theory/
│   │       ├── symbolic/
│   │       │   ├── __init__.py
│   │       │   ├── sat.py           # Calls Rust core
│   │       │   ├── smt.py           # Wraps z3-solver (Python)
│   │       │   ├── cas.py           # Wraps sympy (Python)
│   │       │   ├── prolog.py        # Wraps pyswip (Python)
│   │       │   ├── kg.py            # Wraps rdflib (Python)
│   │       │   ├── theorem.py       # Wraps z3/lean (Python)
│   │       │   └── rule_engine.py   # Calls Rust core
│   │       ├── mcp/
│   │       │   ├── __init__.py
│   │       │   ├── server.py        # MCP server implementation
│   │       │   ├── tools.py         # Tool handlers
│   │       │   ├── resources.py     # Resource handlers
│   │       │   └── prompts.py       # Prompt handlers
│   │       ├── integrations/
│   │       │   ├── langchain.py
│   │       │   ├── openai.py
│   │       │   ├── anthropic.py
│   │       │   └── llamaindex.py
│   │       ├── training/
│   │       │   ├── logger.py        # Auto-capture training data
│   │       │   ├── formats.py       # ShareGPT, Alpaca, OpenAI
│   │       │   └── generator.py     # Synthetic data generation
│   │       ├── selector.py          # Algorithm selector (Python layer)
│   │       └── cli.py               # CLI entry point
│   └── tests/
│
├── typescript/                      # TYPESCRIPT SDK
│   ├── package.json
│   ├── src/
│   │   ├── index.ts
│   │   ├── core/                    # WASM bindings to Rust core
│   │   ├── symbolic/               # Native TS wrappers
│   │   ├── mcp/
│   │   │   ├── server.ts           # MCP server (Streamable HTTP)
│   │   │   └── client.ts           # MCP client helper
│   │   └── integrations/
│   │       └── vercel-ai.ts        # Vercel AI SDK integration
│   └── tests/
│
├── knowledge/                       # STRUCTURED KNOWLEDGE BASE
│   ├── algorithms/                  # Per-algorithm metadata (JSON)
│   │   ├── bfs.json
│   │   ├── astar.json
│   │   ├── sat.json
│   │   └── ...
│   ├── categories.json
│   ├── schema.json                  # JSON Schema for algorithm metadata
│   └── decision_tree.json
│
├── training_data/                   # TRAINING & EVALUATION DATA
│   ├── instruction_tuning/
│   ├── preference/
│   ├── symbolic_traces/             # Formal reasoning traces (CoT replacement)
│   ├── evaluation/
│   └── dataset_card.md              # HuggingFace dataset card
│
├── docs/                            # DOCUMENTATION
│   ├── mkdocs.yml
│   ├── getting-started.md
│   ├── mcp-integration.md
│   ├── cookbook/
│   │   ├── sudoku.md
│   │   ├── scheduling.md
│   │   ├── math-proofs.md
│   │   └── agent-reasoning.md
│   └── api-reference/
│
└── examples/
    ├── python/
    │   ├── quick_start.py
    │   ├── mcp_server.py
    │   ├── langchain_agent.py
    │   ├── verified_reasoning.py
    │   └── training_data_gen.py
    ├── typescript/
    │   ├── quick_start.ts
    │   └── mcp_client.ts
    └── notebooks/
        ├── 01_search_algorithms.ipynb
        ├── 02_symbolic_reasoning.ipynb
        ├── 03_cot_vs_symbolic.ipynb
        └── 04_training_data.ipynb
```

---

## 9. Data Flow Examples

### 9.1 Agent Solves a Math Problem via MCP

```
1. User → Agent: "What are the roots of 2x³ - 3x² - 11x + 6?"

2. Agent → MCP Server (tools/call):
   {
     "tool": "solve",
     "arguments": {
       "problem": "Find roots of 2x³ - 3x² - 11x + 6 = 0",
       "engine": "auto"
     }
   }

3. MCP Server → Selector:
   "polynomial equation solving" → score engines
   → cas: 0.95 (keyword: roots, equation, polynomial)
   → smt: 0.30 (could encode, but CAS is better for symbolic roots)

4. MCP Server → CAS Engine (sympy):
   {
     "engine": "cas",
     "operation": "solve",
     "input": { "expression": "2*x**3 - 3*x**2 - 11*x + 6", "variable": "x" }
   }

5. SymPy computes:
   factor(2*x³ - 3*x² - 11*x + 6) = (x - 3)(2x - 1)(x + 2)
   roots = [3, 1/2, -2]

6. CAS Engine → MCP Server:
   {
     "status": "success",
     "result": {
       "roots": [3, 0.5, -2],
       "factored_form": "(x - 3)(2x - 1)(x + 2)"
     },
     "proof_trace": [
       {"step": "rational_root_test", "candidates": [±1, ±2, ±3, ±6, ±1/2, ±3/2]},
       {"step": "evaluate", "x": 3, "result": 0, "found_root": true},
       {"step": "polynomial_division", "quotient": "2x² + 3x - 2"},
       {"step": "factor_quadratic", "result": "(2x - 1)(x + 2)"},
       {"step": "roots_of_factors", "roots": [0.5, -2]}
     ]
   }

7. MCP Server → Agent (tools/result):
   Full response with result + proof trace

8. Agent → User:
   "The roots are x = 3, x = 1/2, and x = -2.
    The polynomial factors as (x - 3)(2x - 1)(x + 2).
    Verified by the symbolic math engine (SymPy)."
```

### 9.2 Agent Solves Scheduling Problem (Long-Running, via MCP Tasks)

```
1. User → Agent: "Schedule 200 conference talks into 10 rooms
                   over 3 days with no speaker conflicts"

2. Agent → MCP Server (tools/call):
   { "tool": "solve", "arguments": { "problem": "...", "structured_input": { ... } } }

3. Selector: large CSP → "smt" engine (200 vars × 10 rooms × 3 days = ~6000 constraints)

4. MCP Server: problem is large → create Task
   → returns: { "task_id": "task_sched_001" }

5. Agent polls (tasks/get):
   → { "status": "running", "progress": 0.23, "detail": "1380/6000 constraints encoded" }
   → { "status": "running", "progress": 0.67, "detail": "solver searching..." }
   → { "status": "completed", "result": { "schedule": {...}, "conflicts": 0 } }

6. Agent → User: "Here's your schedule. All 200 talks assigned,
                   zero speaker conflicts. Verified by SMT solver."
```

### 9.3 Training Data Auto-Generation

```
1. Developer runs solver with logging enabled:

   logger = TrainingLogger(format="sharegpt")
   with logger:
       result = solve("Can 4 queens be placed on a 4×4 board?")

2. Logger captures the full pipeline:

   {
     "conversations": [
       {
         "from": "human",
         "value": "Can 4 queens be placed on a 4×4 board with no attacks?"
       },
       {
         "from": "gpt",
         "value": "This is a constraint satisfaction problem (4-Queens).
                   I'll use a SAT solver for a definitive answer.\n\n
                   **Encoding:** 16 boolean variables (one per cell),
                   constraints for exactly-one-per-row, per-column,
                   and no diagonal attacks.\n\n
                   **Result:** SAT (satisfiable). One valid placement:
                   Q at (0,1), (1,3), (2,0), (3,2).\n\n
                   **Verification:** Confirmed — no two queens share
                   a row, column, or diagonal."
       }
     ],
     "metadata": {
       "engine": "sat",
       "formal_input": { "clauses": [...], "num_vars": 16 },
       "formal_output": { "satisfiable": true, "model": {...} },
       "selection_reasoning": "CSP with boolean constraints → SAT solver"
     }
   }

3. Aggregated dataset published to HuggingFace for finetuning.
```

---

## 10. MCP Server Configuration

### 10.1 Default Config

```toml
# problemsolving.toml
[server]
name = "problemsolving"
version = "1.0.0"
transport = "stdio"                    # or "http"
port = 3847                            # for HTTP transport

[engines]
enabled = "all"                        # or ["sat", "smt", "cas", "astar"]
default_timeout_ms = 30000
max_concurrent_solves = 4

[backends]
sat = "varisat"                        # or "z3_sat", "cadical"
smt = "z3"
cas = "sympy"
prolog = "pyswip"                      # or "kanren"
kg = "rdflib"

[tasks]
enabled = true
max_concurrent_tasks = 8
task_timeout_ms = 300000               # 5 min max

[training]
auto_log = false                       # log all calls as training data
log_path = "./training_log.jsonl"
format = "sharegpt"

[selector]
mode = "rule_based"                    # or "embedding", "hybrid"
embedding_model = "all-MiniLM-L6-v2"  # for NL problem matching
```

### 10.2 Claude Desktop Integration

To use with Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "problemsolving": {
      "command": "problemsolving",
      "args": ["serve", "--stdio"],
      "env": {}
    }
  }
}
```

Or for remote (HTTP) mode:

```json
{
  "mcpServers": {
    "problemsolving": {
      "url": "http://localhost:3847/mcp",
      "transport": "streamable-http"
    }
  }
}
```

---

## 11. Architecture Decisions

### ADR-001: Rust Core with Language Bindings

**Decision:** Implement core algorithms in Rust, expose via PyO3 (Python) and wasm-bindgen (TypeScript).

**Reasoning:** Algorithms like SAT solving, graph search, and game tree search are compute-intensive. Rust gives C-level performance with memory safety. PyO3/maturin produce native Python wheels that pip installs seamlessly. WASM gives browser and Node.js compatibility from the same codebase.

**Trade-off:** Higher initial development cost vs. "write once, run everywhere" for performance-critical code.

**Alternative rejected:** Pure Python. Fast to write but 10-100x slower for core algorithms. Acceptable for prototyping (Phase 1) but not for a library that handles production workloads.

### ADR-002: Python-First for Ecosystem Integrations

**Decision:** Symbolic engine wrappers (Z3, SymPy, SWI-Prolog, rdflib) are native Python, not Rust.

**Reasoning:** These tools have mature Python APIs. Wrapping them through Rust FFI would add complexity with no performance benefit (the heavy computation happens in the backend engine, not the wrapper). Python is also where the LLM ecosystem lives (LangChain, LlamaIndex, HuggingFace).

**Trade-off:** TypeScript SDK cannot call SymPy/Z3 directly. It must go through the MCP server (HTTP transport) to reach Python-hosted engines. This is acceptable because these engines are inherently server-side.

### ADR-003: MCP as Primary Integration Protocol

**Decision:** MCP (Model Context Protocol) is the primary way agents interact with the toolkit, not a REST API or gRPC.

**Reasoning:** MCP is becoming the universal standard for agent-tool communication (adopted by Anthropic, OpenAI, Google, Microsoft). Building MCP-native means any MCP-compatible agent can use the toolkit with zero custom integration code. The Tools + Resources + Prompts primitives map perfectly to our use cases. The Tasks primitive handles long-running solves.

**Trade-off:** MCP is newer and still evolving. We also provide direct SDK imports for non-MCP use cases.

### ADR-004: Solver Protocol as Internal Abstraction

**Decision:** All engines communicate through a standardized JSON request/response protocol internally.

**Reasoning:** This provides a single interface between the selector, the MCP layer, and the engine backends. Adding a new engine means implementing one interface. Training data is naturally structured as protocol exchanges. The protocol can be versioned independently of any SDK.

**Trade-off:** Slight overhead for JSON serialization on hot paths. Mitigated by allowing direct Rust function calls when using the Rust SDK.

### ADR-005: Ship Python-Only First, Add Rust in Phase 2

**Decision:** Phase 1 ships as pure Python. Rust core added in Phase 2.

**Reasoning:** Adoption speed > performance in early days. A working Python package on PyPI in week 3 beats a perfect Rust core that ships in month 3. The solver protocol abstraction means swapping Python implementations for Rust is transparent to users.

**Trade-off:** Early users experience slower performance on large problems. Acceptable because most demo/evaluation problems are small.

---

## 12. Security Considerations

- **Input validation:** All solver inputs are validated against JSON Schemas before execution. Malformed inputs return clear errors, never crash the server.
- **Resource limits:** Every solver call has a configurable timeout and memory limit. No single call can exhaust system resources.
- **No code execution:** The `translate` tool returns formal representations (CNF, SMT-LIB, etc.) but never executes arbitrary code. Prolog queries run in a sandboxed SWI-Prolog instance.
- **MCP auth:** HTTP transport supports OAuth 2.1 per MCP spec. stdio transport inherits the caller's permissions.
- **Training data privacy:** The training logger only captures problem/solution pairs, never user identity or conversation context.

---

## 13. Testing Strategy

| Level | What | How |
|-------|------|-----|
| Unit | Individual algorithms produce correct results | Property-based testing (hypothesis for Python, proptest for Rust). Known solutions for benchmark problems. |
| Integration | Engine wrappers return correct protocol responses | Solver protocol round-trip tests with known-answer problems |
| MCP | Tools/resources/prompts work correctly over JSON-RPC | MCP client test harness sends requests, validates responses against spec |
| E2E | "Natural language in, correct answer out" | Benchmark suite of 100+ problems with known correct answers and expected engine selections |
| Performance | Large problems complete within time limits | Regression benchmarks: SAT (10K vars), graph search (100K nodes), SMT (1K constraints) |
| Cross-language | Python/TS/Rust SDKs produce identical results | Shared test vectors run against all SDK implementations |

---

## 14. Metrics & Observability

| Metric | Purpose |
|--------|---------|
| `solver.calls.total` | Total solve requests (by engine) |
| `solver.latency.p50/p99` | Response time distribution |
| `selector.accuracy` | Did auto-selection pick the right engine? (measured against benchmark) |
| `solver.success_rate` | Percentage of calls that return a solution |
| `solver.verification_rate` | Percentage of solutions that pass verification |
| `training.examples_generated` | Training data volume |
| `mcp.active_connections` | MCP client count (HTTP transport) |
