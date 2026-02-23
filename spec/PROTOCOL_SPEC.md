# ProblemSolving Protocol Specification

Version 0.1.0 — RFC-Style

## Abstract

This document specifies the Solver Protocol used by the ProblemSolving toolkit. It defines the request/response envelope format, engine registration contract, MCP integration layer, and error handling conventions. The protocol enables any consumer (direct Python import, MCP client, REST API, or agent framework) to interact with solver engines through a uniform interface.

---

## 1. Terminology

- **Engine:** A solver implementation that handles a specific class of problems (e.g., BFS, DPLL, CAS).
- **Registry:** A collection of named engines available for dispatch.
- **Consumer:** Any code that sends requests to the protocol (MCP client, Python caller, agent).
- **Solver Protocol:** The request/response envelope format defined in this document.
- **Protocol Adapter:** A function that converts dict input/output to/from an engine's native API.

**Key words:** "MUST", "MUST NOT", "SHOULD", "MAY" follow RFC 2119 conventions.

---

## 2. Request Envelope

### 2.1 Format

A Solver Request is a structured message with the following fields:

```
SolverRequest {
    id:         string        // Unique request identifier
    engine:     string        // Target engine name
    operation:  string        // Operation name (default: "solve")
    input:      object        // Engine-specific input data
    config:     object        // Optional configuration overrides
}
```

### 2.2 Field Requirements

| Field | Required | Default | Constraints |
|-------|----------|---------|-------------|
| `id` | No | Auto-generated `req_<12-hex-chars>` | MUST be unique per request |
| `engine` | Yes | — | MUST match a registered engine name or `"auto"` |
| `operation` | No | `"solve"` | Engine-defined; MUST be a non-empty string |
| `input` | Yes | — | MUST be a JSON-serializable dict |
| `config` | No | `{}` | MUST be a JSON-serializable dict |

### 2.3 ID Generation

When `id` is not provided, the protocol layer MUST generate one using the format:

```
"req_" + hex(uuid4())[:12]
```

This produces IDs like `req_a1b2c3d4e5f6`.

### 2.4 Serialization

`SolverRequest.to_dict()` MUST produce:

```json
{
    "id": "req_a1b2c3d4e5f6",
    "engine": "bfs",
    "operation": "solve",
    "input": { ... },
    "config": { }
}
```

The key `input_data` in the Python dataclass maps to `input` in the serialized form.

---

## 3. Response Envelope

### 3.1 Format

```
SolverResponse {
    id:           string          // Matches request ID
    status:       string          // "success" | "error"
    engine:       string          // Engine that handled the request
    result:       object | null   // Engine output (success only)
    error:        object | null   // Error details (error only)
    proof_trace:  array[object]   // Optional reasoning steps
    metadata:     object          // Optional timing/stats
}
```

### 3.2 Success Response

When an engine completes successfully:

- `status` MUST be `"success"`.
- `result` MUST contain the engine's output dict.
- `error` MUST be `null`.

```json
{
    "id": "req_a1b2c3d4e5f6",
    "status": "success",
    "engine": "bfs",
    "result": {
        "path": ["A", "B", "D"],
        "nodes_explored": 4
    }
}
```

### 3.3 Error Response

When an engine fails or is not found:

- `status` MUST be `"error"`.
- `error` MUST contain at least `code` and `message`.
- `result` MUST be `null`.

```json
{
    "id": "req_a1b2c3d4e5f6",
    "status": "error",
    "engine": "unknown_engine",
    "error": {
        "code": "UNKNOWN_ENGINE",
        "message": "Engine 'unknown_engine' not found.",
        "suggestion": "Use list_engines() to see available engines."
    }
}
```

### 3.4 Error Codes

| Code | When |
|------|------|
| `UNKNOWN_ENGINE` | Requested engine not in registry |
| `SOLVER_ERROR` | Engine raised an exception during solve |
| `INVALID_INPUT` | Input data does not match engine schema |
| `TIMEOUT` | Engine exceeded time limit (future) |

### 3.5 Proof Trace

The `proof_trace` field MAY contain an ordered list of reasoning steps:

```json
{
    "proof_trace": [
        {"step": 1, "action": "expand", "node": "A", "frontier": ["B", "C"]},
        {"step": 2, "action": "expand", "node": "B", "frontier": ["C", "D"]}
    ]
}
```

Proof traces are engine-defined and SHOULD be human-readable.

### 3.6 Metadata

The `metadata` field MAY contain:

```json
{
    "metadata": {
        "solve_time_ms": 12.5,
        "engine_version": "0.1.0",
        "nodes_explored": 42
    }
}
```

### 3.7 Serialization

`SolverResponse.to_dict()` MUST omit `null` fields and empty collections:

- If `result` is `None`, the `result` key is omitted.
- If `error` is `None`, the `error` key is omitted.
- If `proof_trace` is empty, the key is omitted.
- If `metadata` is empty, the key is omitted.

---

## 4. Engine Registration Contract

### 4.1 Registration

An engine is registered by providing:

```python
registry.register(
    name: str,                              # Unique engine name
    solve_fn: Callable[[dict], dict],       # Protocol adapter function
    tags: list[str] | None = None,          # Categorical tags
)
```

### 4.2 Engine Name Convention

Engine names MUST be:

- Lowercase alphanumeric with underscores (regex: `[a-z][a-z0-9_]*`)
- Unique within a registry
- Descriptive of the algorithm (e.g., `bfs`, `dpll_sat`, `gradient_descent`)

### 4.3 Protocol Adapter Contract

Every engine MUST provide a protocol adapter function with signature:

```python
def engine_solve_from_dict(input_data: dict[str, Any]) -> dict[str, Any]
```

The adapter:

1. MUST accept a single `dict` argument.
2. MUST return a `dict` with the engine's result.
3. MAY raise exceptions (the registry wrapper catches them and creates error responses).
4. MUST NOT modify the input dict.
5. SHOULD validate required input keys.

### 4.4 Tag Convention

Tags SHOULD use one of the standard categories:

| Tag | Meaning |
|-----|---------|
| `search` | Graph/tree search algorithms |
| `pathfinding` | Finds paths between nodes |
| `optimal` | Guarantees optimal solution |
| `heuristic` | Uses heuristic guidance |
| `optimization` | Numeric optimization |
| `gradient` | Gradient-based methods |
| `evolutionary` | Population-based methods |
| `metaheuristic` | General-purpose optimization |
| `sat` | Boolean satisfiability |
| `csp` | Constraint satisfaction |
| `constraint` | Constraint-based solving |
| `symbolic` | Symbolic computation |
| `algebra` | Algebraic operations |
| `calculus` | Differentiation/integration |
| `smt` | Satisfiability Modulo Theories |
| `arithmetic` | Integer/real arithmetic |
| `logic` | Logic programming |

### 4.5 Engine Handle Flow

When `RegisteredEngine.handle(request)` is called:

```
1. Extract input_data from request
2. Call solve_fn(input_data)
   ├── Success → SolverResponse.success(...)
   └── Exception → SolverResponse.make_error(code="SOLVER_ERROR", message=str(e))
3. Return SolverResponse
```

The handle method MUST never raise an exception to the caller.

---

## 5. MCP Integration

### 5.1 Protocol Compliance

The MCP server follows the Model Context Protocol specification. It exposes tools, resources, and prompts through a JSON-RPC 2.0 transport.

### 5.2 Server Manifest

The server manifest MUST include:

```json
{
    "name": "problemsolving",
    "version": "0.1.0",
    "description": "Universal AI reasoning toolkit",
    "tools": [ ... ]
}
```

### 5.3 Tool Definitions

Each tool MUST include:

- `name`: Tool identifier (string)
- `description`: Human-readable description (string)
- `inputSchema`: JSON Schema for input parameters (object)

### 5.4 Tool Dispatch

The `dispatch_tool(tool_name, params)` function:

1. MUST look up the handler by `tool_name`.
2. MUST return `{"error": "Unknown tool: <name>"}` for unregistered tools.
3. MUST pass `params` dict to the handler unchanged.
4. MUST return the handler's result dict.

### 5.5 Tool-to-Engine Mapping

```
MCP Tool "solve"          → handle_solve()          → problemsolving.solve()
MCP Tool "select_algorithm" → handle_select_algorithm() → selector.select_algorithm()
MCP Tool "list_engines"    → handle_list_engines()   → problemsolving.list_engines()
MCP Tool "verify"          → handle_verify()         → solve() + comparison
MCP Tool "explain_algorithm" → handle_explain_algorithm() → knowledge JSON lookup
```

### 5.6 Verify Tool Semantics

The `verify` tool:

1. Runs the specified engine with the given input.
2. For each key in `expected`, compares against the actual result.
3. Returns `verified: true` only if ALL expected keys match exactly.
4. Returns `mismatches` array listing each difference.

---

## 6. Knowledge Base Contract

### 6.1 Location

Algorithm metadata files MUST be stored at:

```
knowledge/algorithms/<algorithm_id>.json
```

### 6.2 Required Fields

Each metadata file MUST conform to `knowledge/schema.json` and include:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier matching filename |
| `name` | string | Human-readable name |
| `category` | string | Primary category |
| `description` | string | One-sentence description |
| `when_to_use` | array[string] | Scenarios where this is appropriate |
| `complexity` | object | `time` and `space` complexity strings |
| `parameters` | array[object] | Input parameters with types |
| `tags` | array[string] | Searchable tags |

### 6.3 Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `subcategory` | string | Finer categorization |
| `when_not_to_use` | array[string] | Anti-patterns |
| `related_algorithms` | array[string] | Related algorithm IDs |
| `use_cases` | array[string] | Concrete use cases |
| `difficulty` | string | `beginner`, `intermediate`, `advanced` |
| `python_module` | string | Python import path |
| `embedding_text` | string | Text optimized for semantic search |

---

## 7. Data Flow

### 7.1 Direct Python Call

```
User Code
  → solve(engine="bfs", input_data={...})
    → get_default_registry()
    → registry.get("bfs")
    → SolverRequest(engine="bfs", input_data={...})
    → RegisteredEngine.handle(request)
      → bfs_solve_from_dict(input_data)
      → SolverResponse.success(...)
    → return SolverResponse
```

### 7.2 MCP Call

```
MCP Client (Agent)
  → JSON-RPC: {"method": "tools/call", "params": {"name": "solve", ...}}
    → dispatch_tool("solve", params)
      → handle_solve(params)
        → problemsolving.solve(engine=params["engine"], ...)
          → [same as 7.1]
        → response.to_dict()
      → return dict
    → JSON-RPC response
```

### 7.3 Auto-Selection Flow

```
solve(engine="auto", problem_type="pathfinding", features={"weighted": True, "has_heuristic": True})
  → select_algorithm("pathfinding", {"weighted": True, "has_heuristic": True})
    → match rules by specificity
    → return {"algorithm": "astar", "reasoning": "..."}
  → solve(engine="astar", input_data={...})
    → [normal engine dispatch]
```

---

## 8. Versioning

### 8.1 Protocol Version

The protocol version follows semantic versioning: `MAJOR.MINOR.PATCH`.

- **MAJOR:** Breaking changes to request/response envelope format.
- **MINOR:** New optional fields or new engines.
- **PATCH:** Bug fixes, no schema changes.

### 8.2 Compatibility

- Consumers MUST ignore unknown fields in responses.
- Engines MUST NOT require fields not in the current spec.
- New optional fields MUST have sensible defaults.
