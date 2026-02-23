# Roadmap: Problem-Solving Algorithms — Universal AI Toolkit

## Current State

The repository contains markdown files documenting ~30 classical AI problem-solving algorithms across 11 categories, plus a complete symbolic reasoning module (8 engine types) designed to replace chain-of-thought with formal solvers. Each algorithm includes when-to-use guidance, use cases, pseudocode, and user-flow descriptions.

**What's good:** Broad coverage of both classical algorithms and symbolic reasoning engines; consistent structure; practical decision guides; clear LLM integration patterns for symbolic engines.

**What's missing:** No runnable code, no machine-readable metadata, no agent-friendly interfaces, no training data format.

---

## Target Architecture: Four Layers

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: Agent & LLM Integration                   │
│  MCP server, function-calling schemas, auto-selector│
├─────────────────────────────────────────────────────┤
│  Layer 3: Training & Evaluation Data                │
│  Instruction pairs, preference data, benchmarks     │
├─────────────────────────────────────────────────────┤
│  Layer 2: Structured Knowledge Base                 │
│  JSON/YAML metadata, decision tree, embeddings      │
├─────────────────────────────────────────────────────┤
│  Layer 1: Runnable Python SDK                       │
│  Implementations, type hints, tests, CLI            │
└─────────────────────────────────────────────────────┘
```

---

## Layer 1: Runnable Python SDK (`problemsolving`)

### Goal
A pip-installable Python package with real implementations of every algorithm.

### Proposed Structure

```
problemsolving/
├── pyproject.toml
├── README.md
├── src/
│   └── problemsolving/
│       ├── __init__.py
│       ├── search/
│       │   ├── bfs.py          # BFS with generic state interface
│       │   ├── dfs.py
│       │   ├── ucs.py
│       │   ├── greedy.py
│       │   ├── astar.py
│       │   └── __init__.py
│       ├── optimization/
│       │   ├── gradient_descent.py
│       │   ├── genetic.py
│       │   ├── simulated_annealing.py
│       │   └── __init__.py
│       ├── csp/
│       │   ├── backtracking.py
│       │   ├── local_search.py
│       │   └── __init__.py
│       ├── logic/
│       │   ├── propositional.py
│       │   ├── first_order.py
│       │   └── __init__.py
│       ├── probabilistic/
│       │   ├── bayesian.py
│       │   ├── mdp.py
│       │   └── __init__.py
│       ├── ml/
│       │   ├── supervised.py
│       │   ├── kmeans.py
│       │   ├── qlearning.py
│       │   └── __init__.py
│       ├── krr/
│       │   ├── semantic_net.py
│       │   ├── frames.py
│       │   ├── ontology.py
│       │   ├── reasoning.py    # deductive, inductive, abductive
│       │   └── __init__.py
│       ├── nlp/
│       │   ├── parsing.py
│       │   ├── semantic.py
│       │   ├── topic_model.py
│       │   ├── translation.py
│       │   ├── sentiment.py
│       │   └── __init__.py
│       ├── game_theory/
│       │   ├── minimax.py
│       │   ├── alpha_beta.py
│       │   ├── nash.py
│       │   └── __init__.py
│       ├── expert_systems/
│       │   ├── forward_chain.py
│       │   └── __init__.py
│       ├── cbr/
│       │   ├── case_based.py
│       │   └── __init__.py
│       ├── symbolic/               # Symbolic reasoning (CoT replacement)
│       │   ├── sat_solver.py       # SAT wrapper (pysat/z3 backend)
│       │   ├── smt_solver.py       # SMT wrapper (z3 backend)
│       │   ├── logic_program.py    # Prolog-style resolution (pyswip/kanren)
│       │   ├── rule_engine.py      # Forward chaining / Rete
│       │   ├── cas.py              # Computer algebra (sympy backend)
│       │   ├── kg_reasoner.py      # Knowledge graph / OWL reasoning
│       │   ├── theorem_prover.py   # Automated theorem proving
│       │   ├── hybrid.py           # Multi-engine composition
│       │   ├── translator.py       # NL → formal representation (LLM helper)
│       │   └── __init__.py
│       ├── selector.py         # Auto-selects algorithm based on problem description
│       └── cli.py              # CLI entry point
├── tests/
│   ├── test_search.py
│   ├── test_optimization.py
│   └── ...
└── examples/
    ├── maze_solver.py
    ├── scheduling.py
    └── ...
```

### Design Principles

- **Generic interfaces**: Each algorithm works with abstract `State`, `Problem`, `Graph` types so users plug in their own domains.
- **Consistent API**: Every algorithm follows `solve(problem, **config) -> Solution` pattern.
- **Observable**: Yield intermediate steps for debugging/visualization (generator-based where appropriate).
- **Dependency-light core**: Pure Python for core algorithms, optional dependencies for ML/NLP layers (numpy, etc.).
- **Type hints + docstrings**: Full typing for IDE support and documentation generation.

### Example API

```python
from problemsolving.search import astar
from problemsolving.optimization import simulated_annealing
from problemsolving import select_algorithm

# Direct use
solution = astar.solve(
    start=initial_state,
    goal_test=lambda s: s == target,
    neighbors=get_neighbors,
    heuristic=manhattan_distance,
    cost=edge_cost
)

# Auto-selection
algo = select_algorithm(
    problem_type="pathfinding",
    weighted=True,
    has_heuristic=True,
    needs_optimal=True
)
# Returns: astar with explanation of why
```

---

## Layer 2: Structured Knowledge Base

### Goal
Machine-readable metadata for every algorithm, enabling RAG retrieval, agent decision-making, and documentation generation.

### Format: JSON Schema per Algorithm

```json
{
  "id": "astar",
  "name": "A* Search",
  "category": "search",
  "subcategory": "informed_search",
  "description": "Optimal weighted pathfinding using heuristic + cost",
  "when_to_use": [
    "Weighted graphs with a good heuristic available",
    "Need optimal (lowest-cost) path, not just any path",
    "State space is finite and searchable"
  ],
  "when_not_to_use": [
    "No meaningful heuristic exists (use UCS instead)",
    "Unweighted graph (BFS is simpler and sufficient)",
    "Infinite/continuous state spaces without discretization"
  ],
  "complexity": {
    "time": "O(b^d) worst case, much better with good heuristic",
    "space": "O(b^d) — stores all generated nodes"
  },
  "parameters": [
    {"name": "start", "type": "State", "description": "Initial state"},
    {"name": "goal_test", "type": "Callable", "description": "Returns true if state is goal"},
    {"name": "neighbors", "type": "Callable", "description": "Returns adjacent states with costs"},
    {"name": "heuristic", "type": "Callable", "description": "Admissible estimate to goal"}
  ],
  "returns": {
    "type": "Solution",
    "fields": ["path", "cost", "nodes_expanded"]
  },
  "related_algorithms": ["ucs", "greedy", "bfs", "dijkstra"],
  "use_cases": [
    "Map/navigation routing",
    "Robot path planning",
    "Game AI pathfinding",
    "Planning problems with varying action costs"
  ],
  "tags": ["search", "pathfinding", "optimal", "heuristic", "graph"],
  "difficulty": "intermediate",
  "prerequisites": ["graph_basics", "priority_queues"],
  "pseudocode": "...",
  "python_module": "problemsolving.search.astar",
  "embedding_text": "A* search finds the optimal lowest-cost path in weighted graphs using a heuristic function. Best when you need guaranteed optimal solutions and have an admissible heuristic. Combines actual cost (g) with estimated remaining cost (h) to prioritize promising paths."
}
```

### Decision Tree (JSON)

A machine-readable decision tree that agents can traverse to pick the right algorithm:

```json
{
  "question": "What type of problem are you solving?",
  "branches": {
    "pathfinding": {
      "question": "Are edges weighted?",
      "branches": {
        "no": {"answer": "bfs"},
        "yes": {
          "question": "Do you have a heuristic?",
          "branches": {
            "yes": {
              "question": "Need optimal solution?",
              "branches": {
                "yes": {"answer": "astar"},
                "no": {"answer": "greedy"}
              }
            },
            "no": {"answer": "ucs"}
          }
        }
      }
    },
    "constraint_satisfaction": {
      "question": "How large is the problem?",
      "branches": {
        "small_exact": {"answer": "backtracking"},
        "large_approximate": {"answer": "local_search_csp"}
      }
    }
  }
}
```

### Files

```
knowledge/
├── algorithms/           # One JSON per algorithm
│   ├── bfs.json
│   ├── astar.json
│   └── ...
├── decision_tree.json    # The algorithm selector logic
├── categories.json       # Taxonomy of algorithm categories
├── schema.json           # JSON Schema for validation
└── embeddings/           # Pre-computed embeddings for RAG
    ├── config.yaml       # Embedding model + params
    └── vectors.parquet   # Algorithm descriptions as vectors
```

---

## Layer 3: Training & Evaluation Data

### Goal
Datasets for finetuning LLMs to better select, explain, and apply these algorithms.

### Dataset Types

#### 3a. Instruction-Tuning Pairs

Problem descriptions paired with correct algorithm selection and implementation guidance.

```jsonl
{
  "instruction": "I have a maze represented as a grid. Each cell is either open or blocked. I need to find the shortest path from top-left to bottom-right.",
  "response": "This is an unweighted shortest-path problem on a grid graph. Use **BFS (Breadth-First Search)** because:\n1. The grid has uniform step costs (each move = 1 step)\n2. BFS guarantees the shortest path in unweighted graphs\n3. It's simpler and faster than A* when all edges have equal weight\n\nHere's how to apply it:\n- State: (row, col) position\n- Neighbors: 4-directional moves to open cells\n- Goal test: reached bottom-right corner\n\n```python\nfrom problemsolving.search import bfs\nsolution = bfs.solve(start=(0,0), goal_test=lambda s: s==(rows-1,cols-1), neighbors=grid_neighbors)\n```",
  "category": "algorithm_selection",
  "correct_algorithm": "bfs",
  "difficulty": "easy"
}
```

#### 3b. Preference / Ranking Data (for RLHF / DPO)

Pairs where one algorithm choice is better than another for a given problem.

```jsonl
{
  "prompt": "I need to find the cheapest flight route from NYC to Tokyo with multiple layovers.",
  "chosen": "Use A* Search or UCS. This is a weighted shortest-path problem where edge weights are flight costs. A* is preferred if you have a heuristic (e.g., great-circle distance), otherwise UCS gives optimal cost.",
  "rejected": "Use DFS to explore all possible routes and pick the cheapest. DFS will find a route quickly.",
  "explanation": "DFS doesn't guarantee optimal cost and may explore unnecessarily deep paths. UCS/A* are designed for weighted optimal pathfinding."
}
```

#### 3c. Chain-of-Thought Reasoning Traces

Step-by-step reasoning for algorithm selection.

```jsonl
{
  "problem": "A hospital needs to schedule 50 nurses across 3 shifts for 30 days, satisfying constraints: no nurse works >5 consecutive days, each shift has minimum staffing, nurses have availability preferences.",
  "reasoning": [
    "Step 1: Identify problem type — this is assigning values (shifts) to variables (nurse-day slots) with constraints. This is a Constraint Satisfaction Problem (CSP).",
    "Step 2: Assess scale — 50 nurses × 30 days = 1500 variables. This is a large CSP.",
    "Step 3: Exact vs approximate — with 1500 variables, exact backtracking will be too slow. Need approximate/local search methods.",
    "Step 4: Select algorithm — Local Search CSP (start with random feasible-ish assignment, iteratively reduce violations). Could also use Simulated Annealing to escape local minima.",
    "Step 5: Implementation plan — define variables, domains, hard constraints (min staffing), soft constraints (preferences). Use SA-based local search with violation count as cost."
  ],
  "selected_algorithm": "local_search_csp",
  "alternatives_considered": ["backtracking", "genetic_algorithm"]
}
```

#### 3d. Evaluation Benchmarks

Test problems with known correct answers for measuring model performance.

```jsonl
{
  "id": "eval_001",
  "problem": "Find shortest path in unweighted graph with 1000 nodes.",
  "correct_algorithms": ["bfs"],
  "acceptable_algorithms": ["bfs", "astar"],
  "wrong_algorithms": ["dfs", "greedy"],
  "rubric": {
    "algorithm_selection": 3,
    "justification_quality": 3,
    "implementation_correctness": 4
  }
}
```

### Files

```
training_data/
├── instruction_tuning/
│   ├── algorithm_selection.jsonl    # 200+ problem→algorithm pairs
│   ├── implementation_guidance.jsonl # How to apply each algorithm
│   └── explanation.jsonl            # Explain algorithm behavior
├── preference/
│   ├── algorithm_ranking.jsonl      # Better vs worse choices
│   └── explanation_quality.jsonl    # Good vs bad explanations
├── chain_of_thought/
│   ├── selection_reasoning.jsonl    # Step-by-step reasoning
│   └── debugging_traces.jsonl       # "Algorithm X failed because..."
├── evaluation/
│   ├── benchmarks.jsonl             # Test problems + rubrics
│   ├── edge_cases.jsonl             # Tricky problems
│   └── scoring.py                   # Auto-evaluation script
└── README.md                        # Dataset card (format, license, usage)
```

---

## Layer 4: Agent & LLM Integration

### Goal
Plug-and-play integration with agent frameworks and LLM tool-calling systems.

### 4a. MCP Server (Model Context Protocol)

An MCP server that exposes algorithms as tools any Claude/MCP-compatible agent can call.

```python
# MCP tool definitions
tools = [
    {
        "name": "select_algorithm",
        "description": "Given a problem description, recommends the best algorithm",
        "parameters": {
            "problem_description": "string",
            "constraints": "object (optional)"
        }
    },
    {
        "name": "run_algorithm",
        "description": "Execute a specific algorithm on a given problem",
        "parameters": {
            "algorithm": "string (e.g., 'astar', 'bfs')",
            "problem_data": "object"
        }
    },
    {
        "name": "explain_algorithm",
        "description": "Get detailed explanation of an algorithm",
        "parameters": {
            "algorithm": "string",
            "detail_level": "string (brief|detailed|with_code)"
        }
    }
]
```

### 4b. OpenAI / Anthropic Function-Calling Schemas

```json
{
  "type": "function",
  "function": {
    "name": "solve_with_algorithm",
    "description": "Solve a problem using the appropriate classical AI algorithm. Use this when the user has a search, optimization, constraint satisfaction, or planning problem.",
    "parameters": {
      "type": "object",
      "properties": {
        "problem_type": {
          "type": "string",
          "enum": ["pathfinding", "optimization", "csp", "logical_inference", "probabilistic", "adversarial", "classification", "clustering"]
        },
        "problem_data": {
          "type": "object",
          "description": "Problem-specific data (graph, constraints, etc.)"
        },
        "preferences": {
          "type": "object",
          "properties": {
            "needs_optimal": {"type": "boolean"},
            "time_budget": {"type": "string"},
            "explainability": {"type": "boolean"}
          }
        }
      },
      "required": ["problem_type", "problem_data"]
    }
  }
}
```

### 4c. LangChain / LlamaIndex Wrappers

```python
from problemsolving.integrations.langchain import ProblemSolvingToolkit

# Creates a set of LangChain tools from the algorithm library
toolkit = ProblemSolvingToolkit()
tools = toolkit.get_tools()
# Returns: [SelectAlgorithmTool, RunAlgorithmTool, ExplainAlgorithmTool, ...]
```

### Files

```
integrations/
├── mcp/
│   ├── server.py           # MCP server implementation
│   └── tool_definitions.json
├── openai/
│   └── function_schemas.json
├── langchain/
│   ├── tools.py
│   └── agent.py            # Example agent using the toolkit
├── llamaindex/
│   └── tools.py
└── examples/
    ├── claude_agent.py      # Claude using MCP tools
    ├── openai_agent.py      # GPT using function calling
    └── autonomous_solver.py # Agent that auto-selects and runs algorithms
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
- Set up Python package structure with pyproject.toml
- Implement core search algorithms (BFS, DFS, UCS, A*, Greedy) with tests
- Implement optimization algorithms (gradient descent, GA, SA) with tests
- Create the JSON schema for algorithm metadata
- Write metadata for all search and optimization algorithms

### Phase 2: Symbolic Reasoning Engines (Weeks 4-6)
- Implement SAT solver wrapper (pysat + z3 backends) with NL→CNF translator
- Implement SMT solver wrapper (z3) with NL→SMT-LIB translator
- Implement Prolog-style logic programming (pyswip/kanren)
- Implement rule engine (forward chaining with Rete)
- Implement CAS wrapper (sympy) with NL→symbolic translator
- Implement KG reasoner (rdflib + owlready2)
- Build verification loop pattern (LLM proposes → engine verifies)
- Build hybrid multi-engine composition

### Phase 3: Remaining Classical Algorithms (Weeks 7-8)
- Implement CSP, logic, probabilistic, game theory modules
- Implement KRR, expert systems, CBR modules
- NLP module (lightweight — interfaces + integration hooks for real NLP libraries)
- Complete JSON metadata for all algorithms + symbolic engines
- Build the decision tree selector (now includes symbolic engine routing)

### Phase 4: Knowledge Base & Training Data (Weeks 9-11)
- Generate instruction-tuning dataset (200+ examples, including symbolic reasoning tasks)
- Create preference/ranking pairs: CoT vs symbolic reasoning comparisons
- Write symbolic reasoning traces (formal solver steps instead of natural language CoT)
- Build evaluation benchmarks: correctness on SAT, math, logic, KG tasks
- Create pre-computed embeddings for RAG retrieval

### Phase 5: Agent Integration (Weeks 12-14)
- Build MCP server (expose both classical algorithms and symbolic engines as tools)
- Create OpenAI function-calling schemas (including solve → verify loop)
- Build LangChain and LlamaIndex tool wrappers
- Write example agents demonstrating: NL → symbolic engine → NL pipeline
- CLI tool for interactive algorithm + engine selection

### Phase 6: Polish & Launch (Weeks 15-16)
- Documentation site (MkDocs or similar)
- PyPI publishing
- Example notebooks (Jupyter)
- Dataset cards for HuggingFace
- GitHub Actions CI/CD

---

## Quick Start: What to Build First

If you want to start coding today, here's the highest-impact first step:

1. **Create `pyproject.toml`** and package skeleton
2. **Implement `astar.py`** as the reference algorithm (most versatile, good test case for the generic interface pattern)
3. **Write `astar.json`** metadata (establishes the schema all others follow)
4. **Write one instruction-tuning example** for A* (establishes the training data format)

This gives you one complete vertical slice through all four layers, which you can then replicate across the other 29 algorithms.

---

## Key Design Decisions to Make

1. **Pure Python vs. NumPy core?** Pure Python is more portable but slower. Recommendation: Pure Python core with optional NumPy acceleration for optimization/ML algorithms.

2. **NLP module scope?** Full NLP implementations are massive projects. Recommendation: Provide interfaces and hooks that delegate to spaCy, HuggingFace, etc., rather than reimplementing tokenizers and transformers.

3. **Training data generation?** Manual curation is high quality but slow. Recommendation: Seed with 50 hand-written examples, then use an LLM to generate 500+ synthetic examples, then human-review a subset.

4. **Embedding model for RAG?** Recommendation: Support multiple (sentence-transformers as default, OpenAI embeddings as option). Store the embedding config alongside vectors.

5. **License?** MIT or Apache 2.0 for maximum adoption. Training data may need a separate data-specific license (like CC-BY-4.0).
