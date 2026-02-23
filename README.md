# Problem Solving Strategies in AI

A universal toolkit of classical AI algorithms and symbolic reasoning engines for agents, LLM applications, finetuning, and optimization.

**Files:**
- `algorithms.md` — Classical AI algorithms with pseudocode and user flows
- `symbolic_reasoning.md` — Symbolic engines as Chain-of-Thought replacement (SAT, SMT, Prolog, CAS, KG reasoning, theorem provers)
- `ARCHITECTURE.md` — System architecture: Rust core, Python/TS SDKs, MCP server, solver protocol
- `ROADMAP.md` — Plan to turn this into a runnable SDK, knowledge base, training data, and agent integration layer
- `IMPLEMENTATION_PLAN.md` — Phase-by-phase TDD implementation plan (15 phases, 300+ tests)

---

## 1. Search Algorithms

### Uninformed Search
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform Cost Search (UCS)

### Informed Search
- A* Search
- Greedy Search

---

## 2. Optimization
- Gradient Descent
- Genetic Algorithms
- Simulated Annealing

---

## 3. Constraint Satisfaction Problems (CSPs)
- Backtracking
- Local Search
  - Hill Climbing
  - Simulated Annealing
  - Genetic Algorithms

---

## 4. Logical Reasoning
- Propositional Logic
- First-Order Logic

---

## 5. Probabilistic Methods
- Bayesian Networks
- Markov Decision Processes (MDPs)

---

## 6. Machine Learning
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

---

## 7. Knowledge Representation and Reasoning (KRR)

### Knowledge Representation
- Semantic Networks
- Frames
- Ontology

### Reasoning
- Deductive Reasoning
- Inductive Reasoning
- Abductive Reasoning

---

## 8. Natural Language Processing (NLP)
- Parsing
- Semantic Analysis
- Topic Modeling
- Machine Translation
- Sentiment Analysis

---

## 9. Game Theory
- Nash Equilibrium
- Dominant Strategy
- Minimax
- Alpha-Beta Pruning
- Mixed Strategies

---

## 10. Expert Systems
- Rule-Based Reasoning

---

## 11. Case-Based Reasoning (CBR)
- Retrieval
- Reuse
- Revise
- Retain
- Information Retrieval

---

## 12. Symbolic Reasoning (Chain-of-Thought Replacement)

Formal engines that replace LLM free-form reasoning with provably correct computation. See `symbolic_reasoning.md` for full details.

### Solver Engines
- SAT Solvers (DPLL, CDCL) — boolean satisfiability
- SMT Solvers (Z3, CVC5) — satisfiability modulo theories (integers, reals, arrays)
- Logic Programming (Prolog, miniKanren) — relational/rule-based reasoning
- Rule Engines (Rete, forward chaining) — production systems, compliance

### Symbolic Math
- Computer Algebra Systems (SymPy, SageMath) — exact algebraic computation

### Knowledge Reasoning
- Knowledge Graph Reasoners (OWL, RDFS) — ontology inference

### Formal Verification
- Theorem Provers (Lean, Coq, Vampire) — machine-checked proofs

### Agent Integration Patterns
- Tool-based: agent calls symbolic engine as a tool
- Verification loop: LLM proposes, engine verifies
- Neurosymbolic: symbolic engine drives, LLM provides heuristics

