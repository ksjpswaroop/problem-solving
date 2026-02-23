# Classical AI Algorithms for Problem Solving

# 1) Search Algorithms

## 1.1 BFS (Breadth-First Search)

### When to use
	•	Unweighted graphs; want shortest path in #edges
	•	Good for “levels” / minimum hops

### Use cases
	•	Shortest hops in social graphs
	•	Maze solving on grid with uniform step cost
	•	Finding nearest available resource/node

### Pseudocode

BFS(start, goal):
  Q ← queue()
  Q.push(start)
  visited ← {start}
  parent ← map()          # to reconstruct path

  while Q not empty:
    u ← Q.pop_front()
    if u == goal:
      return reconstruct_path(parent, start, goal)

    for v in neighbors(u):
      if v not in visited:
        visited.add(v)
        parent[v] = u
        Q.push(v)

  return failure

### User flow (product/agent)
	1.	User asks: “Find shortest route (fewest steps).”
	2.	System builds graph/state space.
	3.	Run BFS until goal found.
	4.	Return path + steps.

 

## 1.2 DFS (Depth-First Search)

### When to use
	•	Need to explore deeply, memory-limited
	•	Want any solution (not necessarily shortest)
	•	Useful for backtracking-style exploration

### Use cases
	•	Exploring all configurations
	•	Cycle detection / topological ordering (variants)
	•	Puzzle solving (with pruning)

### Pseudocode

DFS(u, goal):
  if u == goal: return success
  mark u visited
  for v in neighbors(u):
    if v not visited:
      parent[v] = u
      if DFS(v, goal) == success:
        return success
  return failure

### User flow
	1.	User: “Find a feasible solution quickly.”
	2.	System: start DFS; optionally limit depth/time.
	3.	Return found solution or “none found”.

 

1.3 UCS (Uniform Cost Search)

### When to use
	•	Weighted edges with nonnegative costs
	•	Need lowest-cost path, not just fewest steps

### Use cases
	•	Cheapest travel path (time, cost, energy)
	•	Planning with varying action costs

### Pseudocode

UCS(start, goal):
  PQ ← priority_queue()            # min by g-cost
  PQ.push(start, cost=0)
  parent ← map()
  g ← map(default=∞)
  g[start] = 0

  while PQ not empty:
    u ← PQ.pop_min()
    if u == goal:
      return reconstruct_path(parent, start, goal)

    for (v, w) in neighbors_with_cost(u):
      if g[u] + w < g[v]:
        g[v] = g[u] + w
        parent[v] = u
        PQ.push(v, cost=g[v])

  return failure

### User flow
	1.	User defines objective: minimize cost.
	2.	System defines edge weights.
	3.	Run UCS and return least-cost plan.

 

1.4 Greedy Best-First Search

### When to use
	•	Have heuristic estimate to goal
	•	Want fast solutions; optimality not guaranteed

### Use cases
	•	Real-time routing when “good enough” is ok
	•	Fast approximate planning in large spaces

### Pseudocode

GreedyBestFirst(start, goal, h):
  PQ ← priority_queue()         # min by h(node)
  PQ.push(start, priority=h(start))
  visited ← set()
  parent ← map()

  while PQ not empty:
    u ← PQ.pop_min()
    if u == goal: return reconstruct_path(parent, start, goal)
    if u in visited: continue
    visited.add(u)

    for v in neighbors(u):
      if v not in visited:
        parent[v] = u
        PQ.push(v, priority=h(v))

  return failure

### User flow
	1.	User: “Make it fast.”
	2.	System chooses heuristic h.
	3.	Greedy search returns quick candidate path.

 

1.5 A* Search

### When to use
	•	Weighted pathfinding with a good heuristic
	•	Want optimal solution if heuristic is admissible/consistent

### Use cases
	•	Maps, robotics, game pathfinding
	•	Planning problems with costs

### Pseudocode

AStar(start, goal, h):
  open ← priority_queue()                 # min by f=g+h
  open.push(start, priority=h(start))
  parent ← map()
  g ← map(default=∞)
  g[start]=0

  while open not empty:
    u ← open.pop_min()
    if u == goal:
      return reconstruct_path(parent, start, goal)

    for (v, w) in neighbors_with_cost(u):
      tentative = g[u] + w
      if tentative < g[v]:
        g[v] = tentative
        parent[v] = u
        f = g[v] + h(v)
        open.push(v, priority=f)

  return failure

### User flow
	1.	User: “Find best route.”
	2.	System sets cost function + heuristic.
	3.	A* returns optimal route + proof (cost).

 

2) Optimization

2.1 Gradient Descent

### When to use
	•	Continuous optimization; differentiable objective
	•	Training neural nets, regression, etc.

### Use cases
	•	Model training
	•	Minimizing loss functions
	•	Control parameter tuning

### Pseudocode

GradientDescent(θ, loss, lr, steps):
  for t in 1..steps:
    g = ∇θ loss(θ)
    θ = θ - lr * g
  return θ

How to use
	•	Define loss(θ)
	•	Compute gradient (autodiff)
	•	Choose lr + schedule

### User flow
	1.	User selects objective (min error).
	2.	System runs training loop.
	3.	Returns best parameters + metrics.

 

2.2 Genetic Algorithms (GA)

### When to use
	•	Discrete/black-box objective
	•	Hard search space; no gradients

### Use cases
	•	Hyperparameter search
	•	Scheduling, routing, combinatorial optimization
	•	Neural architecture search (simple versions)

### Pseudocode

GeneticAlgorithm():
  P = initialize_population()
  for gen in 1..G:
    fitness = evaluate(P)
    parents = select(P, fitness)
    children = crossover(parents)
    children = mutate(children)
    P = survive(P, children, fitness)
  return best_individual(P)

### User flow
	1.	User defines fitness function.
	2.	System generates candidates.
	3.	Iteratively evolves best solution.

 

2.3 Simulated Annealing (SA)

### When to use
	•	Discrete optimization; avoid local minima
	•	Accepts worse moves early, becomes greedy later

### Use cases
	•	TSP approximations
	•	Layout, assignment, scheduling
	•	Any “score-based” optimization

### Pseudocode

SimulatedAnnealing(s0):
  s = s0
  T = T0
  while T > Tmin:
    s_new = random_neighbor(s)
    Δ = cost(s_new) - cost(s)
    if Δ <= 0: s = s_new
    else if rand() < exp(-Δ / T): s = s_new
    T = cool(T)          # e.g., T = αT
  return s

### User flow
	1.	User defines cost + neighbor function.
	2.	System runs SA, returns best found.

 

3) CSP (Constraint Satisfaction Problems)

3.1 Backtracking (CSP Solver)

### When to use
	•	Variables + domains + constraints
	•	Need exact solution (or all solutions)

### Use cases
	•	Sudoku, scheduling, timetabling
	•	Configuration (product bundles that must match rules)

### Pseudocode

Backtrack(assignment):
  if assignment complete: return assignment

  X = select_unassigned_variable()
  for value in order_domain_values(X):
    if consistent(X=value, assignment):
      assignment[X] = value
      result = Backtrack(assignment)
      if result != failure: return result
      remove assignment[X]
  return failure

### User flow
	1.	User enters constraints (rules).
	2.	System searches assignments.
	3.	Returns valid assignment or “none”.

 

3.2 Local Search for CSP (Hill Climbing / SA / GA)

### When to use
	•	Very large CSP; exact backtracking too slow
	•	Want “good enough” feasible assignment

### Use cases
	•	Large schedules (shifts, classes)
	•	Resource allocation under many constraints

### Pseudocode (generic)

LocalSearchCSP(s0):
  s = s0                       # a complete assignment (maybe violating constraints)
  while not stop:
    if violations(s) == 0: return s
    s = pick_neighbor_with_fewer_violations(s)
  return best_seen

### User flow
	1.	User: “Find a feasible schedule fast.”
	2.	System starts from random complete assignment.
	3.	Minimizes violations until feasible.

 

4) Logical Reasoning

4.1 Propositional Logic (Inference via Resolution)

### When to use
	•	Facts are boolean propositions
	•	Need crisp true/false inference

### Use cases
	•	Rule checks, access policy, configuration validity
	•	Simple theorem proving

### Pseudocode (resolution sketch)

Entails(KB, query):
  clauses = CNF(KB ∧ ¬query)
  new = {}
  loop:
    for each pair (ci, cj) in clauses:
      resolvents = Resolve(ci, cj)
      if {} in resolvents: return TRUE
      new = new ∪ resolvents
    if new ⊆ clauses: return FALSE
    clauses = clauses ∪ new

### User flow
	1.	User provides rules + facts.
	2.	System checks if query is entailed.
	3.	Returns yes/no + explanation trace.

 

4.2 First-Order Logic (Unification + Forward/Backward Chaining)

### When to use
	•	Need variables, relations, quantifiers
	•	Rich symbolic reasoning, knowledge bases

### Use cases
	•	Knowledge graphs with rules
	•	Expert systems, compliance reasoning

### Pseudocode (backward chaining sketch)

Prove(goal, KB):
  if goal matches a fact in KB: return success
  for rule in KB where rule.head unifies with goal:
    θ = unify(rule.head, goal)
    if all Prove(subgoalθ, KB) for subgoal in rule.body:
      return success
  return failure

### User flow
	1.	User asks query (“Is X allowed?”).
	2.	System finds rules, unifies, proves subgoals.
	3.	Returns proof tree or failure.

 

5) Probabilistic Methods

5.1 Bayesian Networks (Inference by Variable Elimination)

### When to use
	•	Uncertainty with causal structure
	•	Need posterior probabilities given evidence

### Use cases
	•	Diagnostics (fault detection)
	•	Risk scoring, medical triage
	•	Sensor fusion

### Pseudocode (high-level)

BayesInfer(BN, QueryVar, Evidence):
  factors = CPTs conditioned on Evidence
  for hidden var Z not in QueryVar/Evidence:
    factors = sum_out(Z, factors)
  result = multiply_all(factors)
  return normalize(result over QueryVar)

### User flow
	1.	User provides evidence (symptoms/signals).
	2.	System computes posterior P(Query | evidence).
	3.	Returns probability + top explanations.

 

5.2 Markov Decision Processes (MDPs) — Value Iteration

### When to use
	•	Sequential decisions under uncertainty
	•	Want optimal policy maximizing long-term reward

### Use cases
	•	Robotics, inventory control
	•	Resource management, routing with uncertainty
	•	Agent planning under stochastic outcomes

### Pseudocode

ValueIteration(S, A, P, R, γ):
  V(s)=0 for all s
  repeat until convergence:
    for s in S:
      V_new(s) = max_a Σ_{s'} P(s'|s,a) * ( R(s,a,s') + γ * V(s') )
    V = V_new
  π(s) = argmax_a Σ_{s'} P(s'|s,a) * ( R + γV(s') )
  return π, V

### User flow
	1.	User defines reward (what matters).
	2.	System computes optimal policy.
	3.	Returns policy + expected value.

 

6) Machine Learning

6.1 Supervised Learning (Generic Training Loop)

### When to use
	•	Labeled data (x, y)

### Use cases
	•	Classification, regression, ranking

### Pseudocode

TrainSupervised(model, data, loss, optimizer):
  initialize θ
  for epoch in 1..E:
    for batch (x, y) in data:
      y_hat = model(x; θ)
      L = loss(y_hat, y)
      θ = optimizer.step(θ, ∇θ L)
  return θ

### User flow
	1.	User uploads labeled dataset.
	2.	Train + validate.
	3.	Deploy model for predictions.

 

6.2 Unsupervised Learning (Clustering example: K-means)

### When to use
	•	No labels; want groups/structure

### Use cases
	•	Customer segmentation
	•	Topic discovery
	•	Anomaly detection (with variants)

### Pseudocode (K-means)

KMeans(X, K):
  initialize centroids μ1..μK
  repeat until stable:
    assign each xi to nearest μk
    update μk = mean(points assigned to k)
  return assignments, centroids

### User flow
	1.	User provides data.
	2.	System groups and summarizes clusters.
	3.	User labels clusters / acts on them.

 

6.3 Reinforcement Learning (Q-learning)

### When to use
	•	Learn by interaction, sparse reward
	•	No fixed labeled dataset

### Use cases
	•	Control, games, scheduling policies

### Pseudocode

QLearning():
  initialize Q(s,a)=0
  for episode in 1..N:
    s = start
    while not terminal:
      a = ε-greedy(Q, s)
      s', r = step(s,a)
      Q(s,a) = Q(s,a) + α*(r + γ*max_a' Q(s',a') - Q(s,a))
      s = s'
  return Q

### User flow
	1.	User defines environment + reward.
	2.	Agent trains via episodes.
	3.	Deploy greedy policy.

 

7) Knowledge Representation & Reasoning (KRR)

7.1 Semantic Networks (Graph Traversal Reasoning)

### When to use
	•	Concepts + relationships; inheritance-like queries

### Use cases
	•	“Is X a type of Y?”
	•	Taxonomies, lightweight ontologies

### Pseudocode

IsA(x, y, graph):
  return reachable(x, y) via edges labeled "is-a"

### User flow
	1.	User asks relationship query.
	2.	System graph-traverses and returns proof path.

 

7.2 Frames (Slot-Filling with Defaults)

### When to use
	•	Structured objects with attributes (slots), defaults, inheritance

### Use cases
	•	Product configs, entity profiles, templates

### Pseudocode

GetSlot(entity, slot):
  if slot in entity: return entity[slot]
  if entity.parent exists: return GetSlot(entity.parent, slot)
  return default(slot)

### User flow
	1.	User requests entity details.
	2.	System resolves slots (direct + inherited).
	3.	Returns filled frame.



7.3 Ontologies (Reasoning over Classes/Relations)

### When to use
	•	Need formal schema with constraints (subclass, domain/range)

### Use cases
	•	Enterprise knowledge graph, compliance, data integration

### Pseudocode (very high-level)

OntologyReason(query):
  load axioms + facts
  run reasoner to compute implied facts
  answer query from closure

### User flow
	1.	User defines schema + facts.
	2.	System runs reasoner.
	3.	Query returns entailed facts + explanation.



7.4 Deductive Reasoning

### When to use
	•	From general rules → certain conclusions

### Pseudocode

Deduce(rules, facts):
  repeat:
    added = false
    for rule (A1∧...∧Ak -> B):
      if all Ai in facts and B not in facts:
        facts.add(B); added=true
  until not added
  return facts

Flow
	1.	Provide rules + facts
	2.	Compute closure
	3.	Return derived facts



7.5 Inductive Reasoning

### When to use
	•	From examples → general rule (uncertain)

### Pseudocode (sketch)

Induce(examples):
  find pattern/rule R that best explains examples
  return R with confidence

Flow
	1.	Provide examples
	2.	Learn rule/model
	3.	Use for prediction



7.6 Abductive Reasoning

### When to use
	•	Infer best explanation for observations

### Pseudocode (sketch)

Abduce(observation, hypotheses, rules):
  best = None
  for h in hypotheses:
    if (rules ∪ {h}) entails observation:
      score = plausibility(h)
      best = argmax(score)
  return best

Flow
	1.	Provide observation
	2.	Generate hypotheses
	3.	Pick most plausible explanation



8) NLP

8.1 Parsing

### When to use
	•	Need syntax tree / structure

### Pseudocode (shift-reduce sketch)

Parse(tokens, grammar):
  stack=[]
  for tok in tokens:
    shift tok to stack
    while can_reduce(stack, grammar):
      reduce(stack)
  return stack_as_parse_tree()

Flow
	1.	User inputs text
	2.	Tokenize + parse
	3.	Output parse tree / structure



8.2 Semantic Analysis

### When to use
	•	Convert text → meaning representation (entities, relations, intent)

### Pseudocode

SemanticAnalyze(text):
  tokens = tokenize(text)
  parse = parse(tokens)
  entities = NER(tokens)
  relations = extract_relations(parse, entities)
  intent = classify_intent(text)
  return {intent, entities, relations}



8.3 Topic Modeling (LDA sketch)

### Pseudocode

LDA(docs, K):
  initialize topic assignments randomly
  repeat:
    for each word in each doc:
      resample topic based on doc-topic and topic-word counts
  return topic-word distributions




8.4 Machine Translation (Seq2Seq sketch)

Translate(src):
  enc = Encoder(src)
  tgt = []
  while not end:
    next_token = DecoderStep(enc, tgt)
    append next_token
  return detokenize(tgt)


 

8.5 Sentiment Analysis

Sentiment(text):
  features = embed(text)
  score = classifier(features)
  return label(score)

NLP ### User flow (common)
	1.	User inputs text
	2.	Preprocess (clean/tokenize)
	3.	Run model / pipeline
	4.	Return structured output + confidence



9) Game Theory / Adversarial Search

9.1 Nash Equilibrium (conceptual compute via best response)

### When to use
	•	Multi-agent strategic interactions

### Pseudocode (best-response iteration sketch)

NashBestResponse(game):
  initialize strategies σ1..σn
  repeat:
    for i in players:
      σi = best_response(i, σ_-i)
  until convergence
  return σ


 

9.2 Dominant Strategy (check)

FindDominantStrategy(player i):
  for strategy s in Si:
    if for all opponent strategies t: payoff(i, s, t) >= payoff(i, s', t) for all s' != s:
      return s
  return none


 

9.3 Minimax

### When to use
	•	Two-player zero-sum games (adversarial planning)

Minimax(state, depth, maximizing):
  if terminal(state) or depth==0:
    return evaluate(state)

  if maximizing:
    best=-∞
    for a in actions(state):
      best = max(best, Minimax(result(state,a), depth-1, false))
    return best
  else:
    best=+∞
    for a in actions(state):
      best = min(best, Minimax(result(state,a), depth-1, true))
    return best




9.4 Alpha-Beta Pruning

AlphaBeta(state, depth, α, β, maximizing):
  if terminal or depth==0: return evaluate(state)

  if maximizing:
    v=-∞
    for a in actions(state):
      v=max(v, AlphaBeta(result(state,a), depth-1, α, β, false))
      α=max(α,v)
      if α>=β: break
    return v
  else:
    v=+∞
    for a in actions(state):
      v=min(v, AlphaBeta(result(state,a), depth-1, α, β, true))
      β=min(β,v)
      if α>=β: break
    return v


 

9.5 Mixed Strategies (compute distribution, sketch)

ComputeMixedStrategy(payoff_matrix):
  solve for probability vector p such that opponent indifferent among best responses
  return p

Game theory flow
	1.	Define players, actions, payoffs
	2.	Compute equilibrium / best policy
	3.	Recommend strategy

 

10) Expert Systems

10.1 Rule-Based Reasoning (Forward Chaining)

### When to use
	•	Clear if/then knowledge; explainability required

### Use cases
	•	Compliance checks
	•	Troubleshooting decision trees
	•	Eligibility & policy systems

### Pseudocode

ForwardChain(rules, facts):
  repeat:
    fired=false
    for rule in rules:
      if rule.conditions ⊆ facts and rule.conclusion ∉ facts:
        facts.add(rule.conclusion)
        fired=true
  until not fired
  return facts

### User flow
	1.	User inputs facts (symptoms, attributes)
	2.	System fires rules
	3.	Returns decision + rule trace



11) Case-Based Reasoning (CBR)

11.1 Retrieval

Retrieve(query, case_base):
  for case in case_base:
    score(case)=similarity(query, case)
  return top_k cases by score

11.2 Reuse

Reuse(retrieved_cases, query):
  propose_solution = adapt(best_case.solution, query)
  return propose_solution

11.3 Revise

Revise(solution, feedback):
  if solution fails:
    solution = repair(solution, feedback)
  return solution

11.4 Retain

Retain(query, final_solution, outcome, case_base):
  new_case = {query, final_solution, outcome}
  case_base.add(new_case)

CBR ### User flow
	1.	User describes problem
	2.	System retrieves similar past cases
	3.	Adapts solution
	4.	Tests / revises
	5.	Stores new solved case



“Where to use what” (fast chooser)
	•	Need shortest path, unweighted → BFS
	•	Need cheapest path, weighted → UCS / A*
	•	Have heuristic + want optimal → A*
	•	Have heuristic + want fast → Greedy
	•	Exact CSP → Backtracking
	•	Huge CSP / approximate → Local search / SA / GA
	•	Continuous differentiable optimization → Gradient descent
	•	Black-box / combinatorial optimization → GA / SA
	•	Uncertainty with causal model → Bayesian networks
	•	Sequential decision under uncertainty → MDPs
	•	Explainable rule decisions → Expert system
	•	Reuse prior solutions → CBR
	•	Adversarial planning → Minimax / Alpha-beta
	•	Text tasks → NLP pipeline (parse/semantic/topic/MT/sentiment)


Tell me which of the 3 you want.
