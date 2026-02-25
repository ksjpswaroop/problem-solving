# Examples

This folder contains runnable examples for the currently implemented features.

## Run prerequisites

From repository root:

```bash
pip install -e .
```

If you use symbolic features in examples:

```bash
pip install -e ".[cas,smt,sat]"
```

## Run examples

```bash
PYTHONPATH=src python examples/python/real_world_logistics_planning.py
PYTHONPATH=src python examples/python/constraint_based_scheduling.py
PYTHONPATH=src python examples/python/llm_verified_reasoning.py
PYTHONPATH=src python examples/python/mcp_usage.py
```

## Example overview

- `python/real_world_logistics_planning.py`
  - Delivery route planning on a weighted city graph.
  - Demonstrates algorithm auto-selection and path/cost output.

- `python/constraint_based_scheduling.py`
  - Shift scheduling using CSP and policy consistency check via SAT.

- `python/llm_verified_reasoning.py`
  - OpenAI-compatible LLM connectivity.
  - Uses deterministic solver verification so the final answer is checkable.
  - Falls back to a local deterministic plan if no API key is set.

- `python/mcp_usage.py`
  - Shows MCP manifest discovery and calls to `list_engines`, `select_algorithm`,
    `solve`, `verify`, and `explain_algorithm`.

## LLM environment variables

`llm_verified_reasoning.py` reads:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
