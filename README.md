# Oracle

An AI + ML + deterministic causal event-prediction engine.

Oracle generates, scores, and backtests predictions across domains — markets, macroeconomics,
geopolitics, climate, crypto, health, and more — by combining:

- **Deterministic models** — rule-based domain logic (`src/oracle/deterministic/`)
- **Statistical pattern libraries** — recurring historical structures (`src/oracle/patterns/`)
- **Non-deterministic components** — stochastic / ML predictors (`src/oracle/nondeterministic/`)
- **LLM narrative modules** — via an Ollama-compatible API, configured with the `ORACLE_LLM_BASE_URL` and `ORACLE_LLM_API_KEY` environment variables (no keys stored in the repo)
- **Evolution** — LLM-driven evolutionary improvement of predictors (`src/oracle/evolution/`)

## Layout

| Path | Purpose |
|---|---|
| `src/oracle/prediction/` | ~100 prediction modules around a central `engine.py` (causal events, scale bridging, human outcomes, …) |
| `src/oracle/domains/` | 44 domain models |
| `src/oracle/web/` | FastAPI app + dashboard |
| `src/oracle/backtesting/` | Prediction backtesting |
| `src/oracle/optimization/` | Optimizers |
| `src/oracle/ingestion/` | Data ingestion |
| `tests/` | 77 pytest modules |
| `demo.py` | Demo driver |
| `oracle_server.py` | uvicorn launcher (port 8005) |

## Install

Requires Python 3.10+. The package uses a src-layout with setuptools, so an editable install puts everything on your path:

```bash
pip install -e ".[dev]"
pytest
python demo.py
```

`.[dev]` includes pytest and pytest-asyncio; a plain `pip install -e .` installs only the runtime dependencies (numpy, pandas, scipy, scikit-learn, fastapi, uvicorn, pydantic, httpx, jinja2, matplotlib).

## License

MIT