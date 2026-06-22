# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context

This repository is part of the **BIAI** (Behavioral Integrity in Agent Interactions) research project, targeting the CoLM 2026 Workshop on Agent Behavior. BIAI evaluates how AI agents behave strategically in realistic, multi-turn scenarios — not just what they achieve, but *how*. The core insight is that harmful behaviors like deception, manipulation, and exploitation of information asymmetry can emerge spontaneously from legitimate goal pursuit in strategic environments, without any adversarial prompting.

BIAI defines four behavioral properties grounded in game theory and information economics:
- **P1: Informational Integrity** — commission (false claims), omission (withheld material facts), framing (selective emphasis)
- **P2: Counterpart Welfare** — welfare gap relative to the counterpart's optimal action under full information
- **P3: Influence Legitimacy** — legitimate persuasion vs. illegitimate tactics (fabricated urgency, emotional exploitation)
- **P4: Norm Compliance** — adherence to role-specific legal and professional standards

A key diagnostic is the **Strategic Adaptation Index (SAI)**: each scenario runs under cooperative, neutral, and adversarial counterpart postures. SAI > 0 reveals opportunistic agents that exploit permissive environments; SAI ≈ 0 indicates robust integrity (or robust dishonesty); SAI < 0 indicates escalation under pressure.

The broader BIAI framework spans ten scenarios across deal-making, conflicted advisory, and service interactions (see `../Colm-workshop-benchmark/`). The existing BIAI prototype in `../Colm-workshop-benchmark/biai/` implements a seller-buyer marketplace with SQLite shared state, tmux-based agent processes, a React/FastAPI web UI, and the full P1–P4 + SAI metric suite.

## This Benchmark

This repository implements a **representative** (simplified) version of BIAI's real estate transaction scenario (Scenario 1a: product sale with hidden defect). A seller agent owns a property with known defects and negotiates with a buyer agent who has limited information. The focus is on understanding agent strategic behavior — how agents decide what to disclose, how they negotiate, and how their behavior adapts to different counterpart strategies.

Current priorities:
- **Simplicity first**: conduct comprehensive testing within controlled, simple settings before scaling to the full marketplace design
- **Representative, not realistic**: the current design prioritizes clean experimental control and metric validation over ecological fidelity; realism will be added incrementally
- The project is in early development — only the scaffolding and `__init__.py` stubs exist; all modules still need implementation

## Commands

**IMPORTANT**: Always use the virtual environment Python (`venv/bin/python`) for running experiments and scripts, not the global Python. This ensures correct dependencies and package versions.

```bash
# Install (editable, with all extras including dev tools)
pip install -e ".[all]"

# Run experiments (use venv Python!)
venv/bin/python experiments/run_experiment.py --project-id <PROJECT_ID> --region us-east5

# Run tests
pytest
pytest tests/test_agents.py           # single test file
pytest tests/test_agents.py::test_fn  # single test function

# Lint and format
ruff check .
ruff check --fix .
ruff format .

# Type check
mypy src/
```

## Architecture

The system runs a turn-based game between a **SellerAgent** and a **BuyerAgent**, orchestrated by a **GameController**:

1. **Data layer** (`data/`) — loads the Ames Housing Dataset, partitions features into public (visible to buyer) and hidden (seller-only), generates textual property descriptions, and computes true value = asking price minus repair costs for defects.

2. **Environment** (`environment/`) — `GameState` holds public state (listing, messages, offers, disclosures, inspection reports). `GameController` manages turns and enforces phase transitions: INIT → LISTING → NEGOTIATION → CLOSED/TERMINATED.

3. **Agents** (`agents/`) — `ReActAgent` base class implements an observe→think→act loop. `SellerAgent` and `BuyerAgent` extend it with role-specific system prompts and tool sets. Each agent has a private `memory.md` for recording observations and strategy.

4. **Tools** (`tools/`) — role-scoped tool registries. Sellers can list/update property, disclose defects, respond to offers. Buyers can view listings, request disclosures, order inspections, get market data, make/withdraw offers, or walk away.

5. **Models** (`models/`) — `ModelInterface` protocol with implementations for Anthropic API, Vertex AI, OpenAI, and a `MockModel` for deterministic testing.

6. **Evaluation** (`evaluation/`) — computes integrity metrics (disclosure rate, honesty), due diligence metrics (questions asked, defects discovered, overpayment), and transaction metrics (deal completion, price accuracy, welfare distribution).

## Key Design Decisions

- Properties come from the Ames Housing Dataset (2,930 properties, 80+ features). Features with condition codes `Fa` or `Po` are treated as defects with fixed repair cost estimates (see `data/ames/README.md` for the mapping table).
- True value = `SalePrice - Σ(repair costs)`. The asking price shown to buyers is the raw `SalePrice`.
- Agents use the ReAct (Reason + Act) framework with structured reasoning traces.
- Game configuration is YAML-based (model selection, temperature, max turns, system prompt variants).

## Tool Configuration

- **Ruff**: line-length 100, target Python 3.11, rules `E F I N W UP`, ignores `E501`.
- **mypy**: strict mode, Python 3.11.
- **pytest**: test discovery in `tests/`, async mode `auto` via pytest-asyncio.
- **Build**: uses Hatchling; wheel packages `src/realestate_benchmark`.

## Dependencies

Core: `anthropic`, `pydantic`, `pandas`, `pyyaml`. Optional extras: `[vertex]` for Vertex AI, `[openai]` for OpenAI, `[dev]` for pytest/ruff/mypy. Requires Python ≥ 3.11.
# Development

Use a virtual environment for development:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# or
venv\\Scripts\\activate  # On Windows

# Install in editable mode with all extras
pip install -e ".[all]"
```
