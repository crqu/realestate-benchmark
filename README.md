# Real Estate Transaction Benchmark

A benchmark for evaluating AI agent behavior in asymmetric information scenarios through simulated real estate transactions.

## Overview

This benchmark measures how AI agents handle:
- **Information disclosure** - Do sellers reveal material defects?
- **Due diligence** - Do buyers ask the right questions?
- **Negotiation** - Do parties reach fair agreements?
- **Trust and verification** - How do agents balance trust with verification?

## Quick Start

```bash
# Install
pip install -e .

# Run a basic game
python examples/basic_game.py

# Run full benchmark
python examples/run_benchmark.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GAME CONTROLLER                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   PUBLIC GAME STATE                     │
│  Listings │ Messages │ Offers │ Disclosures │ Market   │
└─────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│     SELLER AGENT      │   │     BUYER AGENT       │
│  • Model Interface    │   │  • Model Interface    │
│  • ReAct Engine       │   │  • ReAct Engine       │
│  • memory.md          │   │  • memory.md          │
│  • Seller Tools       │   │  • Buyer Tools        │
│  • Private: Defects   │   │  • Private: Budget    │
└───────────────────────┘   └───────────────────────┘
```

## Key Features

### Data Source
Properties are generated from the **Ames Housing Dataset** with:
- Public features visible to all (size, bedrooms, location)
- Hidden features known only to sellers (condition codes, defects)
- Authentic textual descriptions derived from data values

### Agent Architecture
Each agent follows the **ReAct** (Reasoning + Acting) framework:
1. **Observe** - Gather current state
2. **Think** - Reason about next action
3. **Act** - Execute tool call

### Memory System
Agents maintain private `memory.md` files for:
- Recording observations
- Tracking conversation history
- Planning strategy

## Documentation

- [Design Document](docs/DESIGN.md) - Full architecture and specifications

## Project Structure

```
realestate-benchmark/
├── src/realestate_benchmark/
│   ├── agents/          # ReAct agent implementations
│   ├── models/          # LLM interface adapters
│   ├── environment/     # Game state and controller
│   ├── tools/           # Agent tool definitions
│   ├── data/            # Ames dataset processing
│   └── evaluation/      # Metrics and reporting
├── data/ames/           # Dataset files
├── tests/               # Test suite
└── examples/            # Usage examples
```

## License

MIT License
