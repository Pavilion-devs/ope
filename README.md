# GridGuard

Reinforcement Learning-Based Power Grid Stabilization System

## Overview

GridGuard is an intelligent control framework that uses Reinforcement Learning (PPO) to detect and prevent cascading failures in electrical power grids. It acts as an AI co-pilot for grid operators, providing autonomous stabilization recommendations while keeping humans in the decision-making loop.

## Features

- **Early Warning System**: Detects grid instability before cascading failures
- **PPO Agent**: Learns to stabilize the grid through reinforcement learning
- **Tier-based Recommendations**: Ranks human interventions by impact and disruption
- **IEEE 14-Bus Simulation**: Built on standard power grid test cases
- **Complex Scenario Training**: Handles multi-component failures, cascading events

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ope

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo
python3 -m src.gridguard.main --mode demo

# Train agent (100k steps)
python3 -m src.gridguard.main --mode train --timesteps 100000

# Interactive assessment
python3 -m src.gridguard.main --mode assess
```

## Project Structure

```
ope/
├── src/gridguard/
│   ├── environment.py         # IEEE 14-bus grid simulation
│   ├── mock_env.py           # Simplified simulation
│   ├── scenario_generator.py  # Synthetic training data
│   ├── anomaly_detector.py   # Instability detection
│   ├── agent.py              # PPO agent
│   └── main.py               # Entry point
├── saved_model/              # Trained models
├── report.md                 # Detailed report
├── README.md
└── requirements.txt
```

## Results

| Scenario | Survival Rate |
|----------|--------------|
| Easy | 100% |
| Medium | 67% |
| Hard | 73% |

## Documentation

See [report.md](report.md) for detailed documentation including:
- System architecture
- How it works
- Training methodology
- Complex scenario testing

## License

MIT License

## Author

Created by Favourolaboye
