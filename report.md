# GridGuard: Reinforcement Learning-Based Power Grid Stabilization System

## 1. Project Overview

**GridGuard** is an intelligent control framework that uses Reinforcement Learning (RL) to detect and prevent cascading failures in electrical power grids. The system acts as an AI co-pilot for grid operators, providing autonomous stabilization recommendations while keeping humans in the decision-making loop for critical interventions.

### Core Objectives
- Predict grid instability before cascading failures occur
- Recommend ranked human interventions based on severity
- Autonomously execute soft control actions (generation redispatch, tap adjustments)
- Maintain grid stability with minimal disruption

---

## 2. How It Works

### 2.1 System Architecture

GridGuard consists of four interconnected layers:

```
┌─────────────────────────────────────────────────────────────┐
│                  GRIDGUARD SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  Simulation     │───▶│  RL Decision Layer          │  │
│  │  Environment   │    │  (PPO Agent)                 │  │
│  │  (Mock Grid)   │    └─────────────────────────────┘  │
│  └─────────────────┘              │                     │
│         │                         │                     │
│         ▼                         ▼                     │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  Early Warning  │───▶│  Recommendation Engine     │  │
│  │  System         │    │  (Tier-based Actions)       │  │
│  └─────────────────┘    └─────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 State Space

The system observes the grid through a 54-dimensional state vector:
- **14 bus voltages** (per unit)
- **14 phase angles** (radians)
- **20 line loadings** (percentage of thermal limit)
- **5 generator outputs** (normalized)
- **1 frequency deviation**

### 2.3 Action Space

The agent can execute two categories of actions:

| Action Type | Description | Disruption Level |
|-------------|-------------|------------------|
| Generator Redispatch | Adjust output of 5 generators | Low |
| Tap Adjustments | Modify transformer tap positions | Low |
| Load Shedding | Reduce non-critical loads | High |
| Islanding | Segment grid for protection | High |

### 2.4 Reward Function

```
R = -L_shed - Penalty_violations + Bonus_stability
```

Where:
- `L_shed`: Total load shed (penalty)
- `Penalty_violations`: Voltage/frequency/thermal violations
- `Bonus_stability`: Reward for maintaining stable operation

Heavy penalties (-1000) are applied for:
- Grid collapse
- Total blackout
- Frequency outside deadband
- Thermal overload

---

## 3. Tools & Technologies

### 3.1 Core Libraries

| Library | Purpose |
|---------|---------|
| **Python 3.12** | Programming language |
| **NumPy** | Numerical computations |
| **Pandas** | Data handling |
| **PyPSA** | Power grid simulation (IEEE 14-bus) |
| **Gymnasium** | RL environment interface |
| **Stable-Baselines3** | PPO RL algorithm implementation |
| **PyTorch** | Deep learning backend |
| **Scikit-learn** | Anomaly detection (Isolation Forest, Random Forest) |

### 3.2 Project Structure

```
ope/
├── GRIDGUARD.txt              # Original specification
├── requirements.txt           # Dependencies
├── saved_model/              # Trained models
└── src/gridguard/
    ├── __init__.py
    ├── environment.py         # IEEE 14-bus grid (PyPSA)
    ├── mock_env.py           # Simplified simulation
    ├── scenario_generator.py  # Synthetic training data
    ├── anomaly_detector.py   # Instability detection
    ├── agent.py              # PPO agent + recommendations
    └── main.py               # Entry point
```

---

## 4. Training Results

### 4.1 Training Progression

The agent was trained using Proximal Policy Optimization (PPO) with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 |
| Batch Size | 64 |
| Gamma | 0.99 |
| PPO Clip Range | 0.2 |
| Network Architecture | [256, 128, 64] |

### 4.2 Performance Over Training Steps

| Training Steps | Avg Eval Reward | Episode Length | Status |
|---------------|-----------------|----------------|--------|
| 10,000 | -130 | 80 (collapses) | Poor |
| 100,000 | +87 | 100 (complete) | Good |
| 200,000 | +100 | 100 (complete) | **Excellent** |

The agent improved from collapsing early to completing all 100 steps with positive rewards.

---

## 5. Complex Scenarios Testing

### 5.1 Scenario Categories

We added three difficulty levels with various failure modes:

| Difficulty | Scenarios |
|------------|-----------|
| **Easy** | load_increase, line_trip, gen_outage |
| **Medium** | load_spike, multi_line_trip, cascading_start |
| **Hard** | multi_gen_outage, voltage_collapse, weather_storm |

### 5.2 Scenario Descriptions

| Scenario | Description | Impact |
|----------|-------------|--------|
| load_increase | Sudden demand spike | Moderate voltage drop |
| line_trip | Transmission line failure | Redistributed loading |
| gen_outage | Generator goes offline | Voltage instability |
| load_spike | Extreme demand surge | High loading |
| multi_line_trip | Multiple line failures | Cascading risk |
| cascading_start | Progressive degradation | System weakening |
| multi_gen_outage | Multiple generators fail | Severe voltage drop |
| voltage_collapse | Terminal voltage collapse | Near-blackout |
| weather_storm | Storm conditions | Combined failures |

### 5.3 Results After Complex Scenario Training

```
=== Final Test Results ===

EASY:   15/15 survived (100%) ████████████████████
MEDIUM: 10/15 survived (67%)  ████████████░░░░░░
HARD:   11/15 survived (73%)   █████████████░░░░░
```

| Difficulty | Survival Rate | Interpretation |
|------------|--------------|----------------|
| **Easy** | 100% | Agent handles basic failures reliably |
| **Medium** | 67% | Handles most moderate events |
| **Hard** | 73% | Survives severe events (including voltage collapse) |

---

## 6. Key Insights

### 6.1 Why It Works

1. **PPO Stability**: Clipped objective prevents drastic policy changes
2. **Reward Shaping**: Gradual penalties teach the agent to avoid collapse
3. **Continuous Actions**: Fine-grained control over grid parameters
4. **Synthetic Data**: Generated training scenarios cover diverse failure modes

### 6.2 Challenges Faced

1. **Sparse Rewards**: Catastrophic failures are rare in real grids
2. **Simulator Gap**: Mock environment differs from real power flow
3. **Multi-objective Tradeoffs**: Safety vs. cost vs. stability

### 6.3 Recommendations for Production

1. Use actual power flow simulator (PyPSA with real IEEE cases)
2. Implement curriculum learning (start easy, gradually increase difficulty)
3. Add human-in-the-loop validation for high-disruption actions
4. Train on real historical outage data
5. Implement ensemble methods for anomaly detection

---

## 7. Conclusion

GridGuard demonstrates that Reinforcement Learning can effectively learn to stabilize power grids under various failure conditions. The system achieves:

- ✅ **100% survival** on basic failure scenarios
- ✅ **Autonomous soft control** for routine stabilization
- ✅ **Tier-based recommendations** for human decision-making
- ✅ **Early warning** through anomaly detection

The complex scenario testing shows the agent can handle severe grid events, though production deployment would require additional training on real grid data and tighter integration with power system simulators.

---

## 8. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python3 -m src.gridguard.main --mode demo

# Train agent
python3 -m src.gridguard.main --mode train --timesteps 100000

# Test with loaded model
python3 -m src.gridguard.main --mode assess
```

---

**Project Version**: 1.0  
**Last Updated**: February 18, 2026  
**Framework**: Stable-Baselines3 (PPO) + Gymnasium + PyPSA
