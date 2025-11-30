# Traffic Light Control using Reinforcement Learning

A reinforcement learning-based traffic light controller for managing a two-way intersection using temporal-difference learning algorithms. The project implements and compares three variants of SARSA to minimize vehicle congestion at a simulated intersection.

## Overview

This project models traffic light control as a Markov Decision Process (MDP) and uses reinforcement learning to learn optimal signal-switching policies. The goal is to minimize vehicle waiting time by intelligently managing green light allocation between two roads.

## MDP Formulation

### State Space
The state at time step *t* is represented as: **x_t = (q₁, q₂, g, δ)**

- **q₁, q₂** ∈ {0, 1, ..., 18}: Queue lengths for Road 1 (East-West) and Road 2 (North-South)
- **g** ∈ {1, 2}: Road currently with green light
- **δ** ∈ {0, 1, ..., 10}: Time steps since last signal switch

Total state space: **7,922 states** (19 × 19 × 2 × 11)

### Action Space
- **a = 0**: Continue current green signal
- **a = 1**: Switch to the other road (only allowed when δ = 10)

A mandatory 10-second buffer enforces safe transitions between signal changes.

### Reward Function
**r(x, a) = -(q₁ + q₂)**

Negative total queue length encourages the agent to minimize congestion.

### Environment Dynamics
- **Arrivals**: Bernoulli process with probabilities p₁ = 0.28, p₂ = 0.4
- **Departures**: Dynamic probability based on signal status
  - 0.9 for green road
  - 0.9(1 - δ²/100) for recently switched road (δ ∈ [0,10])
  - 0 after 10 seconds of red

## Implementation

### Custom Gym Environment
`GymTrafficEnv` implements the traffic intersection with:
- Stochastic vehicle arrivals and departures
- Red-to-green delay modeling
- 30-minute episode duration (1800 time steps)
- Queue length capping at 18 vehicles per road

### Algorithms Implemented

#### 1. SARSA (On-Policy TD Control)
Classical SARSA learning Q-values from actual action sequences:
```
Q(x,a) ← Q(x,a) + α[r + βQ(x',a') - Q(x,a)]
```

#### 2. Expected SARSA
Uses expected Q-values over all actions for more stable learning:
```
Q(x,a) ← Q(x,a) + α[r + β∑P(a'|x')Q(x',a') - Q(x,a)]
```

#### 3. Value Function SARSA
Memory-efficient variant learning only state values V(x), deriving Q-values via simulation:
```
V(x) ← V(x) + α[r + βV(x') - V(x)]
q(x,a) = r(x,a) + β∑P(x'|x,a)V(x')
```

### Key Features
- ε-greedy exploration with softmax-based action probabilities
- State space truncation (queue ≤ 20) for computational efficiency
- Decay schedule: ε starts at 1.0, decays to 0.05
- Learning rate α and discount factor β tuning

## Results

Performance comparison over 1800 time steps:

| Algorithm | Avg. Queue Length | Road 1 Response | Road 2 Response | Balance | Performance |
|-----------|-------------------|-----------------|-----------------|---------|-------------|
| **SARSA** | 23.81 | Fast, frequent dips | Constant saturation | Poor | Moderate |
| **Expected SARSA** | **19.77** | Efficient oscillation | Quick recovery | Good | **Best** |
| **Value SARSA** | 24.30 | Slow adaptation | Severe congestion | Poor | Weakest |

### Key Findings

**Expected SARSA** emerged as the best-performing algorithm:
- Achieved lowest average queue length (19.77)
- Balanced service between both roads
- Quick recovery from congestion spikes
- Robust policy generalization

**SARSA** showed moderate performance but:
- Exhibited bias toward Road 1
- Road 2 frequently saturated at maximum capacity
- On-policy learning led to overfitting on frequently visited states

**Value Function SARSA** underperformed despite memory efficiency:
- Highest congestion (24.30 average)
- Slow response to queue buildup
- Indirect action evaluation resulted in poor real-time adaptability

## Advantages Over Dynamic Programming

This SARSA-based approach offers several benefits over Value/Policy Iteration:

1. **Model-Free Learning**: No need for complete transition probabilities P(x'|x,a)
2. **Online & Incremental**: Updates occur after each interaction, not full state sweeps
3. **Scalability**: More efficient for large state spaces
4. **Practical**: Works with simulators or real-world data without explicit model knowledge

## Project Structure

```
.
├── GymTraffic.py       # Custom Gym environment
├── training.py         # SARSA variants implementation
├── testing.py          # Policy evaluation and visualization
├── policy1.npy         # Trained SARSA policy
├── policy2.npy         # Trained Expected SARSA policy
├── policy3.npy         # Trained Value Function SARSA policy
└── Report.pdf          # Detailed analysis and results
```

## Usage

### Training
```python
# Train algorithms (example)
python training.py
```

### Testing
```python
# Evaluate trained policies
python testing.py
```

The testing script generates:
- Queue length plots for both roads over time
- Action sequence visualization
- Performance metrics comparison

## Requirements

- Python 3.x
- OpenAI Gymnasium
- NumPy
- Matplotlib

## Conclusion

This project demonstrates the effectiveness of temporal-difference learning for real-time traffic control. Expected SARSA's superior performance highlights the importance of exploration strategy and value averaging in dynamic environments. The implementation provides a foundation for more complex traffic management scenarios with multiple intersections or variable traffic patterns.

---

*Reinforcement Learning Assignment - Traffic Light Control*
