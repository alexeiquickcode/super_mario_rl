# Super Mario Bros RL Agent

A reinforcement learning agent that learns to play Super Mario Bros with PPO built from scratch.

## Installation (new GPUs) 
```bash
./nightly-install.sh
```

## PPO Mario Policy Network Architecture Design
This module implements a pure CNN-based policy network for Super Mario Bros PPO training.

### Why Pure CNN (No RNNs/LSTMs)?
#### 1. Markov Decision Process Assumption
Mario environments satisfy the Markov property well as each frame contains enough visual information to make good decisions (e.g mario's position, nearby enemies). CNNs have easier to train policies with few parameters making them more stable.

#### 2. Temporal Dependencies Are Handled by PPO Algorithm
The temporal aspects of reinforcement learning are handled by the PPO algorithm itself and not by the network architecture:

**Generalized Advantage Estimation (GAE)**: Works backwards through trajectories to compute advantages, propagating temporal credit assignment:
- Temporal Difference   | δ = r + γV(s') - V(s)
- Advantage Propagation | GAE = δ + γλ * GAE_next

**Value Function Learning**: The critic learns to predict discounted future returns, capturing temporal patterns and long-term consequences.

**Discounted Returns**: γ^t weighting ensures actions consider future rewards, creating temporal dependency through the reward signal.

#### 3. When to use RNNs/LSTMs?
Recurrent architectures are needed when:
- **Partial Observability**: Current frame doesn't contain sufficient information
- **Hidden State**: Important information not visible (e.g., enemy patterns, timers)
- **Long-term Memory**: Decisions require remembering events from many steps ago
- **Temporal Patterns**: Success depends on recognising sequences of states/actions

## Next Steps
- Figure out fireball and big states in reward function
- Trial CNN + LSTM or CNN + Transformer architechtures
- Trial deeper CNN architechtures
- ROMs for Super Mario 2

