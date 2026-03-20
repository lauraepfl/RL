# RL – Reinforcement Learning Project

A group reinforcement learning project implementing classic tabular RL algorithms
on a configurable GridWorld environment.

## Algorithms

| Algorithm | Type | File |
|-----------|------|------|
| **Q-Learning** | Off-policy TD control | `rl_project/agents/q_learning.py` |
| **SARSA** | On-policy TD control | `rl_project/agents/sarsa.py` |
| **REINFORCE** | Monte-Carlo policy gradient | `rl_project/agents/reinforce.py` |

## Environment

`GridWorldEnv` is a configurable grid navigation task:

```
S F F F
F H F H
F F F H
H F F G
```

* **S** – start position  
* **G** – goal (reward `+1`, episode ends)  
* **H** – hole (reward `−1`, episode ends)  
* **F** – free cell (reward `0`)

Supports optional **stochastic transitions** (slippery grid) and a configurable
`max_steps` truncation limit.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training all agents

```bash
python train.py
```

### Options

```
python train.py --agent qlearning      # only Q-learning
python train.py --agent sarsa          # only SARSA
python train.py --agent reinforce      # only REINFORCE
python train.py --episodes 2000        # custom episode count
python train.py --slippery             # stochastic transitions
python train.py --plot                 # show training curves
python train.py --seed 123             # set random seed
```

### Using the library directly

```python
from rl_project.environment import GridWorldEnv
from rl_project.agents import QLearningAgent

env = GridWorldEnv()
agent = QLearningAgent(
    n_states=env.observation_space_n,
    n_actions=env.action_space_n,
    learning_rate=0.1,
    gamma=0.99,
    seed=42,
)

for episode in range(500):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
    agent.decay_epsilon()
```

## Project Structure

```
RL/
├── rl_project/
│   ├── __init__.py
│   ├── environment.py      # GridWorldEnv
│   ├── utils.py            # moving_average, plot_training, ReplayBuffer
│   └── agents/
│       ├── __init__.py
│       ├── q_learning.py   # QLearningAgent
│       ├── sarsa.py        # SARSAAgent
│       └── reinforce.py    # REINFORCEAgent
├── tests/
│   ├── test_environment.py
│   ├── test_agents.py
│   └── test_utils.py
├── train.py                # CLI training script
├── requirements.txt
└── setup.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```
