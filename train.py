"""
Training Script
===============
Train and compare Q-Learning, SARSA, and REINFORCE on the GridWorld
environment.

Usage
-----
    python train.py                          # train all three agents
    python train.py --agent qlearning        # train only Q-learning
    python train.py --episodes 2000 --plot   # custom episodes with plot
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rl_project.environment import GridWorldEnv
from rl_project.agents import QLearningAgent, SARSAAgent, REINFORCEAgent
from rl_project.utils import moving_average, plot_training


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_qlearning(
    env: GridWorldEnv,
    n_episodes: int = 1000,
    seed: int = 42,
) -> tuple[QLearningAgent, list[float]]:
    """Train a Q-learning agent; return the trained agent and per-episode rewards."""
    agent = QLearningAgent(
        n_states=env.observation_space_n,
        n_actions=env.action_space_n,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        seed=seed,
    )
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=None)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)
    return agent, rewards


def train_sarsa(
    env: GridWorldEnv,
    n_episodes: int = 1000,
    seed: int = 42,
) -> tuple[SARSAAgent, list[float]]:
    """Train a SARSA agent; return the trained agent and per-episode rewards."""
    agent = SARSAAgent(
        n_states=env.observation_space_n,
        n_actions=env.action_space_n,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        seed=seed,
    )
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=None)
        action = agent.select_action(obs)
        total_reward = 0.0
        done = False
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_obs)
            agent.update(obs, action, reward, next_obs, next_action, done)
            obs = next_obs
            action = next_action
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)
    return agent, rewards


def train_reinforce(
    env: GridWorldEnv,
    n_episodes: int = 1000,
    seed: int = 42,
) -> tuple[REINFORCEAgent, list[float]]:
    """Train a REINFORCE agent; return the trained agent and per-episode rewards."""
    agent = REINFORCEAgent(
        n_states=env.observation_space_n,
        n_actions=env.action_space_n,
        learning_rate=0.01,
        gamma=0.99,
        use_baseline=True,
        baseline_lr=0.1,
        seed=seed,
    )
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=None)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward)
            obs = next_obs
            total_reward += reward
        agent.update()
        rewards.append(total_reward)
    return agent, rewards


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def greedy_q_action(agent: QLearningAgent | SARSAAgent, state: int) -> int:
    """Return the greedy action for *state* from a tabular Q-table agent."""
    return int(np.argmax(agent.q_table[state]))


def evaluate(
    env: GridWorldEnv,
    agent: QLearningAgent | SARSAAgent | REINFORCEAgent,
    n_episodes: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate a greedy policy for *n_episodes* episodes.

    Parameters
    ----------
    agent:
        Trained agent.  For Q-learning / SARSA the greedy Q-table action is
        used; for REINFORCE the stochastic policy is sampled.
    """
    use_q_table = isinstance(agent, (QLearningAgent, SARSAAgent))
    rewards = []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            if use_q_table:
                action = greedy_q_action(agent, obs)
            else:
                action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        if total_reward > 0:
            successes += 1
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": successes / n_episodes,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agents on GridWorld.")
    parser.add_argument(
        "--agent",
        choices=["qlearning", "sarsa", "reinforce", "all"],
        default="all",
        help="Which agent to train (default: all).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000).",
    )
    parser.add_argument(
        "--slippery",
        action="store_true",
        help="Enable stochastic transitions (slippery grid).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display training curves after training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    env = GridWorldEnv(is_slippery=args.slippery)
    n_ep = args.episodes
    seed = args.seed

    agents_to_run: list[str] = (
        ["qlearning", "sarsa", "reinforce"] if args.agent == "all" else [args.agent]
    )

    for name in agents_to_run:
        print(f"\n{'='*50}")
        print(f"Training {name.upper()} for {n_ep} episodes …")
        t0 = time.time()

        if name == "qlearning":
            agent, rewards = train_qlearning(env, n_episodes=n_ep, seed=seed)
        elif name == "sarsa":
            agent, rewards = train_sarsa(env, n_episodes=n_ep, seed=seed)
        else:
            agent, rewards = train_reinforce(env, n_episodes=n_ep, seed=seed)

        elapsed = time.time() - t0
        avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        print(f"Done in {elapsed:.1f}s | Last-100 avg reward: {avg:.3f}")

        stats = evaluate(env, agent)
        print(
            f"Eval (100 ep) | mean reward: {stats['mean_reward']:.3f} "
            f"± {stats['std_reward']:.3f} | success rate: {stats['success_rate']:.1%}"
        )

        if args.plot:
            plot_training(rewards, title=f"{name.upper()} – Training rewards")

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
