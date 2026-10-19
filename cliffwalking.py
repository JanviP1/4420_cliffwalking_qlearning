"""
Name: Janvi Patel
CPSC 4420: Assignment 2
Cliffwalking environment with tabular Q-learning
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import os
from qlearning import Config, SlipWrapper

def make_env(slip_p=0.0, seed=None, render_mode=None):
    try:
        env = gym.make("CliffWalking-v1", render_mode=render_mode, max_episode_steps=200)
    except Exception:  # older Gymnasium versions only register CliffWalking-v0
        env = gym.make("CliffWalking-v0", render_mode=render_mode, max_episode_steps=200)
    if slip_p > 0:
        env = SlipWrapper(env, p=slip_p, seed=seed)
    return env

"""Initialize the Q-table with zeros."""
def initialize_Q(n_states, n_actions):
    return np.zeros((n_states, n_actions))

"""Return the greedy action for state s based on Q-values."""
def greedy_action(Q, s, rng=None):
    return int(np.argmax(Q[s]))

"""Return an action using epsilon-greedy policy based on Q-values."""
def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return greedy_action(Q, s, rng)

"""Run a single episode of the environment using epsilon-greedy policy derived from Q."""
def run_episode(env, Q, eps, rng):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = epsilon_greedy(Q, state, eps, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    return total_reward

"""Train a tabular Q-learning agent. Returns (Q, per-episode total rewards)."""
def train(env, cfg: Config, seed=0):
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_states, n_actions))  # zero init is optimistic here (all rewards <= 0)
    eps = cfg.eps_start
    episode_rewards = []

    for ep in range(cfg.episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        done, total = False, 0.0
        while not done:
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, terminated, truncated, _ = env.step(a)

            # Terminal handling: no bootstrapping from a true terminal state.
            # A time-limit truncation is NOT terminal, so we still bootstrap.
            target = r if terminated else r + cfg.gamma * Q[s2].max()
            Q[s, a] += cfg.alpha * (target - Q[s, a])

            s, total = s2, total + r
            done = terminated or truncated

        eps = max(cfg.eps_end, eps * cfg.eps_decay)
        episode_rewards.append(total)

    return Q, np.array(episode_rewards)

"""Run the greedy policy (epsilon = 0). Returns dict of metrics."""
def evaluate(Q, env, episodes=30, seed=None):
    rewards, successes, steps_to_goal = [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        done, total, steps = False, 0.0, 0
        terminated = False
        while not done:
            s, r, terminated, truncated, _ = env.step(greedy_action(Q, s))
            total += r
            steps += 1
            done = terminated or truncated

        rewards.append(total)
        successes.append(terminated)
        steps_to_goal.append(steps)

    return {
        "mean_reward": np.mean(rewards),
        "success_rate": np.mean(successes),
        "mean_steps_to_goal": np.mean(steps_to_goal),
    }

"""Record a video of the agent's performance using the greedy policy derived from Q."""
def record_video(Q, video_dir="videos", seed=None):
    env = make_env(slip_p=0.0, seed=seed, render_mode="rgb_array")
    env = RecordVideo(env, video_dir=video_dir, episode_trigger=lambda x: True)
    run_episode(env, Q, 0.0, np.random.default_rng(seed))
    env.close()

def main():
    # Configuration for training
    cfg = Config(
        episodes=500,
        alpha=0.1,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=0.99
    )

    # Create environment
    env = make_env(slip_p=0.0, seed=42)

    # Train the agent
    Q, episode_rewards = train(env, cfg, seed=42)

    # Evaluate the trained agent
    metrics = evaluate(Q, env, episodes=30, seed=42)
    print("Evaluation Metrics:", metrics)

    # Plot 
    window = 20
    smooth = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, alpha=0.3, label="per episode")
    plt.plot(range(window - 1, len(episode_rewards)), smooth, lw=2, label="20-episode average")
    plt.yscale("symlog", linthresh=50)
    plt.xlabel("Training episode"); plt.ylabel("Episode reward"); plt.legend()
    plt.savefig("baseline.png", dpi=150); plt.show()

    # Record a video of the trained agent
    record_video(Q, video_dir="videos", seed=42)

if __name__ == "__main__":
    main()
