"""
Name: Janvi Patel
CPSC 4420: Assignment 2
CliffWalking Environment with Tabular Q-learning
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import os
from dataclasses import dataclass, replace

@dataclass
class Config:
    alpha: float = 0.1        # learning rate
    gamma: float = 0.99       # discount factor
    eps_start: float = 1.0    # initial exploration rate
    eps_end: float = 0.1      # exploration floor
    eps_decay: float = 0.99   # multiplied into epsilon after each episode
    episodes: int = 500

class SlipWrapper(gym.ActionWrapper):
    """With probability p, replace the intended action with a different one."""
    def __init__(self, env, p=0.0, seed=None):
        super().__init__(env)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def action(self, action):
        if self.p > 0 and self.rng.random() < self.p:
            others = [a for a in range(self.action_space.n) if a != action]
            return int(self.rng.choice(others))
        return action

def make_env(slip_p=0.0, seed=None, render_mode=None):
    try:
        env = gym.make("CliffWalking-v1", render_mode=render_mode, max_episode_steps=200)
    except Exception:  # older Gymnasium versions only register CliffWalking-v0
        env = gym.make("CliffWalking-v0", render_mode=render_mode, max_episode_steps=200)
    if slip_p > 0:
        env = SlipWrapper(env, p=slip_p, seed=seed)
    return env

def initialize_Q(n_states, n_actions):
    """Initialize the Q-table with zeros."""
    return np.zeros((n_states, n_actions))

def greedy_action(Q, s, rng=None):
    """Return the greedy action for state s based on Q-values."""
    return int(np.argmax(Q[s]))

def epsilon_greedy(Q, s, eps, rng):
    """Return an action using epsilon-greedy policy based on Q-values."""
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return greedy_action(Q, s, rng)

def run_episode(env, Q, eps, rng):
    """Run a single episode of the environment using epsilon-greedy policy derived from Q."""
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

def train(env, cfg: Config, seed=0):
    """Train a tabular Q-learning agent. Returns (Q, per-episode total rewards)."""
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

def evaluate(Q, env, episodes=30, seed=None):
    """Run the greedy policy (epsilon = 0). Returns dict of metrics."""
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
        if terminated:
            steps_to_goal.append(steps)

    return {
        "mean_reward": np.mean(rewards),
        "success_rate": np.mean(successes),
        "mean_steps_to_goal": np.mean(steps_to_goal) if steps_to_goal else float("nan"),
    }

def save_results(filename, results_dir="results"):
    """Return a path inside the results folder, creating the folder if needed."""
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)

def plot_results(episode_rewards, window=20, title ="Baseline Q-learning on CliffWalking", filename="baseline.png"):
    """Plot reward per training episode with a moving average."""
    smooth = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, alpha=0.3, label="per episode")
    plt.plot(range(window - 1, len(episode_rewards)), smooth, lw=2, label=f"{window}-episode average")
    plt.yscale("symlog", linthresh=50)
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_results(filename), dpi=150)
    plt.close()

def record_video(Q, video_dir="videos", seed=None, name_prefix="cliffwalking_greedy"):
    """Record a video of the agent's performance using the greedy policy derived from Q."""
    os.makedirs(video_dir, exist_ok=True)
    # clear old recordings so the folder is clean
    for f in os.listdir(video_dir):
        if f.endswith(".mp4"):
            os.remove(os.path.join(video_dir, f))
    env = make_env(slip_p=0.0, seed=seed, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True,
                      name_prefix=name_prefix)
    run_episode(env, Q, 0.0, np.random.default_rng(seed))
    env.close()

def plot_comparison(curves, title, filename, window=20):
    """Plot multiple curves with moving averages for comparison."""
    plt.figure(figsize=(8, 5))
    for label, runs in curves.items():
        runs = np.array(runs)
        smooth = np.array([np.convolve(r, np.ones(window) / window, mode="valid") for r in runs])
        mean, std = smooth.mean(axis=0), smooth.std(axis=0)
        plt.plot(range(window - 1, runs.shape[1]), mean, label=label)
        plt.fill_between(range(window - 1, runs.shape[1]), mean - std, mean + std, alpha=0.15)
    plt.yscale("symlog", linthresh=50)
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward (20-episode moving average)")
    plt.title(title + f" (mean ± std over {len(runs)} seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_results(filename), dpi=150)
    plt.close()

def run_sweep(param, values, base, seeds=range(10), eval_episodes=30):
    """Train each setting on several seeds. Returns curves and a summary table."""
    curves, rows = {}, []
    for v in values:
        cfg = replace(base, **{param: v})
        runs, evals = [], []
        for s in seeds:
            Q, rewards = train(make_env(seed=s), cfg, seed=s)
            runs.append(rewards)
            evals.append(evaluate(Q, make_env(seed=1000 + s), episodes=eval_episodes))
        curves[f"{param}={v}"] = runs
        rows.append((f"{param}={v}",
                     np.mean([e["mean_reward"] for e in evals]),
                     np.mean([e["success_rate"] for e in evals]),
                     np.mean([r[-100:].mean() for r in runs])))
    return curves, rows

def print_table(rows):
    print(f"{'setting':<18}{'eval reward':>12}{'success':>9}{'train last100':>15}")
    for name, ev, sr, tr in rows:
        print(f"{name:<18}{ev:>12.1f}{sr:>9.2f}{tr:>15.1f}")
    print()

def episodes_to_threshold(rewards, threshold=-30, window=20):
    """Return the first episode where the moving average of rewards exceeds the threshold (or nan if never)."""    
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    above_threshold = np.where(smooth > threshold)[0]
    return above_threshold[0] + window if len(above_threshold) > 0 else float("nan")

def compare_configs(configs, seeds=range(10), eval_episodes=100):
    """Train each named config on the same seeds and report eval + learning speed."""
    print(f"{'config':<12}{'eval reward':>12}{'success':>9}{'steps':>7}{'train last100':>15}{'eps to -30':>12}")
    for name, cfg in configs.items():
        evals, runs = [], []
        for s in seeds:
            Q, rewards = train(make_env(seed=s), cfg, seed=s)
            runs.append(rewards)
            evals.append(evaluate(Q, make_env(seed=1000 + s), episodes=eval_episodes))
        print(f"{name:<12}"
              f"{np.mean([e['mean_reward'] for e in evals]):>12.1f}"
              f"{np.mean([e['success_rate'] for e in evals]):>9.2f}"
              f"{np.nanmean([e['mean_steps_to_goal'] for e in evals]):>7.1f}"
              f"{np.mean([r[-100:].mean() for r in runs]):>15.1f}"
              f"{np.nanmean([episodes_to_threshold(r) for r in runs]):>12.0f}")

def run_noise_experiment(cfg, noise_levels=(0.0, 0.10, 0.25), seeds=range(10), eval_episodes=100):
    """Train each noise level on several seeds. Returns curves and a summary table."""
    curves, Qs, rows, robust_rows = {}, {}, [], []

    #Train and evaluate the same noise level
    for p in noise_levels:
        runs, evals = [], []
        for s in seeds:
            Q, rewards = train(make_env(slip_p=p, seed=s), cfg, seed=s)
            Qs[(p,s)] = Q
            runs.append(rewards)
            evals.append(evaluate(Q, make_env(slip_p=p, seed=1000 + s), episodes=eval_episodes))
        curves[f"p={p}"] = runs
        rows.append((f"trained_and_tested p={p}",
                     np.mean([e["mean_reward"] for e in evals]),
                     np.mean([e["success_rate"] for e in evals]),
                     np.mean([r[-100:].mean() for r in runs])))

    #Policy robustness: evaluate each trained policy on all noise levels
    for p_test in noise_levels:
        evals = [evaluate(Qs[(0.0, s)], make_env(slip_p=p_test, seed=2000 + s), episodes=eval_episodes)
                 for s in seeds]
        robust_rows.append((f"p=0 policy, test p={p_test}",
                            np.mean([e["mean_reward"] for e in evals]),
                            np.mean([e["success_rate"] for e in evals]),
                            float("nan")))
    return curves, rows, robust_rows

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
    eval_env = make_env(seed=43)
    metrics = evaluate(Q, eval_env, episodes=30, seed=43)
    print("Evaluation Metrics:", metrics)

    # Plot 
    plot_results(episode_rewards)

    base = Config()
    sweeps = [
        ("alpha", [0.1, 0.5, 0.9], "Effect of learning rate α", "alpha_sweep.png"),
        ("gamma", [0.5, 0.9, 0.99], "Effect of discount factor γ", "gamma_sweep.png"),
        ("eps_decay", [0.95, 0.99, 0.999], "Effect of ε-decay", "epsdecay_sweep.png"),
    ]
    for param, values, title, fname in sweeps:
        curves, rows = run_sweep(param, values, base)
        plot_comparison(curves, title, fname)
        print_table(rows)

    # Record a video of the trained agent
    record_video(Q, video_dir="videos", seed=42)
    
    best = Config(alpha=0.9, gamma=0.9, eps_decay=0.95)
    compare_configs({"baseline": Config(), "best": best})

    # Retrain the best agent and record the retrained agent
    Q_best, _ = train(make_env(seed=42), best, seed=42)
    record_video(Q_best, video_dir="videos", seed=42, name_prefix="cliffwalking_retrained")

    # Run noise experiment to evaluate robustness of the best agent
    curves, rows, robust_rows = run_noise_experiment(best)
    plot_comparison(curves, "Effect of slip probability p", "noise_curves.png")
    print("Trained and tested at same noise level:")
    print_table(rows)
    print("Policy trained at p=0, tested under noise:")
    print_table(robust_rows)    

if __name__ == "__main__":
    main()

