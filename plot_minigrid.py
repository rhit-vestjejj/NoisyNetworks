"""
Generate comparison plots for MiniGrid-DoorKey-6x6 NoisyDQN vs vanilla DQN.

Run after both jobs finish:
  .venv/bin/python plot_minigrid.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RUNS = {
    "noisy":    "runs/doorkey_noisy",
    "baseline": "runs/doorkey_baseline",
}

NOISY_COLOR    = "steelblue"
BASELINE_COLOR = "tomato"

OUT_DIR = "runs/plots_minigrid"
os.makedirs(OUT_DIR, exist_ok=True)


def load(path):
    fpath = os.path.join(path, "log.npz")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Missing: {fpath} — has this run finished?")
    d = np.load(fpath)
    return np.array(d["step"]), np.array(d["mean_reward"])


def smooth(arr, w=10):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _plot(ax, steps, rewards, label, color):
    ax.plot(steps, rewards, alpha=0.2, color=color)
    s = smooth(rewards)
    ax.plot(steps[len(steps) - len(s):], s, color=color, linewidth=2, label=label)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


print("Loading run logs...")
noisy_steps, noisy_rewards = load(RUNS["noisy"])
base_steps,  base_rewards  = load(RUNS["baseline"])

# NoisyDQN only
fig, ax = plt.subplots(figsize=(9, 5))
_plot(ax, noisy_steps, noisy_rewards, "NoisyNet-DQN", NOISY_COLOR)
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("NoisyNet-DQN — MiniGrid-DoorKey-6x6")
ax.legend()
save(fig, os.path.join(OUT_DIR, "noisy.png"))

# Vanilla only
fig, ax = plt.subplots(figsize=(9, 5))
_plot(ax, base_steps, base_rewards, "Vanilla DQN", BASELINE_COLOR)
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("Vanilla DQN — MiniGrid-DoorKey-6x6")
ax.legend()
save(fig, os.path.join(OUT_DIR, "baseline.png"))

# Comparison
fig, ax = plt.subplots(figsize=(9, 5))
_plot(ax, noisy_steps, noisy_rewards, "NoisyNet-DQN", NOISY_COLOR)
_plot(ax, base_steps,  base_rewards,  "Vanilla DQN",  BASELINE_COLOR)
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("NoisyNet-DQN vs Vanilla DQN — MiniGrid-DoorKey-6x6")
ax.legend()
save(fig, os.path.join(OUT_DIR, "comparison.png"))

print(f"\nAll plots saved to {OUT_DIR}/")
