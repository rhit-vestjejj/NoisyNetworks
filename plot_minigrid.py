"""
Generate all comparison plots for MiniGrid-DoorKey-6x6 DQN / DDQN runs.

For each of the 3 implementations produces 3 plots:
  1. <impl>_baseline.png  — vanilla only
  2. <impl>_noisy.png     — NoisyNet only
  3. <impl>_comparison.png — noisy vs vanilla on the same axes

Plus 6 summary plots:
  4. summary_noisy.png      — DQN-Noisy vs DDQN-Noisy vs PER-Noisy
  5. summary_noisy_2.png    — zoomed
  6. summary_noisy_og.png   — full run
  7. summary_baseline.png   — DQN-Vanilla vs DDQN-Vanilla vs PER-Vanilla
  8. summary_baseline_2.png — zoomed
  9. summary_baseline_og.png — full run

Run after all jobs finish:
  .venv/bin/python plot_minigrid.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RUNS = {
    "dqn":  ("runs/minigrid_base_noisy",  "runs/minigrid_base_baseline"),
    "ddqn": ("runs/minigrid_ddqn_noisy",  "runs/minigrid_ddqn_baseline"),
    "per":  ("runs/minigrid_per_noisy",   "runs/minigrid_per_baseline"),
}

LABELS = {
    "dqn":  "DQN",
    "ddqn": "Double DQN",
    "per":  "DQN + PER",
}

NOISY_COLOR    = "steelblue"
BASELINE_COLOR = "tomato"

IMPL_COLORS = {
    "dqn":  "#2196F3",
    "ddqn": "#FF9800",
    "per":  "#9C27B0",
}

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
loaded = {}
for impl, (noisy_dir, base_dir) in RUNS.items():
    noisy_steps, noisy_rewards = load(noisy_dir)
    base_steps,  base_rewards  = load(base_dir)
    loaded[impl] = {
        "noisy": (noisy_steps, noisy_rewards),
        "base":  (base_steps,  base_rewards),
    }

for impl, data in loaded.items():
    label = LABELS[impl]
    print(f"\n--- {label} ---")
    noisy_steps, noisy_rewards = data["noisy"]
    base_steps,  base_rewards  = data["base"]

    # 1. Baseline only
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot(ax, base_steps, base_rewards, f"Vanilla {label}", BASELINE_COLOR)
    ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
    ax.set_title(f"Vanilla {label} — MiniGrid-DoorKey-6x6")
    ax.legend()
    save(fig, os.path.join(OUT_DIR, f"{impl}_baseline.png"))

    # 2. Noisy only
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot(ax, noisy_steps, noisy_rewards, f"NoisyNet-{label}", NOISY_COLOR)
    ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
    ax.set_title(f"NoisyNet-{label} — MiniGrid-DoorKey-6x6")
    ax.legend()
    save(fig, os.path.join(OUT_DIR, f"{impl}_noisy.png"))

    # 3. Both on same axes
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot(ax, noisy_steps, noisy_rewards, f"NoisyNet-{label}", NOISY_COLOR)
    _plot(ax, base_steps,  base_rewards,  f"Vanilla {label}",  BASELINE_COLOR)
    ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
    ax.set_title(f"NoisyNet vs Vanilla — {label} — MiniGrid-DoorKey-6x6")
    ax.legend()
    save(fig, os.path.join(OUT_DIR, f"{impl}_comparison.png"))

# 4. Summary: all noisy variants
print("\n--- Summary plots ---")
fig, ax = plt.subplots(figsize=(9, 5))
for impl, data in loaded.items():
    steps, rewards = data["noisy"]
    _plot(ax, steps, rewards, f"NoisyNet-{LABELS[impl]}", IMPL_COLORS[impl])
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("NoisyNet-DQN vs NoisyNet-DDQN — MiniGrid-DoorKey-6x6")
ax.set_xlim(left=0)
ax.legend()
save(fig, os.path.join(OUT_DIR, "summary_noisy.png"))

fig, ax = plt.subplots(figsize=(9, 5))
for impl, data in loaded.items():
    steps, rewards = data["noisy"]
    _plot(ax, steps, rewards, f"NoisyNet-{LABELS[impl]}", IMPL_COLORS[impl])
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("NoisyNet-DQN vs NoisyNet-DDQN — MiniGrid-DoorKey-6x6 (zoomed)")
ax.set_xlim(left=0)
ax.legend()
save(fig, os.path.join(OUT_DIR, "summary_noisy_2.png"))

fig, ax = plt.subplots(figsize=(9, 5))
for impl, data in loaded.items():
    steps, rewards = data["noisy"]
    _plot(ax, steps, rewards, f"NoisyNet-{LABELS[impl]}", IMPL_COLORS[impl])
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("NoisyNet-DQN vs NoisyNet-DDQN — MiniGrid-DoorKey-6x6")
ax.set_xlim(left=0)
ax.legend()
save(fig, os.path.join(OUT_DIR, "summary_noisy_og.png"))

# 5. Summary: all baseline variants
fig, ax = plt.subplots(figsize=(9, 5))
for impl, data in loaded.items():
    steps, rewards = data["base"]
    _plot(ax, steps, rewards, f"Vanilla {LABELS[impl]}", IMPL_COLORS[impl])
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("Vanilla DQN vs Vanilla DDQN — MiniGrid-DoorKey-6x6")
ax.set_xlim(left=0)
ax.legend()
save(fig, os.path.join(OUT_DIR, "summary_baseline.png"))

fig, ax = plt.subplots(figsize=(9, 5))
for impl, data in loaded.items():
    steps, rewards = data["base"]
    _plot(ax, steps, rewards, f"Vanilla {LABELS[impl]}", IMPL_COLORS[impl])
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("Vanilla DQN vs Vanilla DDQN — MiniGrid-DoorKey-6x6 (zoomed)")
ax.set_xlim(left=0)
ax.legend()
save(fig, os.path.join(OUT_DIR, "summary_baseline_2.png"))

fig, ax = plt.subplots(figsize=(9, 5))
for impl, data in loaded.items():
    steps, rewards = data["base"]
    _plot(ax, steps, rewards, f"Vanilla {LABELS[impl]}", IMPL_COLORS[impl])
ax.set_xlabel("env step"); ax.set_ylabel("mean reward (last 100 eps)")
ax.set_title("Vanilla DQN vs Vanilla DDQN — MiniGrid-DoorKey-6x6")
ax.set_xlim(left=0)
ax.legend()
save(fig, os.path.join(OUT_DIR, "summary_baseline_og.png"))

print(f"\nAll plots saved to {OUT_DIR}/")
