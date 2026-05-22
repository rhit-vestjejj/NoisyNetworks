"""
Generate all paper figures for the NoisyNet x PER factorial study.
Outputs:
  runs/plots_factorial/   — 6 figures for the main 2x2 study
  runs/plots_alpha_sweep/ — 2 figures for the alpha sweep
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def iqm(values):
    """Interquartile mean: mean of middle 50%."""
    v = np.sort(values)
    lo, hi = int(len(v) * 0.25), int(np.ceil(len(v) * 0.75))
    return v[lo:hi].mean()


def iqm_ci(values, n=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    v = np.array(values)
    boot = [iqm(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return iqm(v), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

# ── config ────────────────────────────────────────────────────────────────────

ENVS   = ["empty8", "doorkey6", "multiroom"]
ENV_LABELS = {"empty8": "Empty-8×8", "doorkey6": "DoorKey-6×6", "multiroom": "MultiRoom-N2-S4"}
CONDS  = ["noisy_uniform", "noisy_per", "eps_uniform", "eps_per"]
COND_LABELS = {
    "noisy_uniform": "NoisyNet + Uniform",
    "noisy_per":     "NoisyNet + PER",
    "eps_uniform":   "ε-greedy + Uniform",
    "eps_per":       "ε-greedy + PER",
}
COND_COLORS = {
    "noisy_uniform": "#2196F3",
    "noisy_per":     "#F44336",
    "eps_uniform":   "#4CAF50",
    "eps_per":       "#FF9800",
}
SEEDS = [0, 1, 2, 3, 4]
ALPHA_VALS = ["0.4", "0.6", "0.8"]   # 0.6 reused from factorial

FACT_BASE  = "runs/factorial"
ALPHA_BASE = "runs/alpha_sweep"
OUT_FACT   = "runs/plots_factorial"
OUT_ALPHA  = "runs/plots_alpha_sweep"

os.makedirs(OUT_FACT,  exist_ok=True)
os.makedirs(OUT_ALPHA, exist_ok=True)

FIGW = 10
STYLE = {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(STYLE)

# ── data loaders ─────────────────────────────────────────────────────────────

def load_factorial():
    data = {}
    for env in ENVS:
        data[env] = {}
        for cond in CONDS:
            runs = []
            for seed in SEEDS:
                d = np.load(f"{FACT_BASE}/{env}_{cond}_seed{seed}/log.npz")
                runs.append(d)
            data[env][cond] = runs
    return data


def load_alpha_sweep():
    """Returns dict[alpha][cond][seed] = npz.
    alpha in {0.4, 0.6, 0.8}; cond in {noisy_per, eps_per}.
    0.6 is pulled from the factorial."""
    data = {}
    for alpha in ALPHA_VALS:
        data[alpha] = {}
        for cond in ["noisy_per", "eps_per"]:
            runs = []
            for seed in SEEDS:
                if alpha == "0.6":
                    path = f"{FACT_BASE}/doorkey6_{cond}_seed{seed}/log.npz"
                else:
                    path = f"{ALPHA_BASE}/doorkey6_{cond}_a{alpha}_seed{seed}/log.npz"
                runs.append(np.load(path))
            data[alpha][cond] = runs
    return data


def auc(npz):
    steps, rewards = npz["step"], npz["mean_reward"]
    return np.trapezoid(rewards, steps) / (steps[-1] - steps[0])


def bootstrap_ci(values, n=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    vals = np.array(values)
    boot = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return np.mean(vals), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORIAL PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fact = load_factorial()
rng  = np.random.default_rng(42)

# ── Fig 1: IQM bar chart per env ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(FIGW * 1.2, 4), sharey=False)
fig.suptitle("IQM of Mean-Reward AUC — 2×2 Factorial", fontsize=14, fontweight="bold")

for ax, env in zip(axes, ENVS):
    xs, yerr_lo, yerr_hi, colors = [], [], [], []
    for cond in CONDS:
        aucs = np.array([auc(r) for r in fact[env][cond]])
        m, lo, hi = iqm_ci(aucs, rng=rng)
        xs.append(m)
        yerr_lo.append(m - lo)
        yerr_hi.append(hi - m)
        colors.append(COND_COLORS[cond])

    y_pos = np.arange(len(CONDS))
    bars = ax.barh(y_pos, xs, xerr=[yerr_lo, yerr_hi], color=colors,
                   capsize=4, height=0.6, error_kw={"elinewidth": 1.5})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([COND_LABELS[c] for c in CONDS], fontsize=10)
    ax.set_xlabel("IQM AUC", fontsize=11)
    ax.set_title(ENV_LABELS[env], fontsize=12)
    ax.axvline(0, color="gray", lw=0.8, ls="--")

plt.tight_layout()
fig.savefig(f"{OUT_FACT}/fig1_iqm_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig1_iqm_bar.png")

# ── Fig 2: Interaction term plot ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Interaction Term Δ per Environment\n"
             r"$\Delta = (\mathrm{Noisy+PER} - \mathrm{Noisy+Unif}) - (\varepsilon\mathrm{+PER} - \varepsilon\mathrm{+Unif})$",
             fontsize=11)

deltas, lo_errs, hi_errs = [], [], []
for env in ENVS:
    np_aucs  = np.array([auc(r) for r in fact[env]["noisy_per"]])
    nu_aucs  = np.array([auc(r) for r in fact[env]["noisy_uniform"]])
    ep_aucs  = np.array([auc(r) for r in fact[env]["eps_per"]])
    eu_aucs  = np.array([auc(r) for r in fact[env]["eps_uniform"]])

    delta_mean = (np_aucs.mean() - nu_aucs.mean()) - (ep_aucs.mean() - eu_aucs.mean())

    boot = []
    for _ in range(10000):
        idx = rng.integers(0, 5, 5)
        b = (np_aucs[idx].mean() - nu_aucs[idx].mean()) - \
            (ep_aucs[idx].mean() - eu_aucs[idx].mean())
        boot.append(b)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    deltas.append(delta_mean)
    lo_errs.append(delta_mean - lo)
    hi_errs.append(hi - delta_mean)

y_pos = np.arange(len(ENVS))
colors = ["#E53935" if d < 0 else "#43A047" for d in deltas]
ax.barh(y_pos, deltas, xerr=[lo_errs, hi_errs], color=colors,
        capsize=5, height=0.5, error_kw={"elinewidth": 1.8})
ax.set_yticks(y_pos)
ax.set_yticklabels([ENV_LABELS[e] for e in ENVS])
ax.axvline(0, color="black", lw=1.2)
ax.set_xlabel("Δ_interaction (AUC)")
ax.text(0.02, 0.97, "Negative = PER harms NoisyNet more\nthan it harms ε-greedy",
        transform=ax.transAxes, va="top", fontsize=9, color="#555")

plt.tight_layout()
fig.savefig(f"{OUT_FACT}/fig2_interaction_term.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig2_interaction_term.png")

# ── Fig 3: Learning curves per env ───────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(FIGW * 1.3, 4), sharey=False)
fig.suptitle("Learning Curves (mean ± std across 5 seeds)", fontsize=13, fontweight="bold")

for ax, env in zip(axes, ENVS):
    for cond in CONDS:
        runs = fact[env][cond]
        steps   = runs[0]["step"]
        rewards = np.stack([r["mean_reward"] for r in runs])
        mean    = rewards.mean(0)
        std     = rewards.std(0)
        color   = COND_COLORS[cond]
        ax.plot(steps, mean, label=COND_LABELS[cond], color=color, lw=1.8)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_title(ENV_LABELS[env])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean reward (last 100 ep)" if env == "empty8" else "")

handles = [mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS[c]) for c in CONDS]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.12))
plt.tight_layout()
fig.savefig(f"{OUT_FACT}/fig3_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3_learning_curves.png")

# ── Fig 4: σ trajectory split by replay regime (NoisyNet only) ───────────────

fig, axes = plt.subplots(2, 3, figsize=(FIGW * 1.1, 7), sharey=False)
fig.suptitle("NoisyNet σ Trajectory by Layer and Replay Regime", fontsize=13, fontweight="bold")

layer_names = ["fc1", "fc2"]
replay_map  = {"noisy_uniform": ("Uniform", "#2196F3"), "noisy_per": ("PER", "#F44336")}

for col, env in enumerate(ENVS):
    for row, layer_idx in enumerate([0, 1]):
        ax = axes[row][col]
        for cond, (label, color) in replay_map.items():
            runs  = fact[env][cond]
            steps = runs[0]["step"]
            sigs  = np.stack([r["sigma_per_layer"][:, layer_idx] for r in runs])
            mean  = sigs.mean(0)
            std   = sigs.std(0)
            ax.plot(steps, mean, label=label, color=color, lw=1.8)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_title(f"{ENV_LABELS[env]} — {layer_names[layer_idx]}", fontsize=10)
        ax.set_xlabel("Steps" if row == 1 else "")
        ax.set_ylabel("σ (weight std)" if col == 0 else "")
        if row == 0 and col == 0:
            ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(f"{OUT_FACT}/fig4_sigma_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig4_sigma_trajectory.png")

# ── Fig 5: TD-error distributions per condition ───────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(FIGW * 1.2, 4))
fig.suptitle("TD-Error Mean Distribution Over Training (all seeds pooled)", fontsize=13, fontweight="bold")

for ax, env in zip(axes, ENVS):
    for cond in CONDS:
        pooled = np.concatenate([fact[env][cond][s]["td_mean"] for s in range(5)])
        ax.hist(pooled, bins=60, alpha=0.45, color=COND_COLORS[cond],
                label=COND_LABELS[cond], density=True)
    ax.set_title(ENV_LABELS[env])
    ax.set_xlabel("TD-error mean")
    ax.set_ylabel("Density" if env == "empty8" else "")

handles = [mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS[c]) for c in CONDS]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.12))
plt.tight_layout()
fig.savefig(f"{OUT_FACT}/fig5_td_error_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig5_td_error_dist.png")

# ── Fig 6: PER sample-count vs σ-at-insert scatter ────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(FIGW * 1.2, 4))
fig.suptitle("PER Over-Sampling of High-σ Transitions (seed 0)", fontsize=13, fontweight="bold")

per_conds = ["noisy_per", "eps_per"]
scatter_colors = {"noisy_per": "#F44336", "eps_per": "#FF9800"}

for ax, env in zip(axes, ENVS):
    for cond in per_conds:
        diag = np.load(f"{FACT_BASE}/{env}_{cond}_seed0/per_diag.npz")
        sc   = diag["sample_counts"].astype(float)
        sig  = diag["sigma_at_insert"]
        # Downsample for scatter legibility
        idx  = np.random.default_rng(0).choice(len(sc), min(3000, len(sc)), replace=False)
        ax.scatter(sig[idx], sc[idx], alpha=0.25, s=4,
                   color=scatter_colors[cond], label=COND_LABELS[cond])
    ax.set_title(ENV_LABELS[env])
    ax.set_xlabel("σ at insertion")
    ax.set_ylabel("Sample count" if env == "empty8" else "")

handles = [mpatches.Patch(color=scatter_colors[c], label=COND_LABELS[c])
           for c in per_conds]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
fig.savefig(f"{OUT_FACT}/fig6_per_sigma_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig6_per_sigma_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA SWEEP PLOTS
# ══════════════════════════════════════════════════════════════════════════════

alpha_data = load_alpha_sweep()

# ── Fig A1: α-sweep curves (AUC vs α, per exploration regime) ────────────────

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("α-Sweep: PER Prioritization vs Performance\n(DoorKey-6×6, 5 seeds)", fontsize=12)

sweep_colors = {"noisy_per": "#F44336", "eps_per": "#FF9800"}
alpha_floats = [float(a) for a in ALPHA_VALS]

for cond in ["noisy_per", "eps_per"]:
    means, los, his = [], [], []
    for alpha in ALPHA_VALS:
        aucs = np.array([auc(r) for r in alpha_data[alpha][cond]])
        m, lo, hi = bootstrap_ci(aucs, rng=rng)
        means.append(m)
        los.append(lo)
        his.append(hi)
    means = np.array(means)
    los   = np.array(los)
    his   = np.array(his)
    color = sweep_colors[cond]
    label = COND_LABELS[cond]
    ax.plot(alpha_floats, means, "o-", color=color, label=label, lw=2, ms=7)
    ax.fill_between(alpha_floats, los, his, alpha=0.2, color=color)

ax.set_xlabel("PER α (prioritization exponent)")
ax.set_ylabel("Mean AUC (95% bootstrap CI)")
ax.set_xticks(alpha_floats)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUT_ALPHA}/figA1_alpha_sweep_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved figA1_alpha_sweep_curves.png")

# ── Fig A2: Learning curves at each α, NoisyNet vs ε-greedy ──────────────────

fig, axes = plt.subplots(1, 3, figsize=(FIGW * 1.2, 4), sharey=True)
fig.suptitle("Learning Curves by α — DoorKey-6×6", fontsize=13, fontweight="bold")

for ax, alpha in zip(axes, ALPHA_VALS):
    for cond in ["noisy_per", "eps_per"]:
        runs   = alpha_data[alpha][cond]
        steps  = runs[0]["step"]
        rews   = np.stack([r["mean_reward"] for r in runs])
        mean   = rews.mean(0)
        std    = rews.std(0)
        color  = sweep_colors[cond]
        ax.plot(steps, mean, label=COND_LABELS[cond], color=color, lw=1.8)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_title(f"α = {alpha}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean reward" if alpha == "0.4" else "")

handles = [mpatches.Patch(color=sweep_colors[c], label=COND_LABELS[c])
           for c in ["noisy_per", "eps_per"]]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
fig.savefig(f"{OUT_ALPHA}/figA2_alpha_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved figA2_alpha_learning_curves.png")

print("\nDone. All plots saved.")
