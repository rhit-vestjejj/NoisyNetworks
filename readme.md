# NoisyNetworks

Implementation of **"Noisy Networks for Exploration"** — Fortunato et al., DeepMind, ICLR 2018 ([arXiv:1706.10295](https://arxiv.org/abs/1706.10295)).

The paper's idea: replace ε-greedy / entropy-bonus exploration with **learned parametric noise injected into network weights** (`w = μ + σ⊙ε`). σ is learnt by gradient descent so the agent self-tunes how much to explore.

## What's implemented

Three training variants on **ALE/Asterix-v5** — the game where NoisyNet shows the strongest improvements across all algorithm families in the paper:

| Implementation | NoisyNet | Vanilla (baseline) |
|---|---|---|
| DQN | `train_atari.py --algo dqn` | `+ --no-noisy` |
| Double DQN | `train_atari.py --algo dqn --double` | `+ --no-noisy --double` |
| DQN + PER | `train_atari.py --algo dqn --per` | `+ --no-noisy --per` |

When `--no-noisy` is set, DQN/DDQN use **ε-greedy** with linear decay (Mnih 2015).

| Component | File | Paper reference |
|---|---|---|
| `NoisyLinear` layer (factorised + independent noise) | `noisy_linear.py` | Eq. 8–11, §3.2 init |
| Q-net heads: MLP, Atari conv, Dueling MLP, Dueling Atari | `model.py` | §3.1, Eq. 3 |
| Uniform replay buffer | `replay_buffer.py` | Eq. 14 |
| Prioritized replay buffer (sum-tree) | `per_buffer.py` | Schaul et al. 2016 §3.3 |
| Atari preprocessing (DQN-Nature wrappers) | `atari_wrappers.py` | |
| DQN / DDQN / PER training loop | `train_atari.py` | App. C.1 |

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/AutoROM --accept-license   # downloads Atari ROMs
```

## Running experiments

Use `run_sequential.sh` — runs all 4 jobs one at a time, skipping any that already finished:

```bash
chmod +x run_sequential.sh
./run_sequential.sh
```

Or run any single variant directly:

```bash
# NoisyNet-DQN
.venv/bin/python train_atari.py --env-id ALE/Asterix-v5 --out-dir runs/asterix_base_noisy

# Vanilla DQN (ε-greedy)
.venv/bin/python train_atari.py --env-id ALE/Asterix-v5 --no-noisy --out-dir runs/asterix_base_baseline

# NoisyNet-DDQN
.venv/bin/python train_atari.py --env-id ALE/Asterix-v5 --double --out-dir runs/asterix_ddqn_noisy

# Vanilla DDQN
.venv/bin/python train_atari.py --env-id ALE/Asterix-v5 --no-noisy --double --out-dir runs/asterix_ddqn_baseline

# NoisyNet-DQN + PER
.venv/bin/python train_atari.py --env-id ALE/Asterix-v5 --per --out-dir runs/asterix_per_noisy

# Vanilla DQN + PER
.venv/bin/python train_atari.py --env-id ALE/Asterix-v5 --no-noisy --per --out-dir runs/asterix_per_baseline
```

## Generating plots

After all 4 runs finish:

```bash
.venv/bin/python plot_asterix.py
```

Produces 7 plots in `runs/plots/`:

| File | Contents |
|---|---|
| `dqn_baseline.png` | Vanilla DQN learning curve |
| `dqn_noisy.png` | NoisyNet-DQN learning curve |
| `dqn_comparison.png` | Both DQN variants on the same axes |
| `ddqn_baseline.png` | Vanilla DDQN learning curve |
| `ddqn_noisy.png` | NoisyNet-DDQN learning curve |
| `ddqn_comparison.png` | Both DDQN variants on the same axes |
| `per_baseline.png` | Vanilla DQN+PER learning curve |
| `per_noisy.png` | NoisyNet-DQN+PER learning curve |
| `per_comparison.png` | Both DQN+PER variants on the same axes |
| `summary_noisy.png` | NoisyNet-DQN vs NoisyNet-DDQN vs NoisyNet-DQN+PER |
| `summary_baseline.png` | Vanilla DQN vs Vanilla DDQN vs Vanilla DQN+PER |

## Common knobs

| Flag | Meaning | Default |
|---|---|---|
| `--algo` | `dqn` or `dueling` | `dqn` |
| `--noisy` / `--no-noisy` | NoisyNet vs ε-greedy | `--noisy` |
| `--double` | Double DQN target (online selects, target evaluates) | off |
| `--per` | Prioritized Experience Replay (Schaul et al. 2016) | off |
| `--per-alpha` | PER priority exponent α (0=uniform, 1=full priority) | `0.6` |
| `--per-beta` | PER IS weight exponent β start (anneals to 1.0) | `0.4` |
| `--env-id` | Any ALE environment | `ALE/Breakout-v5` |
| `--total-steps` | Agent steps (×4 = frame count) | `50_000_000` |
| `--seed` | RNG seed | `0` |
| `--lr` | Adam learning rate | `6.25e-5` |
| `--eps-decay` | ε decay steps (vanilla only) | `250_000` |

## Why Asterix

Asterix shows the strongest NoisyNet improvement across all algorithm families in the paper:

| Algorithm | Vanilla | NoisyNet | Improvement |
|---|---|---|---|
| DQN | ~6,253 | ~14,328 | +129% |
| Dueling | paper baseline | paper noisy | ~+170% |
| A3C | ~6,822 | ~32,478 | +376% |

Exploration matters heavily in Asterix — the agent must learn to collect specific items while avoiding enemies, and random ε-greedy exploration spends too much time on low-value actions.

## Paper details we follow

- **Factorised** Gaussian noise for DQN/DDQN (Eq. 10–11).
- **ε-greedy removed** in NoisyNet mode (§3.1).
- **Independent noise samples** for online vs target nets per replay step (§3.1).
- **Double DQN target:** online net selects action `b* = argmax_a Q_online(s', a)`, target net evaluates `Q_target(s', b*)` — reduces Q-value overestimation (Van Hasselt et al. 2016).
- **Prioritized Experience Replay:** proportional prioritization `p_i = (|δ_i| + ε)^α`; IS weights `w_i = (N·P(i))^{-β}` correct for sampling bias; β anneals from 0.4 → 1.0 over full training (Schaul et al. 2016 §3.3).
- **Init:** `μ ~ U[-1/√p, 1/√p]`, `σ = σ₀/√p` with σ₀=0.5. §3.2.
- **Logged per layer:** `Σ̄ = mean(|σ_w|)` (Eq. 20) — the paper's Fig. 3 quantity.

## Not in scope

- Multi-thread A3C (too slow single-threaded for meaningful Atari runs).
- Dueling architecture runs (implemented in `model.py`, runnable via `--algo dueling`).
- The 57-game evaluation table.
