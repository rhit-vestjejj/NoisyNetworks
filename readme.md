# NoisyNetworks

Implementation of **"Noisy Networks for Exploration"** — Fortunato et al., DeepMind, ICLR 2018 ([arXiv:1706.10295](https://arxiv.org/abs/1706.10295)).

The paper's idea: replace ε-greedy / entropy-bonus exploration with **learned parametric noise injected into network weights** (`w = μ + σ⊙ε`). σ is learnt by gradient descent so the agent self-tunes how much to explore.

## What's implemented

Six training variants — three NoisyNet and the three baselines they're compared against in the paper:

| Algorithm | NoisyNet | Vanilla (baseline) |
|---|---|---|
| DQN | `train_atari.py --algo dqn` / `train_cartpole.py` | + `--no-noisy` |
| Dueling | `train_atari.py --algo dueling` / `train_dueling_cartpole.py` | + `--no-noisy` |
| A3C | `train_a3c.py` | + `--no-noisy` |

All four training scripts accept `--noisy` (default) and `--no-noisy`. When `--no-noisy` is set:
- DQN/Dueling use **ε-greedy** with linear decay (Mnih 2015).
- A3C adds the **entropy bonus** `β·H(π)` to the policy loss (paper Eq. 6).

| Component | File | Paper reference |
|---|---|---|
| `NoisyLinear` layer (factorised + independent noise) | `noisy_linear.py` | Eq. 8–11, §3.2 init |
| Q-net heads: MLP, Atari conv, Dueling MLP, Dueling Atari | `model.py` | §3.1, Eq. 3 |
| A3C model (shared encoder + policy/value heads) | `model.py` | §3.1 (A3C), App. A |
| Replay buffers | `replay_buffer.py`, `train_atari.py` | Eq. 14 (D ~ replay) |
| Atari preprocessing (DQN-Nature wrappers) | `atari_wrappers.py` | |
| Algorithm 1 (DQN / Dueling) | `train_cartpole.py`, `train_dueling_cartpole.py`, `train_atari.py` | App. C.1 |
| Algorithm 2 (A3C, single-thread) | `train_a3c.py` | App. C.2 / App. A |

## Setup (laptop)

```bash
uv venv --python 3.12 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
```

(For CartPole only you can skip the Atari extras: `pip install torch matplotlib gymnasium numpy`.)

## CartPole sanity runs (~10–15 s each on CPU)

```bash
# NoisyNet versions
.venv/bin/python train_cartpole.py           --out-dir runs/dqn_noisy
.venv/bin/python train_dueling_cartpole.py   --out-dir runs/dueling_noisy
.venv/bin/python train_a3c.py                --out-dir runs/a3c_noisy

# Baselines
.venv/bin/python train_cartpole.py         --no-noisy --out-dir runs/dqn_baseline
.venv/bin/python train_dueling_cartpole.py --no-noisy --out-dir runs/dueling_baseline
.venv/bin/python train_a3c.py              --no-noisy --out-dir runs/a3c_baseline
```

Each run drops `rewards.png` (learning curve) and, for noisy runs, `sigma.png` (paper Fig. 3 style) into its `--out-dir`. Baseline runs additionally save `epsilon.png` so you can confirm the schedule.

## Atari (server)

### Option 1 — Docker (recommended)

The image bundles CUDA + PyTorch + the RL stack + ROMs:

```bash
docker build -t noisynet .
```

Then run any variant. The `-v $PWD/runs:/app/runs` mount keeps logs on the host.

```bash
# NoisyNet-DQN on Breakout
docker run --gpus all -v "$PWD/runs:/app/runs" noisynet \
    python train_atari.py --algo dqn --env-id ALE/Breakout-v5 \
    --out-dir runs/breakout_dqn_noisy

# Vanilla DQN baseline (epsilon-greedy)
docker run --gpus all -v "$PWD/runs:/app/runs" noisynet \
    python train_atari.py --algo dqn --no-noisy --env-id ALE/Breakout-v5 \
    --out-dir runs/breakout_dqn_baseline

# NoisyNet-Dueling on Asteroids (paper highlights super-human result)
docker run --gpus all -v "$PWD/runs:/app/runs" noisynet \
    python train_atari.py --algo dueling --env-id ALE/Asteroids-v5 \
    --out-dir runs/asteroids_dueling_noisy

# Vanilla Dueling baseline
docker run --gpus all -v "$PWD/runs:/app/runs" noisynet \
    python train_atari.py --algo dueling --no-noisy --env-id ALE/Asteroids-v5 \
    --out-dir runs/asteroids_dueling_baseline

# NoisyNet-A3C on Beam Rider (paper highlights super-human result)
docker run --gpus all -v "$PWD/runs:/app/runs" noisynet \
    python train_a3c.py --env-id ALE/BeamRider-v5 \
    --total-steps 320000000 --out-dir runs/beamrider_a3c_noisy

# Vanilla A3C baseline
docker run --gpus all -v "$PWD/runs:/app/runs" noisynet \
    python train_a3c.py --no-noisy --env-id ALE/BeamRider-v5 \
    --total-steps 320000000 --out-dir runs/beamrider_a3c_baseline
```

Defaults match the paper hyperparams. Full 200M-frame DQN/Dueling runs take ~1–2 days per game per variant on a single modern GPU.

### Option 2 — bare metal

```bash
pip install -r requirements.txt
AutoROM --accept-license   # downloads Atari ROMs
python train_atari.py --algo dqn --env-id ALE/Breakout-v5 \
    --out-dir runs/breakout_dqn_noisy
```

### Common knobs

| Flag | Meaning | Default |
|---|---|---|
| `--algo` | `dqn` or `dueling` | `dqn` |
| `--noisy` / `--no-noisy` | NoisyNet vs ε-greedy / entropy bonus | `--noisy` |
| `--env-id` | Any ALE env (`ALE/Breakout-v5`, `ALE/Asteroids-v5`, …) | `ALE/Breakout-v5` |
| `--total-steps` | Agent steps (×4 for frame count) | `50_000_000` |
| `--seed` | RNG seed | `0` |
| `--lr` | Adam learning rate | `6.25e-5` |
| `--eps-decay` | ε decay steps (vanilla only) | `250_000` |
| `--entropy-coef` | β for A3C entropy bonus (vanilla only) | `0.01` |

## Paper details we follow

- **Factorised** Gaussian noise for DQN/Dueling (Eq. 10–11), **independent** Gaussian noise for A3C (§3 option a).
- **ε-greedy removed** for NoisyNet DQN/Dueling (§3.1).
- **Entropy bonus removed** for NoisyNet A3C (§3.1).
- **Independent noise samples** for online vs target nets per replay step (§3.1, "to avoid bias").
- **Shared noise across the whole rollout** for A3C (Eq. 25–27, App. A).
- **Init:** factorised: `μ ~ U[-1/√p, 1/√p]`, `σ = σ₀/√p` with σ₀=0.5; independent: `μ ~ U[-√(3/p), √(3/p)]`, `σ = 0.017`. §3.2.
- **Dueling combination:** `Q = V + (A - mean_b A)` (Eq. 3); update is double-DQN (Eq. 4–5, 15–16).
- **Logged per layer:** `Σ̄ = mean(|σ_w|)` (Eq. 20) — the paper's Fig. 3 quantity.

## Not in scope

- Multi-thread A3C. The single-thread loop mirrors the math exactly (Eq. 25–27); to scale, wrap with `torch.multiprocessing` and a shared model.
- Prioritised replay (the paper notes `D` "can be uniform or implementing prioritised replay" — uniform is implemented).
- The 57-game evaluation table — code is here, runs are server work.
