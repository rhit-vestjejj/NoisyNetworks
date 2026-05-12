# NoisyNetworks

Implementation of **"Noisy Networks for Exploration"** — Fortunato et al., DeepMind, ICLR 2018 ([arXiv:1706.10295](https://arxiv.org/abs/1706.10295)).

The paper's idea: replace ε-greedy / entropy-bonus exploration with **learned parametric noise injected into network weights** (`w = μ + σ⊙ε`). The σ is learnt by gradient descent, so the agent self-tunes how much to explore.

## What's implemented

| Component | File | Paper reference |
|---|---|---|
| `NoisyLinear` layer (factorised + independent) | `noisy_linear.py` | Eq. 8–11, §3.2 init |
| Q-net heads: MLP, Atari conv, Dueling MLP, Dueling Atari | `model.py` | §3.1, Eq. 3 |
| A3C model (shared encoder + noisy policy/value heads) | `model.py` | §3.1 (A3C), App. A |
| Replay buffers (CartPole + Atari uint8) | `replay_buffer.py`, `train_atari.py` | Eq. 14 (D ~ replay) |
| Atari preprocessing | `atari_wrappers.py` | DQN-Nature wrappers |
| **NoisyNet-DQN training** | `train_cartpole.py`, `train_atari.py --algo dqn` | Algorithm 1, Eq. 14 |
| **NoisyNet-Dueling training** | `train_dueling_cartpole.py`, `train_atari.py --algo dueling` | Algorithm 1, Eq. 15–16 |
| **NoisyNet-A3C training** | `train_a3c.py` | Algorithm 2, Eq. 25–27 |

## Setup (laptop)

```bash
uv venv --python 3.12 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
```

(For CartPole only you can skip the Atari extras — `pip install torch matplotlib gymnasium numpy` is enough.)

## CartPole sanity runs (~10–15 s each on CPU)

```bash
.venv/bin/python train_cartpole.py            # NoisyNet-DQN
.venv/bin/python train_dueling_cartpole.py    # NoisyNet-Dueling (Eq. 15–16)
.venv/bin/python train_a3c.py                 # NoisyNet-A3C (Eq. 25–27)
```

Each run drops `rewards.png` (learning curve) and `sigma.png` (paper Fig. 3 style — shows how σ evolves) into `runs/<name>/`.

## Atari (server)

The fastest path is the Dockerfile, which bundles CUDA + PyTorch + the RL stack + ROMs:

```bash
docker build -t noisynet .
docker run --gpus all -v "$PWD/runs:/app/runs" noisynet \
    python train_atari.py --algo dueling --env-id ALE/Asteroids-v5 \
    --out-dir runs/asteroids_dueling
```

Or install bare-metal:

```bash
pip install -r requirements.txt
AutoROM --accept-license
```

Then:

```bash
# NoisyNet-DQN on Breakout (paper hyperparams; 200M frames = 50M agent steps)
python train_atari.py --algo dqn --env-id ALE/Breakout-v5 \
    --total-steps 50000000 --out-dir runs/breakout_dqn

# NoisyNet-Dueling on Asteroids (paper highlights super-human result)
python train_atari.py --algo dueling --env-id ALE/Asteroids-v5 \
    --total-steps 50000000 --out-dir runs/asteroids_dueling

# NoisyNet-A3C on Beam Rider (paper highlights super-human result)
python train_a3c.py --env-id ALE/BeamRider-v5 \
    --total-steps 320000000 --out-dir runs/beamrider_a3c
```

Defaults match the paper. Adjust `--total-steps`, `--lr`, etc. as needed; full DQN/Dueling runs are GPU-bound and take ~1–2 days per game.

## Paper details we follow

- **Factorised** Gaussian noise for DQN/Dueling (Eq. 10–11), **independent** Gaussian noise for A3C (§3, option a).
- **ε-greedy removed** for DQN/Dueling (§3.1) — exploration comes only from the weight noise.
- **Entropy bonus removed** for A3C (§3.1).
- **Independent noise samples** for online vs target net per replay step (§3.1, "to avoid bias").
- **Shared noise across the whole rollout** for A3C so the policy stays consistent (Eq. 25–27, App. A).
- **Init:** `μ ~ U[-1/√p, 1/√p]`, `σ = σ₀/√p` with σ₀=0.5 (factorised); `μ ~ U[-√(3/p), √(3/p)]`, `σ = 0.017` (independent). §3.2.
- **Dueling combination:** `Q = V + (A - mean_b A)` (Eq. 3); update is double-DQN (Eq. 4–5, 15–16).
- **Logged per layer:** `Σ̄ = mean(|σ_w|)` (Eq. 20) — the paper's Fig. 3 quantity.

## Not in scope

- Multi-thread A3C. The single-thread loop in `train_a3c.py` mirrors the math (Eq. 25–27) exactly; the only thing missing is parallel actor-learners updating shared parameters.
- Prioritised replay (the paper notes the replay distribution `D` "can be uniform or implementing prioritised replay" — uniform is implemented).
- The 57-game Atari evaluation table — code is here, runs are server work.
