"""
NoisyNet-DQN / NoisyNet-Dueling on Atari (paper Algorithm 1, Appendix C.1).

Switch with --algo {dqn,dueling}.

Hyperparameters follow the original DQN-Nature settings used in the paper:
  - replay buffer 1M transitions
  - batch size 32
  - Adam lr 6.25e-5  (paper uses RMSProp 2.5e-4 for vanilla DQN; the NoisyNet
    paper kept the originals — this is configurable)
  - gamma 0.99
  - target update every 32k env steps
  - 200M total frames (= 50M agent steps with frame_skip=4)
  - learning starts after 80k frames

Loss:
  DQN     : Eq. 14    L = E[(r + g * max_b Q(y, b, eps'; zeta-) - Q(x, a, eps; zeta))^2]
  Dueling : Eq. 15-16 L with double-DQN target:
                       b* = argmax_b Q(y, b, eps''; zeta)   (online action select)
                       target uses Q(y, b*, eps'; zeta-)    (target net evaluate)

Run on a GPU server. Locally you can do a smoke test for a few hundred k
frames just to make sure nothing crashes, but it will not solve any game.
"""

import argparse
import os
import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from atari_wrappers import make_atari
from model import NoisyDQNAtari, NoisyDuelingAtari


# ---------- Replay buffer storing uint8 frame stacks (memory-efficient) ----------
class FrameReplay:
    """Stores observations as uint8 to save RAM (1M x 4 x 84 x 84 = ~28 GB
    if float32, ~7 GB if uint8)."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.idx = 0
        self.full = False
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def push(self, s, a, r, s_next, terminal):
        self.obs[self.idx] = s
        self.next_obs[self.idx] = s_next
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.terminals[self.idx] = terminal
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size, device):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.terminals[idx], device=device),
        )


def build_net(algo, n_actions):
    if algo == "dqn":
        return NoisyDQNAtari(n_actions, factorised=True)
    if algo == "dueling":
        return NoisyDuelingAtari(n_actions, factorised=True)
    raise ValueError(algo)


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_atari(args.env_id, seed=args.seed)
    obs_shape = env.observation_space.shape  # (4, 84, 84)
    n_actions = env.action_space.n

    online_net = build_net(args.algo, n_actions).to(device)
    target_net = build_net(args.algo, n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optim = torch.optim.Adam(online_net.parameters(), lr=args.lr, eps=1.5e-4)
    buffer = FrameReplay(args.buffer_capacity, obs_shape)

    # ---- logs ----
    episode_rewards = deque(maxlen=100)
    log = {"step": [], "mean_reward": [], "sigma": []}

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs)
    ep_reward = 0.0
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # Algorithm 1 lines 5-6: sample noise, act greedily.
        online_net.reset_noise()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            action = int(online_net(obs_t).argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.asarray(next_obs)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_reward += reward

        if done:
            episode_rewards.append(ep_reward)
            obs, _ = env.reset()
            obs = np.asarray(obs)
            ep_reward = 0.0

        # Algorithm 1 lines 12-25.
        if len(buffer) >= args.learning_starts and step % args.train_freq == 0:
            s, a, r, s_next, term = buffer.sample(args.batch_size, device)

            # Order: any reset_noise() between online forward and backward
            # mutates the eps buffer that backward needs. So compute the
            # target first, then resample for the online prediction.
            with torch.no_grad():
                if args.algo == "dueling":
                    # Action-selection sample on the online net (xi'' in paper).
                    online_net.reset_noise()
                    b_star = online_net(s_next).argmax(dim=1, keepdim=True)
                    # Target-evaluation sample on the target net (xi').
                    target_net.reset_noise()
                    q_next = target_net(s_next).gather(1, b_star).squeeze(1)
                else:
                    target_net.reset_noise()
                    q_next = target_net(s_next).max(dim=1).values
                y = r + args.gamma * (1.0 - term) * q_next

            # Online sample for the actual gradient update (xi). MUST be last.
            online_net.reset_noise()
            q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            loss = F.smooth_l1_loss(q_pred, y)  # Huber, as in DQN-Nature
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            optim.step()

        if step % args.target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Periodic logging.
        if step % args.log_every == 0:
            with torch.no_grad():
                # Mean |sigma_w| across all noisy layers (paper Eq. 20).
                sigmas = []
                for m in online_net.modules():
                    if hasattr(m, "weight_sigma"):
                        sigmas.append(m.weight_sigma.abs().mean().item())
                mean_sigma = float(np.mean(sigmas))

            mean_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            log["step"].append(step)
            log["mean_reward"].append(mean_r)
            log["sigma"].append(mean_sigma)
            print(
                f"step {step:8d} | reward(last100) {mean_r:7.2f} | "
                f"mean sigma {mean_sigma:.4f} | elapsed {time.time() - t0:6.1f}s"
            )

        if step % args.save_every == 0:
            torch.save(online_net.state_dict(),
                       os.path.join(args.out_dir, "model.pt"))
            np.savez(os.path.join(args.out_dir, "log.npz"), **log)

    env.close()
    np.savez(os.path.join(args.out_dir, "log.npz"), **log)
    plot(args.out_dir, log)


def plot(out_dir, log):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["step"], log["mean_reward"])
    ax.set_xlabel("env step")
    ax.set_ylabel("mean episode reward (last 100)")
    ax.set_title("NoisyNet-DQN/Dueling on Atari")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["step"], log["sigma"])
    ax.set_xlabel("env step")
    ax.set_ylabel(r"mean $|\sigma_w|$ across noisy layers")
    ax.set_title("Noise magnitude over training (paper Fig. 3)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=120)
    plt.close(fig)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="ALE/Breakout-v5")
    p.add_argument("--algo", choices=["dqn", "dueling"], default="dqn")
    p.add_argument("--total-steps", type=int, default=50_000_000,
                   help="agent steps; 200M frames / 4 frame-skip = 50M")
    p.add_argument("--buffer-capacity", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=6.25e-5)
    p.add_argument("--learning-starts", type=int, default=80_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update-freq", type=int, default=8000,
                   help="agent steps; with frame_skip=4 -> 32k frames")
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--save-every", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/atari_noisynet")
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
