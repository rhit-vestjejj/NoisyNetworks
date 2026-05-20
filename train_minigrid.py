"""
DQN / NoisyDQN on MiniGrid-DoorKey-6x6 (or any MiniGrid env).

  --noisy (default) : NoisyNet exploration (factorised Gaussian noise)
  --no-noisy        : classic epsilon-greedy

Observation:
  The 7×7×3 partially-observable grid image, normalised to [0, 1].
  No frame stacking — a single frame is Markov enough for this task.

Network:
  Small CNN (two Conv2d layers) → two FC layers (noisy or plain).

Default hyperparameters target a CPU run finishing in a few hours:
  - 2M total steps, buffer 100k, batch 64, Adam lr 1e-4
  - learning starts after 10k steps, target sync every 1k steps
"""

import argparse
import os
import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_linear import NoisyLinear
from per_buffer import SumTree


# ---------- env ----------

class _ImageObs(gym.ObservationWrapper):
    """Extract 'image' from MiniGrid dict obs; return (C, H, W) float32 in [0,1]."""
    def __init__(self, env):
        super().__init__(env)
        H, W, C = env.observation_space["image"].shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(C, H, W), dtype=np.float32
        )

    def observation(self, obs):
        img = obs["image"].astype(np.float32) / 10.0   # MiniGrid values are 0–10
        return np.transpose(img, (2, 0, 1))             # (H, W, C) → (C, H, W)


def make_minigrid(env_id, seed=0):
    env = gym.make(env_id)
    env = _ImageObs(env)
    env.action_space.seed(seed)
    return env


# ---------- model ----------

def _linear(in_f, out_f, noisy):
    return NoisyLinear(in_f, out_f, factorised=True) if noisy else nn.Linear(in_f, out_f)


class MiniGridQNet(nn.Module):
    def __init__(self, obs_shape, n_actions, noisy=True):
        super().__init__()
        C, H, W = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, C, H, W)).shape[1]
        self.fc1 = _linear(conv_out, 128, noisy)
        self.fc2 = _linear(128, n_actions, noisy)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(self.conv(x))))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ---------- replay ----------

class SimpleReplay:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.idx = 0
        self.full = False
        self.obs       = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs  = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions   = np.zeros(capacity, dtype=np.int64)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def push(self, s, a, r, s_next, done):
        self.obs[self.idx]       = s
        self.next_obs[self.idx]  = s_next
        self.actions[self.idx]   = a
        self.rewards[self.idx]   = r
        self.terminals[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size, device):
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.as_tensor(self.obs[idx],       device=device),
            torch.as_tensor(self.actions[idx],   device=device),
            torch.as_tensor(self.rewards[idx],   device=device),
            torch.as_tensor(self.next_obs[idx],  device=device),
            torch.as_tensor(self.terminals[idx], device=device),
        )


# ---------- prioritized replay ----------

class PrioritizedReplay:
    """Float32 PER buffer for MiniGrid (uses SumTree from per_buffer)."""

    def __init__(self, capacity, obs_shape, alpha=0.6, eps=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0
        self.obs       = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs  = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions   = np.zeros(capacity, dtype=np.int64)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.tree.n_entries

    def push(self, s, a, r, s_next, terminal):
        data_idx = self.tree.write_idx
        self.obs[data_idx]       = s
        self.next_obs[data_idx]  = s_next
        self.actions[data_idx]   = a
        self.rewards[data_idx]   = r
        self.terminals[data_idx] = terminal
        self.tree.add(self.max_priority)

    def sample(self, batch_size, device, beta):
        total = self.tree.total
        segment = total / batch_size
        tree_indices = np.empty(batch_size, dtype=np.int32)
        data_indices = np.empty(batch_size, dtype=np.int32)
        priorities   = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            ti, di, p = self.tree.get(value)
            tree_indices[i] = ti
            data_indices[i] = di
            priorities[i]   = p
        probs   = priorities / total
        weights = (len(self) * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return (
            tree_indices,
            torch.as_tensor(weights, device=device),
            torch.as_tensor(self.obs[data_indices],      device=device),
            torch.as_tensor(self.actions[data_indices],  device=device),
            torch.as_tensor(self.rewards[data_indices],  device=device),
            torch.as_tensor(self.next_obs[data_indices], device=device),
            torch.as_tensor(self.terminals[data_indices], device=device),
        )

    def update_priorities(self, tree_indices, td_errors):
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for ti, p in zip(tree_indices, priorities):
            self.tree.update(int(ti), float(p))
            if p > self.max_priority:
                self.max_priority = float(p)


# ---------- n-step buffer ----------

class NStepBuffer:
    """Accumulates transitions and emits n-step returns.

    The stored (s, a, G, s_n, done) transition uses:
      G = r_t + γ*r_{t+1} + ... + γ^{k-1}*r_{t+k-1}
    where k = n, or k < n if the episode ends early (done=True, s_n = terminal state).
    The caller should use γ^n as the bootstrap discount so that the (1-done) mask
    correctly zeroes the bootstrap term for early-terminal transitions.
    """

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buf = deque()

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def can_pop(self):
        return len(self.buf) >= self.n

    def pop(self):
        t = self._make()
        self.buf.popleft()
        return t

    def flush(self):
        out = []
        while self.buf:
            out.append(self.pop())
        return out

    def _make(self):
        s0, a0 = self.buf[0][0], self.buf[0][1]
        G = 0.0
        s_n, done_n = self.buf[-1][3], self.buf[-1][4]
        for i, (_, _, r, sn, d) in enumerate(self.buf):
            if i == self.n:
                break
            G += (self.gamma ** i) * r
            s_n, done_n = sn, d
            if d:
                break
        return s0, a0, G, s_n, done_n


# ---------- training ----------

def _eps(step, args):
    frac = min(1.0, step / args.eps_decay)
    return args.eps_start + (args.eps_end - args.eps_start) * frac


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_minigrid(args.env_id, seed=args.seed)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    online = MiniGridQNet(obs_shape, n_actions, noisy=args.noisy).to(device)
    target = MiniGridQNet(obs_shape, n_actions, noisy=args.noisy).to(device)
    target.load_state_dict(online.state_dict())
    optim = torch.optim.Adam(online.parameters(), lr=args.lr)
    if args.per:
        buffer = PrioritizedReplay(args.buffer_capacity, obs_shape, alpha=args.per_alpha)
    else:
        buffer = SimpleReplay(args.buffer_capacity, obs_shape)
    nstep = NStepBuffer(args.n_step, args.gamma)
    gamma_n = args.gamma ** args.n_step

    episode_rewards = deque(maxlen=100)
    log = {"step": [], "mean_reward": [], "sigma": [], "epsilon": []}

    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        if args.noisy:
            online.reset_noise()
            with torch.no_grad():
                action = int(online(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(1).item())
        else:
            if random.random() < _eps(step, args):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(online(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        nstep.push(obs, action, reward, next_obs, float(terminated))
        while nstep.can_pop():
            buffer.push(*nstep.pop())
        obs = next_obs
        ep_reward += reward

        if done:
            for t in nstep.flush():
                buffer.push(*t)
            episode_rewards.append(ep_reward)
            obs, _ = env.reset()
            ep_reward = 0.0

        if len(buffer) >= args.learning_starts and step % args.train_freq == 0:
            if args.per:
                beta = min(1.0, args.per_beta + step * (1.0 - args.per_beta) / args.total_steps)
                tree_idx, weights, s, a, r, s_next, term = buffer.sample(args.batch_size, device, beta)
            else:
                s, a, r, s_next, term = buffer.sample(args.batch_size, device)

            with torch.no_grad():
                if args.double:
                    online.reset_noise()
                    b_star = online(s_next).argmax(dim=1, keepdim=True)
                    target.reset_noise()
                    q_next = target(s_next).gather(1, b_star).squeeze(1)
                else:
                    target.reset_noise()
                    q_next = target(s_next).max(dim=1).values
                y = r + gamma_n * (1.0 - term) * q_next

            online.reset_noise()
            q_pred = online(s).gather(1, a.unsqueeze(1)).squeeze(1)

            if args.per:
                element_loss = F.smooth_l1_loss(q_pred, y, reduction='none')
                loss = (weights * element_loss).mean()
                buffer.update_priorities(tree_idx, (q_pred - y).detach().abs().cpu().numpy())
            else:
                loss = F.smooth_l1_loss(q_pred, y)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optim.step()

        if step % args.target_update_freq == 0:
            target.load_state_dict(online.state_dict())

        if step % args.log_every == 0:
            sigmas = [m.weight_sigma.detach().abs().mean().item()
                      for m in online.modules() if isinstance(m, NoisyLinear)]
            mean_sigma = float(np.mean(sigmas)) if sigmas else 0.0
            mean_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            eps = _eps(step, args) if not args.noisy else 0.0
            log["step"].append(step)
            log["mean_reward"].append(mean_r)
            log["sigma"].append(mean_sigma)
            log["epsilon"].append(eps)
            extra = f"sigma {mean_sigma:.4f}" if args.noisy else f"eps {eps:.3f}"
            print(f"step {step:8d} | reward(last100) {mean_r:7.4f} | {extra} | elapsed {time.time()-t0:6.1f}s")

        if step % args.save_every == 0:
            torch.save(online.state_dict(), os.path.join(args.out_dir, "model.pt"))
            np.savez(os.path.join(args.out_dir, "log.npz"), **log)
            _save_plots(args.out_dir, log, args.noisy)

    env.close()
    np.savez(os.path.join(args.out_dir, "log.npz"), **log)
    _save_plots(args.out_dir, log, args.noisy)


def _save_plots(out_dir, log, noisy):
    label = "NoisyNet-DQN" if noisy else "Vanilla-DQN"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["step"], log["mean_reward"])
    ax.set_xlabel("env step"); ax.set_ylabel("mean episode reward (last 100)")
    ax.set_title(f"{label} on MiniGrid-DoorKey-6x6")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=120); plt.close(fig)

    if noisy:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log["step"], log["sigma"])
        ax.set_xlabel("env step"); ax.set_ylabel(r"mean $|\sigma_w|$")
        ax.set_title(f"{label} noise magnitude")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=120); plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log["step"], log["epsilon"])
        ax.set_xlabel("env step"); ax.set_ylabel("epsilon")
        ax.set_title(f"{label} epsilon schedule")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "epsilon.png"), dpi=120); plt.close(fig)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id",           default="MiniGrid-DoorKey-6x6-v0")
    p.add_argument("--noisy",            dest="noisy", action="store_true", default=True)
    p.add_argument("--no-noisy",         dest="noisy", action="store_false")
    p.add_argument("--total-steps",      type=int,   default=2_000_000)
    p.add_argument("--buffer-capacity",  type=int,   default=100_000)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--learning-starts",  type=int,   default=10_000)
    p.add_argument("--train-freq",       type=int,   default=4)
    p.add_argument("--target-update-freq", type=int, default=1_000)
    p.add_argument("--eps-start",        type=float, default=1.0)
    p.add_argument("--eps-end",          type=float, default=0.05)
    p.add_argument("--eps-decay",        type=int,   default=200_000)
    p.add_argument("--double",           action="store_true", default=False,
                   help="use double-DQN target")
    p.add_argument("--per",             action="store_true", default=False,
                   help="use Prioritized Experience Replay")
    p.add_argument("--per-alpha",        type=float, default=0.6)
    p.add_argument("--per-beta",         type=float, default=0.4)
    p.add_argument("--n-step",           type=int,   default=1,
                   help="n-step return length (1 = standard 1-step TD)")
    p.add_argument("--log-every",        type=int,   default=5_000)
    p.add_argument("--save-every",       type=int,   default=50_000)
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--out-dir",          default="runs/minigrid")
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
