"""
DQN / Dueling DQN on Atari, in both noisy and vanilla flavours.

  --algo dqn            : standard DQN target (Eq. 14 / Mnih 2015).
  --algo dueling        : Dueling with double-DQN target (Eq. 15-16 / Wang 2016).

  --noisy     (default) : NoisyNet exploration.
  --no-noisy            : classic epsilon-greedy with linear decay (Mnih 2015).

Hyperparameters follow the DQN-Nature settings used in the paper:
  - replay 1M, batch 32, Adam lr 6.25e-5
  - gamma 0.99, target update every 8k agent steps (= 32k frames)
  - learning starts after 20k agent steps (= 80k frames)
  - epsilon 1.0 -> 0.01 over the first 250k agent steps (= 1M frames)
  - 50M agent steps total (= 200M frames at frame_skip=4)

Run on a GPU server. Full training takes ~1-2 days per game per variant.
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
from model import NoisyDQNAtari, NoisyDuelingAtari, iter_sigma


class FrameReplay:
    """uint8 replay buffer (memory-efficient: ~7 GB for 1M frame-stacks)."""

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
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.terminals[idx], device=device),
        )


def build_net(algo, n_actions, noisy):
    if algo == "dqn":
        return NoisyDQNAtari(n_actions, noisy=noisy, factorised=True)
    if algo == "dueling":
        return NoisyDuelingAtari(n_actions, noisy=noisy, factorised=True)
    raise ValueError(algo)


def epsilon(step, args):
    """Linear epsilon decay (only used when --no-noisy)."""
    frac = min(1.0, step / args.eps_decay)
    return args.eps_start + (args.eps_end - args.eps_start) * frac


def _nstep_push(replay, buf, gamma):
    """Compute discounted n-step return and push to replay.

    If a terminal occurs mid-window the return is truncated there and no
    bootstrap is needed (done=1 ensures that in the TD target).
    """
    G = 0.0
    last = len(buf) - 1
    for i, (_, _, r, _, done) in enumerate(buf):
        G += gamma ** i * r
        if done:
            last = i
            break
    s0, a0 = buf[0][0], buf[0][1]
    sn, dn = buf[last][3], buf[last][4]
    replay.push(s0, a0, G, sn, dn)


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_atari(args.env_id, seed=args.seed)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    online = build_net(args.algo, n_actions, args.noisy).to(device)
    target = build_net(args.algo, n_actions, args.noisy).to(device)
    target.load_state_dict(online.state_dict())
    optim = torch.optim.Adam(online.parameters(), lr=args.lr, eps=1.5e-4)
    buffer = FrameReplay(args.buffer_capacity, obs_shape)

    episode_rewards = deque(maxlen=100)
    log = {"step": [], "mean_reward": [], "sigma": [], "epsilon": []}

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs)
    ep_reward = 0.0
    nstep_buf = deque()
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # ---------- action selection ----------
        if args.noisy:
            online.reset_noise()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                action = int(online(obs_t).argmax(dim=1).item())
        else:
            eps = epsilon(step, args)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                    action = int(online(obs_t).argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.asarray(next_obs)
        done = terminated or truncated

        nstep_buf.append((obs, action, reward, next_obs, float(terminated)))
        if len(nstep_buf) == args.nstep:
            _nstep_push(buffer, nstep_buf, args.gamma)
            nstep_buf.popleft()

        obs = next_obs
        ep_reward += reward

        if done:
            while nstep_buf:
                _nstep_push(buffer, nstep_buf, args.gamma)
                nstep_buf.popleft()
            episode_rewards.append(ep_reward)
            obs, _ = env.reset()
            obs = np.asarray(obs)
            ep_reward = 0.0

        # ---------- learn ----------
        if len(buffer) >= args.learning_starts and step % args.train_freq == 0:
            s, a, r, s_next, term = buffer.sample(args.batch_size, device)

            with torch.no_grad():
                if args.algo == "dueling" or args.double:
                    # Double-DQN: online selects action, target evaluates it.
                    online.reset_noise()
                    b_star = online(s_next).argmax(dim=1, keepdim=True)
                    target.reset_noise()
                    q_next = target(s_next).gather(1, b_star).squeeze(1)
                else:
                    target.reset_noise()
                    q_next = target(s_next).max(dim=1).values
                y = r + (args.gamma ** args.nstep) * (1.0 - term) * q_next

            # Online sample for gradient (xi noise). Must be the last reset_noise.
            online.reset_noise()
            q_pred = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_pred, y)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optim.step()

        if step % args.target_update_freq == 0:
            target.load_state_dict(online.state_dict())

        if step % args.log_every == 0:
            sigmas = list(iter_sigma(online))
            mean_sigma = float(np.mean(sigmas)) if sigmas else 0.0
            mean_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            eps = epsilon(step, args) if not args.noisy else 0.0
            log["step"].append(step)
            log["mean_reward"].append(mean_r)
            log["sigma"].append(mean_sigma)
            log["epsilon"].append(eps)
            extra = f"sigma {mean_sigma:.4f}" if args.noisy else f"eps {eps:.3f}"
            print(
                f"step {step:8d} | reward(last100) {mean_r:7.2f} | {extra} | "
                f"elapsed {time.time() - t0:6.1f}s"
            )

        if step % args.save_every == 0:
            torch.save(online.state_dict(), os.path.join(args.out_dir, "model.pt"))
            np.savez(os.path.join(args.out_dir, "log.npz"), **log)
            plot(args.out_dir, log, args.algo, args.noisy)

    env.close()
    np.savez(os.path.join(args.out_dir, "log.npz"), **log)
    plot(args.out_dir, log, args.algo, args.noisy)


def plot(out_dir, log, algo, noisy):
    title = f"{'NoisyNet-' if noisy else 'vanilla '}{algo.upper()}"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["step"], log["mean_reward"])
    ax.set_xlabel("env step"); ax.set_ylabel("mean episode reward (last 100)")
    ax.set_title(f"{title} on Atari")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=120); plt.close(fig)

    if noisy:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log["step"], log["sigma"])
        ax.set_xlabel("env step")
        ax.set_ylabel(r"mean $|\sigma_w|$ across noisy layers")
        ax.set_title(f"{title} noise magnitude (paper Fig. 3)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=120); plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log["step"], log["epsilon"])
        ax.set_xlabel("env step"); ax.set_ylabel("epsilon")
        ax.set_title(f"{title} epsilon schedule")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "epsilon.png"), dpi=120); plt.close(fig)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="ALE/Breakout-v5")
    p.add_argument("--algo", choices=["dqn", "dueling"], default="dqn")
    p.add_argument("--noisy", dest="noisy", action="store_true", default=True)
    p.add_argument("--no-noisy", dest="noisy", action="store_false")
    p.add_argument("--nstep", type=int, default=1,
                   help="n-step return horizon (1 = standard 1-step TD)")
    p.add_argument("--double", action="store_true", default=False,
                   help="use double-DQN target for the dqn algo")
    p.add_argument("--total-steps", type=int, default=50_000_000)
    p.add_argument("--buffer-capacity", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=6.25e-5)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update-freq", type=int, default=8_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay", type=int, default=250_000)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--save-every", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/atari")
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
