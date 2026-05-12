"""
DQN on CartPole-v1, in both noisy and vanilla flavours.

  --noisy     (default): NoisyNet-DQN, paper Algorithm 1 + Eq. 14. Exploration
              is the weight noise; action selection is greedy on the noisy Q.
  --no-noisy            : vanilla DQN, Mnih 2015 with epsilon-greedy. Linearly
              anneal epsilon from --eps-start to --eps-end over --eps-decay
              steps. ε-greedy action selection.
"""

import argparse
import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import NoisyDQN, iter_sigma
from replay_buffer import ReplayBuffer


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = NoisyDQN(obs_dim, n_actions, noisy=args.noisy).to(device)
    target = NoisyDQN(obs_dim, n_actions, noisy=args.noisy).to(device)
    target.load_state_dict(online.state_dict())
    optim = torch.optim.Adam(online.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer_capacity)

    episode_rewards, sig_steps, sig_layers = [], [], []
    obs, _ = env.reset(seed=args.seed)
    ep_r = 0.0
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # ---------- action selection ----------
        if args.noisy:
            # Paper sec 3.1: greedy on the noisy Q. Fresh noise per action.
            online.reset_noise()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(online(obs_t).argmax(dim=1).item())
        else:
            # Vanilla DQN: epsilon-greedy with linear annealing (Mnih 2015).
            eps = max(args.eps_end,
                      args.eps_start - (args.eps_start - args.eps_end) * step / args.eps_decay)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online(obs_t).argmax(dim=1).item())

        next_obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
        buf.push(obs, action, r, next_obs, float(term))
        obs = next_obs
        ep_r += r
        if done:
            episode_rewards.append(ep_r)
            obs, _ = env.reset()
            ep_r = 0.0

        # ---------- learn ----------
        if len(buf) >= max(args.batch_size, args.learning_starts):
            s, a, rw, sn, te = buf.sample(args.batch_size, device)

            # Target first (no_grad), then online for the gradient update.
            # Important when noisy: prevents reset_noise from mutating the
            # eps buffer that the online forward needs for backward.
            with torch.no_grad():
                target.reset_noise()
                q_next = target(sn).max(dim=1).values
                y = rw + args.gamma * (1.0 - te) * q_next

            online.reset_noise()
            q_pred = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_pred, y)
            optim.zero_grad(); loss.backward(); optim.step()

        if step % args.target_update_freq == 0:
            target.load_state_dict(online.state_dict())

        # ---------- logging ----------
        if step % 200 == 0:
            sig_steps.append(step)
            sigmas = list(iter_sigma(online))
            sig_layers.append(sigmas)  # list of per-layer means, may be empty
        if step % 2000 == 0:
            recent = episode_rewards[-20:] if episode_rewards else [0.0]
            extra = f" | sigma {sigmas}" if args.noisy and sigmas else ""
            print(f"step {step:6d} | episodes {len(episode_rewards):4d} | "
                  f"reward(last20) {np.mean(recent):6.1f}{extra} | "
                  f"elapsed {time.time() - t0:5.1f}s")

    env.close()

    # ---------- save logs + plots ----------
    np.savez(os.path.join(args.out_dir, "log.npz"),
             episode_rewards=np.array(episode_rewards),
             sigma_steps=np.array(sig_steps),
             sigma_layers=np.array(sig_layers) if sig_layers and sig_layers[0] else np.array([]))

    title_suffix = "NoisyNet-DQN" if args.noisy else "vanilla DQN"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_rewards, alpha=0.3, label="episode reward")
    if len(episode_rewards) >= 20:
        smooth = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
        ax.plot(np.arange(len(smooth)) + 19, smooth, label="20-ep moving avg")
    ax.set_xlabel("episode"); ax.set_ylabel("return")
    ax.set_title(f"{title_suffix} on CartPole-v1")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "rewards.png"), dpi=120); plt.close(fig)

    if args.noisy and sig_layers and sig_layers[0]:
        sig_arr = np.array(sig_layers)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(sig_arr.shape[1]):
            ax.plot(sig_steps, sig_arr[:, i], label=f"layer {i}")
        ax.set_xlabel("training step"); ax.set_ylabel(r"mean $|\sigma_w|$")
        ax.set_title("Per-layer noise magnitude (paper Eq. 20)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "sigma.png"), dpi=120); plt.close(fig)

    print(f"Done. Logs + plots in {args.out_dir}/")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=30_000)
    p.add_argument("--buffer-capacity", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--target-update-freq", type=int, default=500)
    # NoisyNet toggle.
    p.add_argument("--noisy", dest="noisy", action="store_true", default=True)
    p.add_argument("--no-noisy", dest="noisy", action="store_false")
    # epsilon-greedy schedule (only used when --no-noisy).
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/cartpole_noisynet")
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
