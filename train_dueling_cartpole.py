"""
NoisyNet-Dueling on CartPole-v1 — same as train_cartpole.py but uses
NoisyDuelingMLP and the double-DQN target from paper Eq. 15-16.
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

from model import NoisyDuelingMLP
from replay_buffer import ReplayBuffer


def train(env_id="CartPole-v1", total_steps=30_000, buffer_capacity=50_000,
          batch_size=64, gamma=0.99, lr=5e-4, learning_starts=1_000,
          target_update_freq=500, seed=0, out_dir="runs/cartpole_dueling"):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = NoisyDuelingMLP(obs_dim, n_actions).to(device)
    target = NoisyDuelingMLP(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    optim = torch.optim.Adam(online.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_capacity)

    rewards, sig_steps, sig_v, sig_a = [], [], [], []
    obs, _ = env.reset(seed=seed)
    ep_r = 0.0
    t0 = time.time()

    for step in range(1, total_steps + 1):
        online.reset_noise()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(online(obs_t).argmax(dim=1).item())
        next_obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buf.push(obs, a, r, next_obs, float(term))
        obs = next_obs
        ep_r += r
        if done:
            rewards.append(ep_r)
            obs, _ = env.reset()
            ep_r = 0.0

        if len(buf) >= max(batch_size, learning_starts):
            s, ac, rw, sn, te = buf.sample(batch_size, device)

            # Order matters: any reset_noise() between forward and backward
            # would mutate the eps buffer that backward needs.
            # 1) action-selection sample (xi'' in paper)
            with torch.no_grad():
                online.reset_noise()
                b_star = online(sn).argmax(dim=1, keepdim=True)
                # 2) target sample (xi')
                target.reset_noise()
                q_next = target(sn).gather(1, b_star).squeeze(1)
                y = rw + gamma * (1.0 - te) * q_next

            # 3) online sample for the actual gradient update (xi)
            online.reset_noise()
            q_pred = online(s).gather(1, ac.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_pred, y)
            optim.zero_grad(); loss.backward(); optim.step()

        if step % target_update_freq == 0:
            target.load_state_dict(online.state_dict())

        if step % 200 == 0:
            with torch.no_grad():
                sig_steps.append(step)
                sig_v.append(online.head.value_out.weight_sigma.abs().mean().item())
                sig_a.append(online.head.adv_out.weight_sigma.abs().mean().item())
        if step % 2000 == 0:
            recent = rewards[-20:] if rewards else [0.0]
            print(f"step {step:6d} | episodes {len(rewards):4d} | "
                  f"reward(last20) {np.mean(recent):6.1f} | "
                  f"sigma V {sig_v[-1]:.4f} A {sig_a[-1]:.4f} | "
                  f"elapsed {time.time() - t0:5.1f}s")

    env.close()
    np.savez(os.path.join(out_dir, "log.npz"),
             episode_rewards=np.array(rewards),
             sigma_steps=np.array(sig_steps),
             sigma_value=np.array(sig_v),
             sigma_advantage=np.array(sig_a))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="episode reward")
    if len(rewards) >= 20:
        smooth = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        ax.plot(np.arange(len(smooth)) + 19, smooth, label="20-ep moving avg")
    ax.set_xlabel("episode"); ax.set_ylabel("return")
    ax.set_title("NoisyNet-Dueling on CartPole-v1")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sig_steps, sig_v, label="value out")
    ax.plot(sig_steps, sig_a, label="advantage out")
    ax.set_xlabel("training step"); ax.set_ylabel(r"mean $|\sigma_w|$")
    ax.set_title("NoisyNet-Dueling: per-head noise magnitude")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/cartpole_dueling")
    a = p.parse_args()
    train(total_steps=a.total_steps, seed=a.seed, out_dir=a.out_dir)
