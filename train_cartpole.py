"""
NoisyNet-DQN on CartPole-v1.

Implements Algorithm 1 from the paper (Appendix C.1):
  - epsilon-greedy is REMOVED (paper sec 3.1). Action selection is greedy
    on the noisy Q-network: a = argmax_b Q(x, b, xi; zeta).
  - Loss is paper Eq. 14:
      L = E[(r + gamma * max_b Q(y, b, eps'; zeta-) - Q(x, a, eps; zeta))^2]
    with independently sampled noise for online and target nets.
  - Noise is resampled before every action and before every replay step.
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

from model import NoisyDQN
from replay_buffer import ReplayBuffer


def train(
    env_id="CartPole-v1",
    total_steps=30_000,
    buffer_capacity=50_000,
    batch_size=64,
    gamma=0.99,
    lr=5e-4,
    learning_starts=1_000,
    target_update_freq=500,  # N^- from Algorithm 1
    seed=0,
    out_dir="runs/cartpole_noisynet",
):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online_net = NoisyDQN(obs_dim, n_actions).to(device)
    target_net = NoisyDQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())  # zeta- <- zeta

    optim = torch.optim.Adam(online_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    # Logs.
    episode_rewards = []                       # one entry per episode
    sigma_log_steps = []                       # global step
    sigma_log_fc1, sigma_log_fc2 = [], []      # mean(|sigma_w|) per layer

    # ---------- main loop (Algorithm 1) ----------
    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    t0 = time.time()

    for step in range(1, total_steps + 1):
        # Sample fresh noise (xi ~ epsilon) and act greedily on Q(x, ., xi; zeta).
        online_net.reset_noise()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(online_net(obs_t).argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_reward += reward

        if done:
            episode_rewards.append(ep_reward)
            obs, _ = env.reset()
            ep_reward = 0.0

        # ---------- learn ----------
        if len(buffer) >= max(batch_size, learning_starts):
            s, a, r, s_next, term = buffer.sample(batch_size, device)

            # Independent noise for online & target (paper sec 3.1: avoids bias).
            online_net.reset_noise()
            target_net.reset_noise()

            # Q(x, a, eps; zeta)
            q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # max_b Q(y, b, eps'; zeta-)   (vanilla DQN target; paper Eq. 14)
            with torch.no_grad():
                q_next = target_net(s_next).max(dim=1).values
                y = r + gamma * (1.0 - term) * q_next

            loss = F.mse_loss(q_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Update target network: zeta- <- zeta every N^- steps.
        if step % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Periodically log Sigma_bar (paper Eq. 20) per noisy layer.
        if step % 200 == 0:
            with torch.no_grad():
                sigma1 = online_net.fc1.weight_sigma.abs().mean().item()
                sigma2 = online_net.fc2.weight_sigma.abs().mean().item()
            sigma_log_steps.append(step)
            sigma_log_fc1.append(sigma1)
            sigma_log_fc2.append(sigma2)

        if step % 2000 == 0:
            recent = episode_rewards[-20:] if episode_rewards else [0.0]
            print(
                f"step {step:6d} | episodes {len(episode_rewards):4d} | "
                f"reward(last20) {np.mean(recent):6.1f} | "
                f"sigma fc1 {sigma_log_fc1[-1]:.4f} fc2 {sigma_log_fc2[-1]:.4f} | "
                f"elapsed {time.time() - t0:5.1f}s"
            )

    env.close()

    # ---------- save logs + plots ----------
    np.savez(
        os.path.join(out_dir, "log.npz"),
        episode_rewards=np.array(episode_rewards),
        sigma_steps=np.array(sigma_log_steps),
        sigma_fc1=np.array(sigma_log_fc1),
        sigma_fc2=np.array(sigma_log_fc2),
    )
    plot(out_dir, episode_rewards, sigma_log_steps, sigma_log_fc1, sigma_log_fc2)
    print(f"\nDone. Logs + plots in {out_dir}/")


def plot(out_dir, rewards, steps, sig1, sig2):
    # Fig. 2 style: episode reward (with running mean).
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="episode reward")
    if len(rewards) >= 20:
        kernel = np.ones(20) / 20
        smooth = np.convolve(rewards, kernel, mode="valid")
        ax.plot(np.arange(len(smooth)) + 19, smooth, label="20-ep moving avg")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("NoisyNet-DQN on CartPole-v1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=120)
    plt.close(fig)

    # Fig. 3 style: Sigma_bar per noisy layer (paper Eq. 20).
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, sig1, label="fc1 (penultimate)")
    ax.plot(steps, sig2, label="fc2 (last)")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\bar{\Sigma}$ = mean($|\sigma_w|$)")
    ax.set_title("Per-layer noise magnitude over training (paper Fig. 3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/cartpole_noisynet")
    args = p.parse_args()
    train(total_steps=args.total_steps, seed=args.seed, out_dir=args.out_dir)
