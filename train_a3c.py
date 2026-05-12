"""
NoisyNet-A3C (paper sec 3.1 / Appendix A / Algorithm 2 in C.2).

This is a single-thread implementation that mirrors the math exactly:

  Eq. 25:  Q_hat_i = sum_{j=i..k-1} g^{j-i} r_{t+j} + g^{k-i} V(x_{t+k}; zeta_V, eps)
  Eq. 26:  zeta_pi <- zeta_pi + a_pi * sum_i grad log pi(a_i|x_i; zeta_pi, eps) [Q_hat_i - V(x_i; zeta_V, eps)]
  Eq. 27:  zeta_V  <- zeta_V  - a_V  * sum_i grad (Q_hat_i - V(x_i; zeta_V, eps))^2

Two paper-mandated details:
  - **No entropy bonus** (sec 3.1).
  - **Noise is fixed for the whole rollout** so the policy stays consistent
    (Eq. 25's `eps_i = eps`). We call reset_noise() at the START of each
    rollout only.

The full multi-thread A3C runs N copies of this loop in parallel against
shared parameters. Single-thread keeps things faithful to the math while
letting us run on a laptop. To go multi-thread, wrap this with
torch.multiprocessing and a shared model.
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

from atari_wrappers import make_atari
from model import make_a3c_atari, make_a3c_mlp


def is_atari(env_id):
    return env_id.startswith("ALE/")


def make_env(env_id, seed):
    if is_atari(env_id):
        return make_atari(env_id, seed=seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    return env


def make_model(env, hidden=128):
    if len(env.observation_space.shape) == 3:
        return make_a3c_atari(env.action_space.n, hidden=256)
    return make_a3c_mlp(env.observation_space.shape[0], env.action_space.n, hidden=hidden)


def to_tensor(obs, device):
    if obs.dtype == np.uint8:
        return torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env_id, args.seed)
    model = make_model(env).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    episode_rewards = []
    log = {"step": [], "mean_reward": [], "sigma_policy": [], "sigma_value": []}

    t0 = time.time()
    step = 0

    while step < args.total_steps:
        # ---------- start of a rollout ----------
        # Algorithm 2 line 7: pick the noise once and keep it fixed for the
        # whole rollout (paper Eq. 25-27).
        model.reset_noise()

        log_probs, values, rewards, entropies, dones = [], [], [], [], []

        for _ in range(args.rollout_size):
            obs_t = to_tensor(obs, device)
            logits, value = model(obs_t)
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)

            # Sample action from the noisy policy.
            action = int(torch.multinomial(probs, num_samples=1).item())
            log_prob = log_probs_all[0, action]

            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += r

            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(float(r))
            dones.append(float(terminated))
            entropies.append(-(probs * log_probs_all).sum())  # logged but unused

            obs = next_obs
            step += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()
                break  # cut the rollout short on episode boundary

        # ---------- bootstrap target (Eq. 25) ----------
        # If we exited because the env terminated, V(terminal) = 0, else use the
        # current value estimate with the SAME noise sample.
        if dones and dones[-1] == 1.0:
            R = torch.zeros(1, device=device)
        else:
            with torch.no_grad():
                _, R = model(to_tensor(obs, device))
                R = R.detach()

        # ---------- compute losses ----------
        policy_loss = torch.zeros(1, device=device)
        value_loss = torch.zeros(1, device=device)

        # Walk the rollout backwards, accumulating Q_hat_i.
        for i in reversed(range(len(rewards))):
            R = rewards[i] + args.gamma * R * (1.0 - dones[i])
            advantage = R - values[i]
            # Eq. 26: policy gradient (no entropy term — paper sec 3.1).
            policy_loss = policy_loss - log_probs[i] * advantage.detach()
            # Eq. 27: value MSE.
            value_loss = value_loss + advantage.pow(2)

        loss = policy_loss + args.value_coef * value_loss

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optim.step()

        # ---------- logging ----------
        if step // args.log_every > (step - len(rewards)) // args.log_every:
            with torch.no_grad():
                sp = model.policy.weight_sigma.abs().mean().item()
                sv = model.value.weight_sigma.abs().mean().item()
            recent = episode_rewards[-20:] if episode_rewards else [0.0]
            log["step"].append(step)
            log["mean_reward"].append(float(np.mean(recent)))
            log["sigma_policy"].append(sp)
            log["sigma_value"].append(sv)
            print(
                f"step {step:7d} | episodes {len(episode_rewards):4d} | "
                f"reward(last20) {np.mean(recent):6.1f} | "
                f"sigma pi {sp:.4f} V {sv:.4f} | "
                f"elapsed {time.time() - t0:6.1f}s"
            )

    env.close()
    np.savez(os.path.join(args.out_dir, "log.npz"),
             episode_rewards=np.array(episode_rewards), **log)
    plot(args.out_dir, episode_rewards, log)


def plot(out_dir, episode_rewards, log):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_rewards, alpha=0.3, label="episode reward")
    if len(episode_rewards) >= 20:
        smooth = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
        ax.plot(np.arange(len(smooth)) + 19, smooth, label="20-ep moving avg")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("NoisyNet-A3C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["step"], log["sigma_policy"], label="policy head")
    ax.plot(log["step"], log["sigma_value"], label="value head")
    ax.set_xlabel("env step")
    ax.set_ylabel(r"mean $|\sigma_w|$")
    ax.set_title("NoisyNet-A3C noise magnitude (paper Fig. 3 style)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=120)
    plt.close(fig)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--rollout-size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/a3c_noisynet")
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
