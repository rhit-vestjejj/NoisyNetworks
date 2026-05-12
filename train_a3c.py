"""
A3C in both noisy and vanilla flavours.

  --noisy     (default): NoisyNet-A3C. Paper sec 3.1 / App A / Algorithm 2.
              No entropy bonus, weight noise fixed for the whole rollout
              (Eq. 25-27).
  --no-noisy            : vanilla A3C (Mnih 2016). Entropy bonus
              `beta * H(pi)` added to the policy loss (paper Eq. 6).

Single-thread implementation that mirrors the math exactly:
  Eq. 25:  Q_hat_i = sum_{j=i..k-1} g^{j-i} r_{t+j} + g^{k-i} V(x_{t+k})
  Eq. 26:  zeta_pi <- zeta_pi + a_pi * sum_i grad log pi(a_i|x_i) * advantage
  Eq. 27:  zeta_V  <- zeta_V  - a_V  * sum_i grad (advantage)^2
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
from model import make_a3c_atari, make_a3c_mlp, iter_sigma


def is_atari(env_id):
    return env_id.startswith("ALE/")


def make_env(env_id, seed):
    if is_atari(env_id):
        return make_atari(env_id, seed=seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    return env


def make_model(env, noisy, hidden=128):
    if len(env.observation_space.shape) == 3:
        return make_a3c_atari(env.action_space.n, hidden=256, noisy=noisy)
    return make_a3c_mlp(env.observation_space.shape[0], env.action_space.n,
                        hidden=hidden, noisy=noisy)


def to_tensor(obs, device):
    if obs.dtype == np.uint8:
        return torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env_id, args.seed)
    model = make_model(env, noisy=args.noisy).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    episode_rewards = []
    log = {"step": [], "mean_reward": [], "sigma_policy": [], "sigma_value": []}

    t0 = time.time()
    step = 0

    while step < args.total_steps:
        # Algorithm 2 line 7: in noisy mode, pick the noise once per rollout
        # so the policy is consistent (paper Eq. 25-27). In vanilla mode this
        # is a no-op.
        if args.noisy:
            model.reset_noise()

        log_probs, values, rewards, entropies, dones = [], [], [], [], []

        for _ in range(args.rollout_size):
            obs_t = to_tensor(obs, device)
            logits, value = model(obs_t)
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)

            action = int(torch.multinomial(probs, num_samples=1).item())
            log_prob = log_probs_all[0, action]
            entropy = -(probs * log_probs_all).sum()

            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += r

            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(float(r))
            dones.append(float(terminated))
            entropies.append(entropy)

            obs = next_obs
            step += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()
                break

        # Bootstrap (Eq. 25): V(terminal)=0; otherwise use current V estimate
        # with the same noise sample.
        if dones and dones[-1] == 1.0:
            R = torch.zeros(1, device=device)
        else:
            with torch.no_grad():
                _, R = model(to_tensor(obs, device))
                R = R.detach()

        policy_loss = torch.zeros(1, device=device)
        value_loss = torch.zeros(1, device=device)
        entropy_sum = torch.zeros(1, device=device)

        for i in reversed(range(len(rewards))):
            R = rewards[i] + args.gamma * R * (1.0 - dones[i])
            advantage = R - values[i]
            # Eq. 26 policy loss (sign flipped because we minimise).
            policy_loss = policy_loss - log_probs[i] * advantage.detach()
            # Eq. 27 value MSE.
            value_loss = value_loss + advantage.pow(2)
            entropy_sum = entropy_sum + entropies[i]

        loss = policy_loss + args.value_coef * value_loss
        # Vanilla A3C: entropy bonus (paper Eq. 6 / Mnih 2016).
        # Paper sec 3.1: this is REMOVED in NoisyNet-A3C.
        if not args.noisy:
            loss = loss - args.entropy_coef * entropy_sum

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optim.step()

        if step // args.log_every > (step - len(rewards)) // args.log_every:
            sigmas = list(iter_sigma(model))
            sp = sigmas[0] if sigmas else 0.0
            sv = sigmas[1] if len(sigmas) > 1 else 0.0
            recent = episode_rewards[-20:] if episode_rewards else [0.0]
            log["step"].append(step)
            log["mean_reward"].append(float(np.mean(recent)))
            log["sigma_policy"].append(sp)
            log["sigma_value"].append(sv)
            extra = f"sigma pi {sp:.4f} V {sv:.4f}" if args.noisy else "epsilon n/a (entropy bonus on)"
            print(f"step {step:7d} | episodes {len(episode_rewards):4d} | "
                  f"reward(last20) {np.mean(recent):6.1f} | {extra} | "
                  f"elapsed {time.time() - t0:6.1f}s")

    env.close()
    np.savez(os.path.join(args.out_dir, "log.npz"),
             episode_rewards=np.array(episode_rewards), **log)
    plot(args.out_dir, episode_rewards, log, args.noisy)


def plot(out_dir, episode_rewards, log, noisy):
    title = "NoisyNet-A3C" if noisy else "vanilla A3C"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_rewards, alpha=0.3, label="episode reward")
    if len(episode_rewards) >= 20:
        smooth = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
        ax.plot(np.arange(len(smooth)) + 19, smooth, label="20-ep moving avg")
    ax.set_xlabel("episode"); ax.set_ylabel("return"); ax.set_title(title)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rewards.png"), dpi=120); plt.close(fig)

    if noisy:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log["step"], log["sigma_policy"], label="policy head")
        ax.plot(log["step"], log["sigma_value"], label="value head")
        ax.set_xlabel("env step"); ax.set_ylabel(r"mean $|\sigma_w|$")
        ax.set_title(f"{title} noise magnitude (paper Fig. 3 style)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=120); plt.close(fig)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--rollout-size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01,
                   help="beta in paper Eq. 6 — only used when --no-noisy")
    p.add_argument("--noisy", dest="noisy", action="store_true", default=True)
    p.add_argument("--no-noisy", dest="noisy", action="store_false")
    p.add_argument("--log-every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/a3c_noisynet")
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
