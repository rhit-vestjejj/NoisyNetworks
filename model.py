"""
All model architectures used in the paper, in both NoisyNet and vanilla forms:

  - NoisyDQN          : MLP Q-net for low-dim envs (CartPole sanity tests).
  - NoisyDQNAtari     : DQN-Nature conv encoder + 2 FC layers.
  - NoisyDuelingMLP   : Dueling (Eq. 3) on a small MLP base.
  - NoisyDuelingAtari : Dueling on the Atari conv encoder.
  - NoisyA3C          : shared encoder + policy head + value head.

Every class takes `noisy: bool`. When True, FC layers are NoisyLinear and the
agent should explore via weight noise. When False, FC layers are plain
nn.Linear and the agent must explore via the classic method for its algo
(epsilon-greedy for DQN/Dueling, entropy bonus for A3C).

Paper sec 3.1: only the *fully connected* layers are noisy; conv layers stay
deterministic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_linear import NoisyLinear


def _linear(in_features, out_features, noisy, factorised):
    """Return a NoisyLinear or a plain nn.Linear, depending on `noisy`."""
    if noisy:
        return NoisyLinear(in_features, out_features, factorised=factorised)
    return nn.Linear(in_features, out_features)


def reset_module_noise(module):
    """Walk the module tree and call reset_noise() on every NoisyLinear.
    Safe to call on vanilla models — it's a no-op there."""
    for m in module.modules():
        if isinstance(m, NoisyLinear):
            m.reset_noise()


def iter_sigma(module):
    """Yield mean(|sigma_w|) per NoisyLinear submodule (paper Eq. 20)."""
    for m in module.modules():
        if isinstance(m, NoisyLinear):
            yield m.weight_sigma.detach().abs().mean().item()


# ----------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------
class AtariConvEncoder(nn.Module):
    """The Mnih 2015 / DQN-Nature conv stack. Input is (N, 4, 84, 84) uint8 or
    float; the layer scales by 1/255 internally so the caller can pass uint8."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.out_dim = 64 * 7 * 7  # 3136

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.net(x)


class MLPEncoder(nn.Module):
    """Tiny MLP base for low-dim envs (CartPole)."""

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# DQN heads
# ----------------------------------------------------------------------
class NoisyDQN(nn.Module):
    """MLP Q-net (CartPole). Two FC layers with ReLU between."""

    def __init__(self, obs_dim, n_actions, hidden=128,
                 noisy=True, factorised=True):
        super().__init__()
        self.fc1 = _linear(obs_dim, hidden, noisy, factorised)
        self.fc2 = _linear(hidden, n_actions, noisy, factorised)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def reset_noise(self):
        reset_module_noise(self)


class NoisyDQNAtari(nn.Module):
    """Atari Q-net: conv encoder + 2 FC. Paper sec 3.1."""

    def __init__(self, n_actions, noisy=True, factorised=True):
        super().__init__()
        self.encoder = AtariConvEncoder()
        self.fc1 = _linear(self.encoder.out_dim, 512, noisy, factorised)
        self.fc2 = _linear(512, n_actions, noisy, factorised)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(self.encoder(x))))

    def reset_noise(self):
        reset_module_noise(self)


# ----------------------------------------------------------------------
# Dueling heads (paper Eq. 3)
# ----------------------------------------------------------------------
class _DuelingHeads(nn.Module):
    """Two parallel streams V(s) and A(s,a) over a shared base, combined as
        Q(s,a) = V(s) + A(s,a) - mean_b A(s,b)         (paper Eq. 3)
    """

    def __init__(self, in_dim, hidden, n_actions, noisy, factorised):
        super().__init__()
        self.value_fc = _linear(in_dim, hidden, noisy, factorised)
        self.value_out = _linear(hidden, 1, noisy, factorised)
        self.adv_fc = _linear(in_dim, hidden, noisy, factorised)
        self.adv_out = _linear(hidden, n_actions, noisy, factorised)

    def forward(self, h):
        v = self.value_out(F.relu(self.value_fc(h)))
        a = self.adv_out(F.relu(self.adv_fc(h)))
        return v + (a - a.mean(dim=1, keepdim=True))


class NoisyDuelingMLP(nn.Module):
    """Dueling on an MLP base (CartPole sanity check)."""

    def __init__(self, obs_dim, n_actions, hidden=128,
                 noisy=True, factorised=True):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, hidden)
        self.head = _DuelingHeads(self.encoder.out_dim, hidden, n_actions,
                                  noisy, factorised)

    def forward(self, x):
        return self.head(self.encoder(x))

    def reset_noise(self):
        reset_module_noise(self)


class NoisyDuelingAtari(nn.Module):
    """Dueling on the Atari conv base (paper main result)."""

    def __init__(self, n_actions, hidden=512, noisy=True, factorised=True):
        super().__init__()
        self.encoder = AtariConvEncoder()
        self.head = _DuelingHeads(self.encoder.out_dim, hidden, n_actions,
                                  noisy, factorised)

    def forward(self, x):
        return self.head(self.encoder(x))

    def reset_noise(self):
        reset_module_noise(self)


# ----------------------------------------------------------------------
# A3C model (paper sec 3.1 + Appendix A)
# ----------------------------------------------------------------------
class NoisyA3C(nn.Module):
    """
    Shared encoder + policy head + value head.

    NoisyNet mode (paper sec 3.1 A3C paragraph): independent Gaussian noise
    (factorised=False). Noise is fixed for the whole rollout (Eq. 25-27)
    so the caller must call reset_noise() at the START of each rollout only.

    Vanilla mode: standard nn.Linear heads, and the caller must add the
    entropy bonus (paper Eq. 6) to the policy loss.
    """

    def __init__(self, encoder, n_actions, hidden=128,
                 noisy=True, factorised=False):
        super().__init__()
        self.encoder = encoder
        self.policy = _linear(encoder.out_dim, n_actions, noisy, factorised)
        self.value = _linear(encoder.out_dim, 1, noisy, factorised)

    def forward(self, x):
        h = self.encoder(x)
        return self.policy(h), self.value(h).squeeze(-1)  # logits, V(x)

    def reset_noise(self):
        reset_module_noise(self)


def make_a3c_mlp(obs_dim, n_actions, hidden=128, noisy=True):
    return NoisyA3C(MLPEncoder(obs_dim, hidden), n_actions, hidden=hidden,
                    noisy=noisy, factorised=False)


class _AtariA3CEncoder(nn.Module):
    """Conv -> Linear-ReLU bottleneck used as the shared trunk in A3C."""

    def __init__(self, hidden=256):
        super().__init__()
        self.conv = AtariConvEncoder()
        self.fc = nn.Linear(self.conv.out_dim, hidden)
        self.out_dim = hidden

    def forward(self, x):
        return F.relu(self.fc(self.conv(x)))


def make_a3c_atari(n_actions, hidden=256, noisy=True):
    return NoisyA3C(_AtariA3CEncoder(hidden), n_actions, hidden=hidden,
                    noisy=noisy, factorised=False)
