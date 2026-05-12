"""
All model architectures used in the paper:

  - NoisyDQN          : MLP Q-net for low-dim envs (CartPole sanity tests).
  - NoisyDQNAtari     : DQN-Nature conv encoder + 2 NoisyLinear FC.
  - NoisyDuelingMLP   : Dueling (Eq. 3) on a small MLP base.
  - NoisyDuelingAtari : Dueling on the Atari conv encoder.
  - NoisyA3C          : shared encoder + noisy policy head + noisy value head.
                        Works for both CartPole (mlp) and Atari (conv) bases.

Paper sec 3.1: only the *fully connected* layers are made noisy. Conv layers
stay deterministic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_linear import NoisyLinear


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
    """MLP NoisyNet-DQN (CartPole). Two NoisyLinear with ReLU between."""

    def __init__(self, obs_dim, n_actions, hidden=128, factorised=True):
        super().__init__()
        self.fc1 = NoisyLinear(obs_dim, hidden, factorised=factorised)
        self.fc2 = NoisyLinear(hidden, n_actions, factorised=factorised)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()


class NoisyDQNAtari(nn.Module):
    """Atari NoisyNet-DQN: conv encoder + 2 NoisyLinear. Paper sec 3.1."""

    def __init__(self, n_actions, factorised=True):
        super().__init__()
        self.encoder = AtariConvEncoder()
        self.fc1 = NoisyLinear(self.encoder.out_dim, 512, factorised=factorised)
        self.fc2 = NoisyLinear(512, n_actions, factorised=factorised)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc2(F.relu(self.fc1(h)))

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()


# ----------------------------------------------------------------------
# Dueling heads (paper Eq. 3)
# ----------------------------------------------------------------------
class _NoisyDuelingHeads(nn.Module):
    """Two parallel streams V(s) and A(s,a) sitting on top of a shared base.
    Combine via Eq. 3:
        Q(s,a) = V(s) + A(s,a) - mean_b A(s,b)
    """

    def __init__(self, in_dim, hidden, n_actions, factorised=True):
        super().__init__()
        self.value_fc = NoisyLinear(in_dim, hidden, factorised=factorised)
        self.value_out = NoisyLinear(hidden, 1, factorised=factorised)
        self.adv_fc = NoisyLinear(in_dim, hidden, factorised=factorised)
        self.adv_out = NoisyLinear(hidden, n_actions, factorised=factorised)

    def forward(self, h):
        v = self.value_out(F.relu(self.value_fc(h)))                 # (N, 1)
        a = self.adv_out(F.relu(self.adv_fc(h)))                     # (N, A)
        return v + (a - a.mean(dim=1, keepdim=True))                 # Eq. 3

    def reset_noise(self):
        for m in (self.value_fc, self.value_out, self.adv_fc, self.adv_out):
            m.reset_noise()


class NoisyDuelingMLP(nn.Module):
    """Dueling NoisyNet on an MLP base (CartPole sanity check)."""

    def __init__(self, obs_dim, n_actions, hidden=128, factorised=True):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, hidden)
        self.head = _NoisyDuelingHeads(self.encoder.out_dim, hidden, n_actions,
                                       factorised=factorised)

    def forward(self, x):
        return self.head(self.encoder(x))

    def reset_noise(self):
        self.head.reset_noise()


class NoisyDuelingAtari(nn.Module):
    """Dueling NoisyNet on the Atari conv base (paper main result)."""

    def __init__(self, n_actions, hidden=512, factorised=True):
        super().__init__()
        self.encoder = AtariConvEncoder()
        self.head = _NoisyDuelingHeads(self.encoder.out_dim, hidden, n_actions,
                                       factorised=factorised)

    def forward(self, x):
        return self.head(self.encoder(x))

    def reset_noise(self):
        self.head.reset_noise()


# ----------------------------------------------------------------------
# A3C model (paper sec 3.1 + Appendix A)
# ----------------------------------------------------------------------
class NoisyA3C(nn.Module):
    """
    Shared encoder + noisy policy head + noisy value head.

    Paper sec 3.1 (A3C paragraph): independent Gaussian noise (factorised=False).
    Paper Eq. 25-27: noise is fixed for the whole rollout, so call reset_noise()
    at the START of each rollout and NOT during it.
    """

    def __init__(self, encoder, n_actions, hidden=128, factorised=False):
        super().__init__()
        self.encoder = encoder
        self.policy = NoisyLinear(encoder.out_dim, n_actions, factorised=factorised)
        self.value = NoisyLinear(encoder.out_dim, 1, factorised=factorised)

    def forward(self, x):
        h = self.encoder(x)
        return self.policy(h), self.value(h).squeeze(-1)  # logits, V(x)

    def reset_noise(self):
        self.policy.reset_noise()
        self.value.reset_noise()


def make_a3c_mlp(obs_dim, n_actions, hidden=128):
    return NoisyA3C(MLPEncoder(obs_dim, hidden), n_actions, hidden=hidden,
                    factorised=False)


class _AtariA3CEncoder(nn.Module):
    """Conv -> Linear-ReLU bottleneck used as the shared trunk in A3C."""

    def __init__(self, hidden=256):
        super().__init__()
        self.conv = AtariConvEncoder()
        self.fc = nn.Linear(self.conv.out_dim, hidden)
        self.out_dim = hidden

    def forward(self, x):
        return F.relu(self.fc(self.conv(x)))


def make_a3c_atari(n_actions, hidden=256):
    return NoisyA3C(_AtariA3CEncoder(hidden), n_actions, hidden=hidden,
                    factorised=False)
