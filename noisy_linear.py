"""
NoisyLinear layer from "Noisy Networks for Exploration"
(Fortunato et al., 2018 - arXiv:1706.10295).

Replaces a standard nn.Linear (Eq. 8) with a noisy variant (Eq. 9) whose
weights and biases are perturbed by learnt parametric noise.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Noisy linear layer. Paper Eq. 9:

        y = (mu_w + sigma_w * eps_w) x + (mu_b + sigma_b * eps_b)

    mu and sigma are learnt; eps is fresh noise sampled at every forward pass
    (until reset_noise is called again).

    Two noise modes:
      - "factorised":  eps_w[i,j] = f(eps_in[j]) * f(eps_out[i])  (Eq. 10/11)
                       cheap: only p+q random draws per layer.
                       Used in DQN / Dueling.
      - "independent": eps_w[i,j] ~ N(0,1) per entry.
                       Used in A3C (paper section 3, option (a)).
    """

    def __init__(self, in_features, out_features,
                 sigma_zero=0.5, factorised=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised
        self.sigma_zero = sigma_zero  # paper sec 3.2: sigma_0 = 0.5

        # Learnable params: means and stds for weights and biases.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnt, refreshed by reset_noise).
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    # ---------- initialisation (paper sec 3.2) ----------
    def reset_parameters(self):
        p = self.in_features
        if self.factorised:
            # mu ~ U[-1/sqrt(p), +1/sqrt(p)],  sigma = sigma_0 / sqrt(p)
            bound = 1.0 / math.sqrt(p)
            sigma_init = self.sigma_zero / math.sqrt(p)
        else:
            # mu ~ U[-sqrt(3/p), +sqrt(3/p)],  sigma = 0.017
            bound = math.sqrt(3.0 / p)
            sigma_init = 0.017

        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    # ---------- noise sampling ----------
    @staticmethod
    def _f(x):
        # f(x) = sgn(x) * sqrt(|x|), used to shape factorised noise (paper sec 3).
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Draw a fresh noise sample. Call between optimisation/action steps."""
        if self.factorised:
            eps_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
            eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
            # Eq. 10: eps_w[i,j] = f(eps_in[j]) * f(eps_out[i])  -> outer product
            self.weight_eps.copy_(eps_out.outer(eps_in))
            # Eq. 11: eps_b[j] = f(eps_j)
            self.bias_eps.copy_(eps_out)
        else:
            self.weight_eps.normal_()
            self.bias_eps.normal_()

    # ---------- forward ----------
    def forward(self, x):
        if self.training:
            # Noisy weights: paper Eq. 9.
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            # At eval time, use the mean (no exploration).
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)
