"""
Plain uniform replay buffer.

Paper Eq. 14 expects (x, a, r, y) tuples sampled from a distribution D over a
replay; uniform sampling is the standard choice and matches DQN (Mnih 2015).
"""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32, device=device),
            torch.as_tensor(a, dtype=torch.long, device=device),
            torch.as_tensor(r, dtype=torch.float32, device=device),
            torch.as_tensor(np.array(s_next), dtype=torch.float32, device=device),
            torch.as_tensor(done, dtype=torch.float32, device=device),
        )
