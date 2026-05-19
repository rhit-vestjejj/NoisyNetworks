"""
Prioritized Experience Replay buffer.
Schaul et al., 2016 — "Prioritized Experience Replay" (arXiv:1511.05952).

Proportional prioritization (sec 3.3):
  p_i = (|delta_i| + eps)^alpha
  P(i) = p_i / sum_j p_j
  w_i  = (N * P(i))^{-beta}   normalized by max weight

Uses a binary sum-tree for O(log N) sampling and priority updates.
"""

import numpy as np
import torch


class SumTree:
    """
    Binary sum-tree. Leaves store priorities; internal nodes store their
    subtree sum. Root (index 1) holds the total. Leaf i is at index
    capacity + i (1-indexed). Supports circular writes like a ring buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, tree_idx, delta):
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += delta

    @property
    def total(self):
        return self.tree[1]

    def add(self, priority):
        """Write priority at current write position; return the tree index."""
        tree_idx = self.write_idx + self.capacity
        old = self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, priority - old)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return tree_idx

    def update(self, tree_idx, priority):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, cumsum):
        """Walk tree to find the leaf whose cumulative priority >= cumsum."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if cumsum <= self.tree[left]:
                idx = left
            else:
                cumsum -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, data_idx, self.tree[idx]


class PrioritizedFrameReplay:
    """
    Memory-efficient uint8 prioritized replay buffer. Same pre-allocated
    numpy arrays as FrameReplay; replaces uniform sampling with priority-
    weighted stratified sampling and importance-sampling weight correction.

    Args:
        capacity:  max transitions stored
        obs_shape: (C, H, W) tuple, e.g. (4, 84, 84)
        alpha:     priority exponent (0 = uniform, 1 = full priority). Paper: 0.6
        eps:       small constant added to |TD error| to avoid zero priority
    """

    def __init__(self, capacity, obs_shape, alpha=0.6, eps=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0   # new transitions get this until first update

        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.tree.n_entries

    def push(self, s, a, r, s_next, terminal):
        data_idx = self.tree.write_idx
        self.obs[data_idx]      = s
        self.next_obs[data_idx] = s_next
        self.actions[data_idx]  = a
        self.rewards[data_idx]  = r
        self.terminals[data_idx] = terminal
        # New transitions use max priority so they are sampled at least once.
        self.tree.add(self.max_priority)

    def sample(self, batch_size, device, beta):
        """
        Stratified priority sampling (paper sec 3.3).

        Returns:
            tree_indices  (np.ndarray, shape [B])  — for update_priorities
            weights       (torch.Tensor, shape [B]) — IS correction, on device
            s, a, r, s_next, term                  — batch tensors on device
        """
        total = self.tree.total
        segment = total / batch_size

        tree_indices = np.empty(batch_size, dtype=np.int32)
        data_indices = np.empty(batch_size, dtype=np.int32)
        priorities   = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, data_idx, priority = self.tree.get(value)
            tree_indices[i] = tree_idx
            data_indices[i] = data_idx
            priorities[i]   = priority

        # Importance-sampling weights: w_i = (N * P(i))^{-beta} / max_w
        probs   = priorities / total
        weights = (len(self) * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            tree_indices,
            torch.as_tensor(weights, device=device),
            torch.as_tensor(self.obs[data_indices],      device=device),
            torch.as_tensor(self.actions[data_indices],  device=device),
            torch.as_tensor(self.rewards[data_indices],  device=device),
            torch.as_tensor(self.next_obs[data_indices], device=device),
            torch.as_tensor(self.terminals[data_indices], device=device),
        )

    def update_priorities(self, tree_indices, td_errors):
        """Update leaf priorities from new TD errors (numpy array)."""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(tree_idx), float(priority))
            if priority > self.max_priority:
                self.max_priority = float(priority)
