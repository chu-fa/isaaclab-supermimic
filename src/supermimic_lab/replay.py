"""Time-major replay with explicit episode boundaries and history sampling."""

from __future__ import annotations

import torch

from .contracts import Transition


class SequenceReplay:
    def __init__(self, capacity, environments, dimensions, device):
        if capacity <= 0 or environments <= 0:
            raise ValueError("Replay capacity and environment count must be positive")
        self.capacity = capacity
        self.environments = environments
        self.device = torch.device(device)
        self.write_index = 0
        self.length = 0
        widths = {
            "state": dimensions.state, "next_state": dimensions.state,
            "action": dimensions.action, "previous_action": dimensions.action,
            "context": dimensions.context,
        }
        self.storage = {
            name: torch.empty(capacity, environments, width, device=device)
            for name, width in widths.items()
        }
        for name in ("reference_index", "episode_step"):
            self.storage[name] = torch.empty(
                capacity, environments, dtype=torch.long, device=device
            )
        self.storage["reward"] = torch.empty(capacity, environments, device=device)
        for name in ("terminated", "episode_end"):
            self.storage[name] = torch.empty(
                capacity, environments, dtype=torch.bool, device=device
            )

    @torch.no_grad()
    def append(self, transition: Transition):
        for name, tensor in transition.tensors().items():
            if tensor.shape != self.storage[name][self.write_index].shape:
                raise ValueError(f"Transition shape mismatch for {name}: {tensor.shape}")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"Non-finite simulator transition: {name}")
        for name, tensor in transition.tensors().items():
            self.storage[name][self.write_index].copy_(tensor)
        self.write_index = (self.write_index + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def _storage_indices(self, chronological_indices):
        return (self.write_index - self.length + chronological_indices) % self.capacity

    def mark_collection_reset(self):
        """Prevent dynamics windows crossing a reset requested outside step()."""
        if self.length:
            self.storage["episode_end"][(self.write_index - 1) % self.capacity] = True

    def windows(self, batch_size, window_length):
        if self.length < window_length:
            raise ValueError("Not enough replay transitions for a model window")
        first_steps = torch.randint(
            self.length - window_length + 1, (batch_size,), device=self.device
        )
        environments = torch.randint(self.environments, (batch_size, 1), device=self.device)
        timeline = first_steps[:, None] + torch.arange(window_length, device=self.device)
        indices = self._storage_indices(timeline)
        windows = {
            name: tensor[indices, environments] for name, tensor in self.storage.items()
        }
        valid = torch.ones(batch_size, window_length, device=self.device)
        if window_length > 1:
            valid[:, 1:] = (~windows["episode_end"][:, :-1]).float().cumprod(dim=1)
        windows["valid"] = valid
        return windows

    def starts(self, batch_size, history_length):
        if self.length == 0:
            raise ValueError("Cannot sample empty replay")
        timeline = torch.randint(self.length, (batch_size,), device=self.device)
        environments = torch.randint(self.environments, (batch_size,), device=self.device)
        indices = self._storage_indices(timeline)
        start = {
            name: tensor[indices, environments] for name, tensor in self.storage.items()
        }
        backwards = torch.arange(history_length - 1, -1, -1, device=self.device)
        available = torch.minimum(timeline, start["episode_step"])
        history_timeline = timeline[:, None] - torch.minimum(backwards[None], available[:, None])
        history_indices = self._storage_indices(history_timeline)
        history = {
            name: self.storage[name][history_indices, environments[:, None]]
            for name in ("state", "previous_action", "context")
        }
        return start, history
