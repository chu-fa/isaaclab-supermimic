"""Executed-transition tracking errors; evaluation only, never policy inputs."""

import torch


class TrackingMetricAccumulator:
    def __init__(self, world_count, device):
        self.count = torch.zeros(world_count, dtype=torch.long, device=device)
        self.totals = {}
        self.maxima = {}
        self.final = {}

    def update(self, values, active):
        self.count += active.long()
        for name, value in values.items():
            if name not in self.totals:
                self.totals[name] = torch.zeros_like(value)
                self.maxima[name] = torch.zeros_like(value)
                self.final[name] = torch.zeros_like(value)
            self.totals[name] += torch.where(active, value, 0.0)
            self.maxima[name] = torch.maximum(
                self.maxima[name], torch.where(active, value, 0.0)
            )
            self.final[name] = torch.where(active, value, self.final[name])

    def record(self, world_index):
        count = int(self.count[world_index])
        if not count:
            return {}
        return {
            name: {
                'mean': float(total[world_index] / count),
                'max': float(self.maxima[name][world_index]),
                'final': float(self.final[name][world_index]),
            }
            for name, total in self.totals.items()
        }
