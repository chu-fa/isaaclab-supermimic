"""Return-based one-second-bin reset sampling from paper Appendix D.1."""

import torch


class ReturnSampler:
    def __init__(self, motions, discount=0.99, *, return_rate=0.2, history_rate=0.001):
        self.motions = motions
        self.discount = discount
        self.return_rate = return_rate
        self.history_rate = history_rate
        count = len(motions.bins)
        self.mean_returns = torch.zeros(count, device=motions.device)
        self.past_weights = torch.full((count,), 1 / count, device=motions.device)
        self.max_discounted_return = 5 * motions.control_period / (1 - discount)

    def probabilities(self):
        difficult = torch.softmax(-25 * self.mean_returns / self.max_discounted_return, dim=0)
        historical = self.past_weights / self.past_weights.sum()
        return 0.5 / len(self.mean_returns) + 0.4 * difficult + 0.1 * historical

    def sample(self, count):
        bins = torch.multinomial(self.probabilities(), count, replacement=True)
        starts = self.motions.bins[bins]
        available = (self.motions.frame_ends[starts] - starts).clamp(
            max=round(1 / self.motions.control_period)
        )
        offset = (torch.rand(count, device=starts.device) * available).long()
        return starts + offset, bins

    @torch.no_grad()
    def update(self, bins, episode_returns):
        totals = torch.zeros_like(self.mean_returns).scatter_add_(0, bins, episode_returns)
        counts = torch.zeros_like(self.mean_returns).scatter_add_(0, bins, torch.ones_like(episode_returns))
        visited = counts > 0
        self.mean_returns[visited] = torch.lerp(
            self.mean_returns[visited], totals[visited] / counts[visited], self.return_rate
        )
        difficult = torch.softmax(-25 * self.mean_returns / self.max_discounted_return, dim=0)
        self.past_weights.lerp_(difficult, self.history_rate)
