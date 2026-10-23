"""Paper-sized networks and finite running feature statistics."""

from __future__ import annotations

import torch
from torch import nn


class FeatureStatistics(nn.Module):
    """Parallel population-moment updates; no clipping or NaN replacement."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))

    @torch.no_grad()
    def update(self, samples):
        flattened = samples.detach().reshape(-1, self.mean.numel())
        if flattened.shape[0] == 0:
            return
        if not bool(torch.isfinite(flattened).all()):
            raise FloatingPointError("Non-finite input to running statistics")
        batch_variance, batch_mean = torch.var_mean(flattened, dim=0, correction=0)
        batch_count = flattened.shape[0]
        total_count = self.sample_count + batch_count
        displacement = batch_mean - self.mean
        previous_sum = self.variance * self.sample_count
        batch_sum = batch_variance * batch_count
        cross_sum = displacement.square() * self.sample_count * batch_count / total_count
        self.mean.add_(displacement * (batch_count / total_count).to(self.mean.dtype))
        self.variance.copy_(((previous_sum + batch_sum + cross_sum) / total_count).to(self.mean.dtype))
        self.sample_count.copy_(total_count)

    def scale(self):
        return self.variance.clamp_min(1e-6).sqrt()

    def forward(self, samples):
        return (samples - self.mean) / self.scale()


def dense_network(input_width, hidden_widths, output_width, *, zero_output=False):
    layers = []
    previous_width = input_width
    for hidden_width in hidden_widths:
        layers.extend((nn.Linear(previous_width, hidden_width), nn.SiLU()))
        previous_width = hidden_width
    output_layer = nn.Linear(previous_width, output_width)
    if zero_output:
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
    else:
        nn.init.normal_(output_layer.weight, std=0.01)
        nn.init.zeros_(output_layer.bias)
    layers.append(output_layer)
    return nn.Sequential(*layers)


class DynamicsNetwork(nn.Module):
    def __init__(self, dimensions, hidden_widths):
        super().__init__()
        self.input_statistics = FeatureStatistics(dimensions.model_features)
        self.target_statistics = FeatureStatistics(dimensions.model_outputs)
        self.network = dense_network(
            dimensions.model_features, hidden_widths, dimensions.model_outputs,
            zero_output=True,
        )

    def forward(self, features):
        standardized = self.network(self.input_statistics(features))
        return standardized * self.target_statistics.scale() + self.target_statistics.mean


class MotionPolicy(nn.Module):
    def __init__(self, dimensions, configuration):
        super().__init__()
        self.observation_statistics = FeatureStatistics(dimensions.observation)
        self.goal_statistics = FeatureStatistics(dimensions.goal)
        self.encoder = dense_network(
            dimensions.goal, configuration.encoder_widths, configuration.latent_size
        )
        self.decoder = dense_network(
            dimensions.observation * configuration.history_frames + configuration.latent_size,
            configuration.decoder_widths, dimensions.action,
        )

    def latent(self, goal):
        return torch.tanh(self.encoder(self.goal_statistics(goal)))

    def forward(self, history, goal):
        observations = self.observation_statistics(history).flatten(start_dim=1)
        return self.decoder(torch.cat((self.latent(goal), observations), dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, feature_width, hidden_widths):
        super().__init__()
        self.statistics = FeatureStatistics(feature_width)
        self.network = dense_network(feature_width, hidden_widths, 1)

    def forward(self, features):
        return self.network(self.statistics(features)).squeeze(-1)
