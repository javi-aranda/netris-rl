import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TetrisFeaturesExtractor(BaseFeaturesExtractor):
    """CNN for the board plus an MLP for the scalar/dict features.

    A flattened MLP struggles to reason about where placements land; a small
    CNN over the 18x10 board learns row/column patterns much more easily.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        board_channels = observation_space["board"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space["board"].sample()[None]
            ).float()
            n_flatten = self.cnn(sample).shape[1]

        self.board_head = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        # `next_piece` is optional for backwards compatibility with models
        # trained before it was added to the observation.
        self.extra_keys = [
            key for key in (
                "current_piece", "next_piece", "row_completion", "holes",
                "bumpiness", "aggregate_height",
            )
            if key in observation_space.spaces
        ]
        extra_dim = sum(observation_space[key].shape[0] for key in self.extra_keys)
        self.extra_head = nn.Sequential(
            nn.Linear(extra_dim, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(features_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        board_features = self.board_head(self.cnn(observations["board"].float()))
        extra = torch.cat(
            [observations[key].float() for key in self.extra_keys],
            dim=1,
        )
        extra_features = self.extra_head(extra)
        return self.combine(torch.cat([board_features, extra_features], dim=1))
