import numpy as np
import torch
from gymnasium import spaces

from netris_rl.features import TetrisFeaturesExtractor


def make_observation_space(include_next_piece=True):
    observation_spaces = {
        "board": spaces.Box(low=0, high=1, shape=(1, 18, 10), dtype=np.float32),
        "current_piece": spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32),
        "row_completion": spaces.Box(low=0, high=10, shape=(18,), dtype=np.float32),
        "holes": spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
        "bumpiness": spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
        "aggregate_height": spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
    }
    if include_next_piece:
        observation_spaces["next_piece"] = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32
        )
    return spaces.Dict(observation_spaces)


def to_tensors(observation):
    return {
        key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
        for key, value in observation.items()
    }


def test_output_shape_with_next_piece():
    space = make_observation_space(include_next_piece=True)
    extractor = TetrisFeaturesExtractor(space, features_dim=128)
    output = extractor(to_tensors(space.sample()))
    assert output.shape == (1, 128)


def test_output_shape_without_next_piece():
    space = make_observation_space(include_next_piece=False)
    extractor = TetrisFeaturesExtractor(space, features_dim=64)
    output = extractor(to_tensors(space.sample()))
    assert output.shape == (1, 64)


def test_board_changes_output():
    space = make_observation_space()
    extractor = TetrisFeaturesExtractor(space, features_dim=64)

    observation = to_tensors(space.sample())
    observation["board"] = torch.zeros_like(observation["board"])
    empty = extractor(observation).detach()

    observation["board"][0, 0, 17, :] = 1.0
    full_row = extractor(observation).detach()

    assert not torch.allclose(empty, full_row)


def test_output_is_finite():
    space = make_observation_space()
    extractor = TetrisFeaturesExtractor(space, features_dim=64)
    output = extractor(to_tensors(space.sample()))
    assert torch.isfinite(output).all()
