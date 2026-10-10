import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from netris_rl.probe_callback import ProbeCallback


class DummyDictEnv(Env):
    """Minimal env with the same observation/action spaces as TetrisEnv."""

    def __init__(self, include_next_piece=True):
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
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Discrete(40)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def make_probe_arrays(count=32):
    rng = np.random.default_rng(0)
    return {
        "board": rng.integers(0, 2, size=(count, 1, 18, 10)).astype(np.float32),
        "current_piece": np.eye(7, dtype=np.float32)[rng.integers(0, 7, size=count)],
        "next_piece": np.eye(7, dtype=np.float32)[rng.integers(0, 7, size=count)],
        "row_completion": rng.integers(0, 10, size=(count, 18)).astype(np.float32),
        "holes": rng.integers(0, 180, size=(count, 1)).astype(np.float32),
        "bumpiness": rng.integers(0, 180, size=(count, 1)).astype(np.float32),
        "aggregate_height": rng.integers(0, 180, size=(count, 1)).astype(np.float32),
        "actions": rng.integers(0, 40, size=count).astype(np.int64),
    }


def test_probe_logs_metrics(tmp_path):
    np.savez_compressed(tmp_path / "demos_0.npz", **make_probe_arrays())

    model = PPO("MultiInputPolicy", DummyDictEnv(), n_steps=16, batch_size=8, verbose=0)
    logger = configure(str(tmp_path / "logs"), [])
    model._logger = logger

    callback = ProbeCallback(
        probe_path=str(tmp_path / "probe_set.npz"),
        demos_glob=str(tmp_path / "demos_*.npz"),
        num_probes=16,
        probe_interval=1,
    )
    callback.model = model
    callback._on_training_start()
    assert callback.probe is not None

    callback._probe_metrics()
    for key in (
        "probe/teacher_agreement",
        "probe/mean_max_prob",
        "probe/action_entropy",
        "probe/mean_value",
    ):
        assert key in logger.name_to_value


def test_probe_filters_missing_keys(tmp_path):
    np.savez_compressed(tmp_path / "demos_0.npz", **make_probe_arrays())

    model = PPO(
        "MultiInputPolicy",
        DummyDictEnv(include_next_piece=False),
        n_steps=16, batch_size=8, verbose=0,
    )
    logger = configure(str(tmp_path / "logs"), [])
    model._logger = logger

    callback = ProbeCallback(
        probe_path=str(tmp_path / "probe_set.npz"),
        demos_glob=str(tmp_path / "demos_*.npz"),
        num_probes=16,
        probe_interval=1,
    )
    callback.model = model
    callback._on_training_start()
    assert callback.probe is not None
    assert "next_piece" not in callback.keys

    callback._probe_metrics()
    assert "probe/teacher_agreement" in logger.name_to_value


def test_probe_disabled_without_demos(tmp_path):
    callback = ProbeCallback(
        probe_path=str(tmp_path / "probe_set.npz"),
        demos_glob=str(tmp_path / "missing_*.npz"),
    )
    assert callback._load_or_build_probe() is None
