import numpy as np
import pytest
from netris import NetrisEnvConfig, TetrisEnv

from netris_rl.teacher import (
    BOARD_WIDTH,
    best_action,
    best_action_lookahead,
    simulate_placement,
)


@pytest.fixture(scope="module")
def env(rom_path):
    environment = TetrisEnv(
        rom_path, config=NetrisEnvConfig(include_next_piece=True)
    )
    yield environment
    environment.close()


@pytest.mark.rom
@pytest.mark.parametrize("shape", ["I", "O", "T", "S", "Z", "L", "J"])
@pytest.mark.parametrize("rotation", [0, 1])
def test_simulation_matches_emulator(env, shape, rotation):
    for target_col in [0, 2, 4]:
        env.force_next_tetromino(shape)
        env.reset()
        before = env.get_locked_board().copy()
        _, _, done, truncated, info = env.step(rotation * 10 + target_col)
        if done or truncated:
            continue

        simulated = simulate_placement(before, shape, rotation, target_col)
        assert simulated is not None
        simulated_board, lines, _, _ = simulated
        assert np.array_equal(simulated_board, env.get_locked_board())
        assert lines == info["lines_cleared"]


@pytest.mark.rom
def test_teacher_clears_lines(env):
    env.reset()
    total_lines = 0
    for _ in range(120):
        shape = env.get_current_piece_name()
        assert shape is not None
        action, _, _ = best_action(env.get_locked_board(), shape)
        _, _, done, truncated, info = env.step(action)
        total_lines = max(total_lines, info["lines_cleared"])
        if done or truncated:
            break
    assert total_lines >= 3


@pytest.mark.rom
def test_lookahead_clears_lines(env):
    env.reset()
    total_lines = 0
    for _ in range(120):
        shape = env.get_current_piece_name()
        next_shape = env.get_next_piece_name()
        action, _, _ = best_action_lookahead(
            env.get_locked_board(), shape, next_shape
        )
        _, _, done, truncated, info = env.step(action)
        total_lines = max(total_lines, info["lines_cleared"])
        if done or truncated:
            break
    assert total_lines >= 3


@pytest.mark.rom
def test_best_action_is_reachable(env):
    env.reset()
    for _ in range(20):
        shape = env.get_current_piece_name()
        action, score, meta = best_action(env.get_locked_board(), shape)
        assert 0 <= action < 40
        assert np.isfinite(score)
        assert 0 <= meta["column"] <= BOARD_WIDTH - 1
        _, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break
