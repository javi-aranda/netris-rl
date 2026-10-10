"""Analytic Tetris placement teacher.

Simulates every placement on the locked board and scores the result with the
Dellacherie heuristic. Used to generate demonstrations for behavior cloning.

Piece rotations live in `netris.gameboy.pieces` and were calibrated against the
emulator, so the simulation matches what the GameBoy actually does.
"""


from netris.gameboy.board import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    clear_lines,
    column_transitions,
    count_holes,
    first_filled_row,
    row_transitions,
    wells,
)
from netris.gameboy.pieces import PIECE_SHAPES

DELLACHERIE_WEIGHTS = {
    "landing_height": -4.500158825082766,
    "lines_cleared": 3.4181268101392694,
    "row_transitions": -3.2178882868487753,
    "column_transitions": -9.348695305445199,
    "holes": -7.899265427351652,
    "wells": -3.3855972247263626,
}


def simulate_placement(board, shape, rotation, target_col):
    """Drop `shape`/`rotation` at `target_col` (bounding-box left edge).

    Returns (resulting_board, lines_cleared, landing_height, actual_col) or None
    if the piece cannot be placed.
    """
    offsets = PIECE_SHAPES[shape][rotation]
    width = max(dx for _, dx in offsets) + 1
    col = max(0, min(BOARD_WIDTH - width, target_col))

    landing = None
    for dy, dx in offsets:
        top = first_filled_row(board, col + dx)
        allowed = top - 1 - dy
        landing = allowed if landing is None else min(landing, allowed)

    if landing is None or landing < -min(dy for dy, _ in offsets):
        return None
    if landing + max(dy for dy, _ in offsets) >= BOARD_HEIGHT:
        return None

    placed = board.copy()
    for dy, dx in offsets:
        placed[landing + dy, col + dx] = 1

    bottom_height = BOARD_HEIGHT - (landing + max(dy for dy, _ in offsets))
    top_height = BOARD_HEIGHT - (landing + min(dy for dy, _ in offsets))
    landing_height = (bottom_height + top_height) / 2.0

    cleared_board, lines = clear_lines(placed)
    return cleared_board, lines, landing_height, col


def evaluate_board(board, lines_cleared, landing_height):
    weights = DELLACHERIE_WEIGHTS
    return (
        weights["landing_height"] * landing_height
        + weights["lines_cleared"] * lines_cleared
        + weights["row_transitions"] * row_transitions(board)
        + weights["column_transitions"] * column_transitions(board)
        + weights["holes"] * count_holes(board)
        + weights["wells"] * wells(board)
    )


def best_action(board, shape):
    """Return (action, score, info) for the best placement of `shape`."""
    best = None
    for rotation in range(4):
        offsets = PIECE_SHAPES[shape][rotation]
        width = max(dx for _, dx in offsets) + 1
        for target_col in range(BOARD_WIDTH - width + 1):
            result = simulate_placement(board, shape, rotation, target_col)
            if result is None:
                continue
            new_board, lines, landing_height, actual_col = result
            score = evaluate_board(new_board, lines, landing_height)
            if best is None or score > best[1]:
                best = (
                    rotation * 10 + actual_col,
                    score,
                    {
                        "rotation": rotation,
                        "column": actual_col,
                        "lines": lines,
                        "score": score,
                    },
                )
    if best is None:
        return 0, float("-inf"), {
            "rotation": 0, "column": 0, "lines": 0, "score": float("-inf"),
        }
    return best


def best_future_score(board, shape):
    """Best Dellacherie score achievable by placing `shape` on `board`."""
    if shape is None:
        return 0.0
    best_score = None
    for rotation in range(4):
        offsets = PIECE_SHAPES[shape][rotation]
        width = max(dx for _, dx in offsets) + 1
        for target_col in range(BOARD_WIDTH - width + 1):
            result = simulate_placement(board, shape, rotation, target_col)
            if result is None:
                continue
            new_board, lines, landing_height, _ = result
            score = evaluate_board(new_board, lines, landing_height)
            if best_score is None or score > best_score:
                best_score = score
    return best_score if best_score is not None else float("-inf")


def best_action_lookahead(board, shape, next_shape, discount=0.9, top_k=10):
    """Best placement of `shape` considering the best use of `next_shape`.

    Only uses the next piece shown on the game's NEXT panel (no ROM/RNG
    peeking). `top_k` limits the search to the most promising current
    placements to keep it fast.
    """
    candidates = []
    for rotation in range(4):
        offsets = PIECE_SHAPES[shape][rotation]
        width = max(dx for _, dx in offsets) + 1
        for target_col in range(BOARD_WIDTH - width + 1):
            result = simulate_placement(board, shape, rotation, target_col)
            if result is None:
                continue
            new_board, lines, landing_height, actual_col = result
            score = evaluate_board(new_board, lines, landing_height)
            candidates.append((score, rotation, actual_col, new_board, lines))

    if not candidates:
        return 0, float("-inf"), {
            "rotation": 0, "column": 0, "lines": 0,
            "score": float("-inf"), "future": 0.0,
        }

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    best = None
    for score, rotation, actual_col, new_board, lines in candidates[:top_k]:
        future = best_future_score(new_board, next_shape)
        combined = score + discount * future
        if best is None or combined > best[1]:
            best = (
                rotation * 10 + actual_col,
                combined,
                {
                    "rotation": rotation,
                    "column": actual_col,
                    "lines": lines,
                    "score": score,
                    "future": future,
                },
            )
    return best
