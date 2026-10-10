"""Demonstration collection CLI (`netris-collect`).

Collects teacher demonstrations (optionally with 1-piece lookahead on the
on-screen NEXT piece) and, with `--policy-model`, runs a DAgger round: the
policy acts while the teacher labels the visited states.
"""

import argparse
import json
import os
from multiprocessing import get_context

import numpy as np
from netris import NetrisEnvConfig, TetrisEnv

from netris_rl.teacher import best_action, best_action_lookahead

OBS_KEYS = [
    "board",
    "current_piece",
    "next_piece",
    "row_completion",
    "holes",
    "bumpiness",
    "aggregate_height",
]


def collect_chunk(task):
    (rank, steps, out_path, policy_model, lookahead, lookahead_top_k,
     include_next_piece, rom_path) = task
    env = TetrisEnv(
        rom_path,
        log_level="error",
        config=NetrisEnvConfig(include_next_piece=include_next_piece),
    )
    obs_keys = [key for key in OBS_KEYS if include_next_piece or key != "next_piece"]
    env.reset()

    policy = None
    if policy_model:
        from stable_baselines3 import PPO

        policy = PPO.load(policy_model, device="cpu")

    observations = {key: [] for key in obs_keys}
    actions = []
    collected = 0
    episodes = 0
    lines_total = 0
    agreements = 0

    while collected < steps:
        shape = env.get_current_piece_name()
        if shape is None:
            env.reset()
            continue

        observation = env.get_observation()
        board = env.get_locked_board()
        if lookahead:
            teacher_action, _score, _meta = best_action_lookahead(
                board, shape, env.get_next_piece_name(), top_k=lookahead_top_k
            )
        else:
            teacher_action, _score, _meta = best_action(board, shape)

        if policy is None:
            action = teacher_action
        else:
            action, _ = policy.predict(observation, deterministic=True)
            action = int(action)
            agreements += int(action == teacher_action)

        for key in obs_keys:
            observations[key].append(observation[key])
        actions.append(teacher_action)

        _, _, done, truncated, info = env.step(action)
        collected += 1

        if done or truncated:
            lines_total += int(info.get("lines_cleared", 0))
            episodes += 1
            env.reset()

    np.savez_compressed(
        out_path,
        **{
            key: np.stack(observations[key]).astype(np.float32)
            for key in obs_keys
        },
        actions=np.asarray(actions, dtype=np.int64),
    )
    env.close()
    return {
        "rank": rank,
        "steps": collected,
        "episodes": episodes,
        "lines": lines_total,
        "agreement": round(agreements / max(1, collected), 4) if policy else None,
        "output": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect teacher demonstrations")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", default="logs/demos")
    parser.add_argument("--prefix", default="demos")
    parser.add_argument("--rom", default=None,
                        help="Path to the GameBoy Tetris ROM (or set NETRIS_ROM)")
    parser.add_argument("--policy-model", default="",
                        help="If set, execute this policy and label its states "
                             "with the teacher (DAgger)")
    parser.add_argument("--lookahead", action="store_true",
                        help="Teacher uses the on-screen NEXT piece for lookahead")
    parser.add_argument("--lookahead-top-k", type=int, default=4)
    parser.add_argument("--no-next-piece", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    workers = max(1, args.workers)
    chunk = args.steps // workers

    tasks = []
    for rank in range(workers):
        steps = chunk if rank < workers - 1 else args.steps - chunk * (workers - 1)
        path = os.path.join(args.output_dir, f"{args.prefix}_{rank}.npz")
        tasks.append((
            rank, steps, path, args.policy_model or None,
            args.lookahead, args.lookahead_top_k,
            not args.no_next_piece, args.rom,
        ))

    context = get_context("spawn")
    with context.Pool(workers) as pool:
        results = pool.map(collect_chunk, tasks)

    total_lines = sum(r["lines"] for r in results)
    total_steps = sum(r["steps"] for r in results)
    total_episodes = sum(r["episodes"] for r in results)
    manifest = {
        "steps": total_steps,
        "episodes": total_episodes,
        "lines": total_lines,
        "lines_per_episode": round(total_lines / max(1, total_episodes), 2),
        "parts": results,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
