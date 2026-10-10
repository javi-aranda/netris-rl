"""Evaluation CLI (`netris-eval`)."""

import argparse
import os

import numpy as np
from netris import NetrisEnvConfig, TetrisEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


def load_model_and_env(model_path, rom_path, vecnorm_path):
    model = PPO.load(model_path, device="cpu")
    include_next_piece = "next_piece" in model.observation_space.spaces

    def _make_env():
        return TetrisEnv(
            rom_path,
            log_level="error",
            config=NetrisEnvConfig(include_next_piece=include_next_piece),
        )

    venv = VecMonitor(DummyVecEnv([_make_env]))
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, venv)
        env.training = False
        env.norm_reward = False
    else:
        env = venv
    return model, env


def main():
    parser = argparse.ArgumentParser(description="Netris evaluation")
    parser.add_argument("--model", default="logs/best_model/best_model.zip")
    parser.add_argument("--rom", default=None,
                        help="Path to the GameBoy Tetris ROM (or set NETRIS_ROM)")
    parser.add_argument("--vecnormalize", default="",
                        help="Optional VecNormalize stats file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of deterministic argmax")
    args = parser.parse_args()

    model, env = load_model_and_env(args.model, args.rom, args.vecnormalize)

    all_lines, all_pieces, all_rewards = [], [], []
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        info = {}
        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rewards, dones, infos = env.step(action)
            episode_reward += float(rewards[0])
            done = bool(dones[0])
            steps += 1
            info = infos[0]
        lines = int(info.get("lines_cleared", 0))
        all_lines.append(lines)
        all_pieces.append(steps)
        all_rewards.append(episode_reward)
        print(
            f"episode {episode + 1}: lines={lines} pieces={steps} "
            f"reward={episode_reward:.2f}"
        )

    print(
        f"\nover {args.episodes} episodes: mean lines={np.mean(all_lines):.2f} "
        f"median={np.median(all_lines):.1f} max={max(all_lines)} | "
        f"mean pieces={np.mean(all_pieces):.1f} | "
        f"mean reward={np.mean(all_rewards):.2f}"
    )
    env.close()


if __name__ == "__main__":
    main()
