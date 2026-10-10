"""PPO training CLI (`netris-train`)."""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from netris import NetrisEnvConfig, TetrisEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from netris_rl.anchored_ppo import AnchoredPPO
from netris_rl.callbacks import NetrisEvalCallback, TetrisMetricsCallback
from netris_rl.features import TetrisFeaturesExtractor
from netris_rl.probe_callback import ProbeCallback

torch.set_num_threads(1)


def make_env(rank, rom_path, include_next_piece, log_dir):
    def _init():
        log_level = "info" if rank == 0 else "error"
        env = TetrisEnv(
            rom_path,
            log_level=log_level,
            config=NetrisEnvConfig(include_next_piece=include_next_piece),
        )
        return env

    return _init


def make_eval_env(rom_path, include_next_piece):
    return TetrisEnv(
        rom_path,
        log_level="error",
        config=NetrisEnvConfig(include_next_piece=include_next_piece),
    )


def build_env(args, include_next_piece):
    env_fns = [
        make_env(rank, args.rom, include_next_piece, args.log_dir)
        for rank in range(args.n_envs)
    ]
    return VecMonitor(
        SubprocVecEnv(env_fns),
        filename=os.path.join(args.log_dir, "training_logs/train"),
    )


def build_eval_env(args, include_next_piece):
    return VecMonitor(
        DummyVecEnv([lambda: make_eval_env(args.rom, include_next_piece)]),
        filename=os.path.join(args.log_dir, "eval_logs/eval"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Netris PPO training")
    parser.add_argument("--rom", default=None,
                        help="Path to the GameBoy Tetris ROM (or set NETRIS_ROM)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--model-path", default="tetris_ppo.zip")
    parser.add_argument("--vecnorm-path", default="tetris_vecnormalize")
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--no-improvement-evals", type=int, default=8)
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-obs-norm", action="store_true")
    parser.add_argument("--no-next-piece", action="store_true")
    parser.add_argument("--anchor-model", default="",
                        help="Frozen reference policy used as a KL anchor")
    parser.add_argument("--anchor-coef", type=float, default=0.05)
    parser.add_argument("--baseline-lines", type=float, default=None)
    parser.add_argument("--probe-interval", type=int, default=25_000)
    parser.add_argument("--probe-demos", default="logs/demos/demos_*.npz",
                        help="Glob of demonstration files used to build the probe set")
    return parser.parse_args()


def measure_baseline(model, eval_env, n_episodes):
    """Deterministic mean lines cleared of `model` on the eval env."""
    lines = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _rewards, dones, infos = eval_env.step(action)
            done = bool(dones[0])
            info = infos[0]
        lines.append(int(info.get("lines_cleared", 0)))
    return float(sum(lines) / max(1, len(lines)))


def train(args=None):
    if args is None:
        args = parse_args()

    include_next_piece = not args.no_next_piece
    norm_obs = not args.no_obs_norm
    os.makedirs(args.log_dir, exist_ok=True)

    venv = build_env(args, include_next_piece)
    if args.resume and os.path.exists(args.vecnorm_path) and norm_obs:
        env = VecNormalize.load(args.vecnorm_path, venv)
    else:
        env = VecNormalize(venv, norm_obs=norm_obs, norm_reward=True, clip_obs=10.0)

    eval_venv = build_eval_env(args, include_next_piece)
    eval_env = VecNormalize(
        eval_venv, norm_obs=norm_obs, norm_reward=False, clip_obs=10.0
    )
    eval_env.training = False

    policy_kwargs = {
        "features_extractor_class": TetrisFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": list(args.net_arch), "vf": list(args.net_arch)},
    }

    model_class = AnchoredPPO if args.anchor_model else PPO
    if args.resume and os.path.exists(args.model_path):
        print("[RESUME] Loading existing model...")
        model = model_class.load(
            args.model_path, env=env,
            tensorboard_log=os.path.join(args.log_dir, "tensorboard/"),
            device="cpu",
        )
    else:
        model = model_class(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.join(args.log_dir, "tensorboard/"),
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            device="cpu",
        )

    if args.anchor_model:
        print(f"[ANCHOR] Loading frozen anchor from {args.anchor_model}")
        anchor = PPO.load(args.anchor_model, device="cpu")
        model.set_anchor(anchor, coef=args.anchor_coef)

    if args.baseline_lines is None:
        baseline_lines = measure_baseline(model, eval_env, args.n_eval_episodes)
        print(f"[BASELINE] Starting model clears {baseline_lines:.2f} lines")
    else:
        baseline_lines = args.baseline_lines

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=args.no_improvement_evals,
        min_evals=args.no_improvement_evals + 2,
        verbose=1,
    )
    eval_callback = NetrisEvalCallback(
        eval_env,
        log_path=os.path.join(args.log_dir, "eval_logs/"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        callback_after_eval=stop_callback,
        deterministic=True,
        render=False,
        best_model_dir=os.path.join(args.log_dir, "best_model"),
        history_path=os.path.join(args.log_dir, "eval_logs/lines_history.csv"),
        baseline_mean_lines=baseline_lines,
        run_number=args.run_number,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(args.log_dir, "checkpoints/"),
        name_prefix="tetris_model",
    )
    metrics_callback = TetrisMetricsCallback(
        log_dir=os.path.join(args.log_dir, "metrics")
    )
    probe_callback = ProbeCallback(
        probe_path=os.path.join(args.log_dir, "probes/probe_set.npz"),
        demos_glob=args.probe_demos,
        probe_interval=args.probe_interval,
    )

    callbacks = CallbackList([
        eval_callback, checkpoint_callback, metrics_callback, probe_callback,
    ])

    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks)
    finally:
        model.save(args.model_path)
        env.save(args.vecnorm_path)


def main():
    train()


if __name__ == "__main__":
    main()
