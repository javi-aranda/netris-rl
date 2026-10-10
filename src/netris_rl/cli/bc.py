"""Behavior cloning CLI (`netris-bc`).

Trains the CNN policy on teacher/DAgger demonstrations with cross-entropy,
optionally warm-starting from an existing model.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from netris import NetrisEnvConfig, TetrisEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from netris_rl.features import TetrisFeaturesExtractor

OBS_KEYS = [
    "board",
    "current_piece",
    "next_piece",
    "row_completion",
    "holes",
    "bumpiness",
    "aggregate_height",
]


def make_env(rom_path, include_next_piece):
    return TetrisEnv(
        rom_path,
        log_level="error",
        config=NetrisEnvConfig(include_next_piece=include_next_piece),
    )


def load_demos(patterns, keys=None):
    keys = keys or OBS_KEYS
    if isinstance(patterns, str):
        patterns = [patterns]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No demo files match {patterns}")
    parts = [np.load(path) for path in paths]
    data = {
        key: np.concatenate([part[key] for part in parts])
        for key in keys + ["actions"]
    }
    print(f"Loaded {len(data['actions'])} demonstrations from {len(paths)} files")
    return data


def batch_observations(data, indices, keys):
    return {
        key: torch.as_tensor(data[key][indices], dtype=torch.float32)
        for key in keys
    }


def accuracy(policy, data, indices, keys, batch_size=1024):
    correct = 0
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            observations = batch_observations(data, batch, keys)
            distribution = policy.get_distribution(observations)
            predictions = distribution.distribution.probs.argmax(dim=1)
            targets = torch.as_tensor(data["actions"][batch], dtype=torch.long)
            correct += int((predictions == targets).sum())
    policy.train()
    return correct / max(1, len(indices))


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning for Netris")
    parser.add_argument("--demos", nargs="+", default=["logs/demos/demos_*.npz"])
    parser.add_argument("--rom", default=None,
                        help="Path to the GameBoy Tetris ROM (or set NETRIS_ROM)")
    parser.add_argument("--init-model", default="",
                        help="Warm-start from an existing PPO/BC model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output", default="logs/bc/bc_model")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--no-next-piece", action="store_true")
    args = parser.parse_args()

    obs_keys = [
        key for key in OBS_KEYS
        if not args.no_next_piece or key != "next_piece"
    ]
    data = load_demos(args.demos, obs_keys)
    total = len(data["actions"])
    rng = np.random.default_rng(0)
    permutation = rng.permutation(total)
    val_size = int(total * args.val_fraction)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    policy_kwargs = {
        "features_extractor_class": TetrisFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": list(args.net_arch), "vf": list(args.net_arch)},
    }

    env = VecMonitor(DummyVecEnv([
        lambda: make_env(args.rom, include_next_piece=not args.no_next_piece)
    ]))
    if args.init_model:
        print(f"Warm-starting from {args.init_model}")
        model = PPO.load(args.init_model, env=env, device="cpu")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=0,
        )
    policy = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    actions = torch.as_tensor(data["actions"], dtype=torch.long)
    start_time = time.time()
    val_accuracy = 0.0

    for epoch in range(args.epochs):
        rng.shuffle(train_indices)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(train_indices), args.batch_size):
            batch = train_indices[start:start + args.batch_size]
            observations = batch_observations(data, batch, obs_keys)
            distribution = policy.get_distribution(observations)
            log_prob = distribution.log_prob(actions[batch])
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        val_accuracy = accuracy(policy, data, val_indices, obs_keys)
        print(
            f"epoch {epoch + 1:02d}/{args.epochs} "
            f"loss={epoch_loss / max(1, batches):.4f} "
            f"val_accuracy={val_accuracy:.3f} "
            f"elapsed={time.time() - start_time:.0f}s",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"Saved BC model to {args.output}.zip (val accuracy {val_accuracy:.3f})")
    env.close()


if __name__ == "__main__":
    main()
