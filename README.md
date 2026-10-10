# netris-rl

Training utilities for the [netris](../netris) Game Boy Tetris environment:
behavior cloning from a heuristic teacher, DAgger rounds and PPO fine-tuning
with a KL anchor.

## Install

```bash
pip install netris-rl
```

## CLIs

```bash
# collect teacher demonstrations (optionally DAgger with --policy-model)
netris-collect --rom /path/to/tetris.gb --steps 30000 --workers 2

# behavior cloning (optionally warm-started with --init-model)
netris-bc --demos "logs/demos/demos_*.npz" --epochs 30 --output logs/bc/model

# PPO fine-tuning, optionally anchored to a frozen reference policy
netris-train --rom /path/to/tetris.gb --resume --no-obs-norm \
    --anchor-model logs/bc/model.zip --total-steps 500000

# evaluation
netris-eval --model logs/best_model/best_model.zip --episodes 20
```

## Pipeline used to reach a line-clearing agent

1. `netris-collect` with the analytic Dellacherie teacher.
2. `netris-bc` (behavior cloning).
3. `netris-collect --policy-model` (DAgger), then `netris-bc --init-model`.
4. `netris-train --resume --anchor-model <bc model>` (anchored PPO).

## License

MIT.
