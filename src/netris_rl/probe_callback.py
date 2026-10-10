"""Callback that tracks decision quality on a fixed probe set.

Every `probe_interval` steps it evaluates the current policy on a fixed set of
board states labeled by the heuristic teacher and logs to TensorBoard:

- `probe/teacher_agreement`: fraction of states where the policy picks the
  same placement as the expert (interpretability metric over training).
- `probe/mean_max_prob`: average confidence of the chosen action.
- `probe/action_entropy`: policy entropy on those states.
- `probe/mean_value`: average critic value estimate.

This makes it easy to see *when* the agent starts thinking like the expert and
whether it is confident, without running a full evaluation.
"""

import glob
import os

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

OBS_KEYS = [
    "board",
    "current_piece",
    "next_piece",
    "row_completion",
    "holes",
    "bumpiness",
    "aggregate_height",
]


class ProbeCallback(BaseCallback):
    def __init__(self, probe_path="logs/probes/probe_set_v2.npz",
                 demos_glob="logs/demos_v2/demos_v2_*.npz", num_probes=1024,
                 probe_interval=25_000, verbose=0):
        super().__init__(verbose)
        self.probe_path = probe_path
        self.demos_glob = demos_glob
        self.num_probes = num_probes
        self.probe_interval = probe_interval
        self.probe = None
        self.keys = list(OBS_KEYS)
        self._last_probe_step = -1

    def _on_training_start(self):
        self.probe = self._load_or_build_probe()
        if self.probe is not None:
            # Only feed the policy the keys its observation space expects (the
            # probe set may carry extra keys such as `next_piece`).
            spaces = getattr(self.model.observation_space, "spaces", {})
            self.keys = [key for key in OBS_KEYS if key in spaces]
            missing = [key for key in self.keys if key not in self.probe]
            if missing:
                print(f"[ProbeCallback] Probe set missing {missing}; disabling")
                self.probe = None
        if self.probe is None:
            print("[ProbeCallback] No probe set available; callback disabled")
        else:
            print(
                f"[ProbeCallback] Probing on {len(self.probe['actions'])} states "
                f"({len(self.keys)} observation keys)"
            )

    def _build_probe_set(self):
        parts = sorted(glob.glob(self.demos_glob))
        if not parts:
            return None
        with np.load(parts[0]) as data:
            total = len(data["actions"])
            count = min(self.num_probes, total)
            indices = np.linspace(0, total - 1, count).astype(int)
            probe = {key: np.asarray(data[key][indices]) for key in OBS_KEYS}
            probe["actions"] = np.asarray(data["actions"][indices])
        os.makedirs(os.path.dirname(self.probe_path) or ".", exist_ok=True)
        np.savez_compressed(self.probe_path, **probe)
        return probe

    def _load_or_build_probe(self):
        if os.path.exists(self.probe_path):
            with np.load(self.probe_path) as data:
                return {key: np.asarray(data[key]) for key in OBS_KEYS + ["actions"]}
        try:
            return self._build_probe_set()
        except Exception as e:  # noqa: BLE001
            print(f"[ProbeCallback] Could not build probe set: {e}")
            return None

    def _on_step(self):
        if self.probe is None:
            return True
        if (self.num_timesteps - self._last_probe_step >= self.probe_interval
                or self._last_probe_step < 0):
            self._last_probe_step = self.num_timesteps
            self._probe_metrics()
        return True

    def _probe_metrics(self):
        observations = {
            key: th.as_tensor(self.probe[key], dtype=th.float32, device=self.model.device)
            for key in self.keys
        }
        actions = th.as_tensor(
            self.probe["actions"], dtype=th.long, device=self.model.device
        )
        with th.no_grad():
            distribution = self.model.policy.get_distribution(observations)
            probs = distribution.distribution.probs
            predictions = probs.argmax(dim=1)
            agreement = float((predictions == actions).float().mean())
            mean_max_prob = float(probs.max(dim=1).values.mean())
            entropy = float(distribution.distribution.entropy().mean())
            values = float(self.model.policy.predict_values(observations).mean())

        if self.logger is not None:
            self.logger.record("probe/teacher_agreement", agreement)
            self.logger.record("probe/mean_max_prob", mean_max_prob)
            self.logger.record("probe/action_entropy", entropy)
            self.logger.record("probe/mean_value", values)
