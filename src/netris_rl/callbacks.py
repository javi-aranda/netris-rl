"""Training callbacks: episode metrics, evaluation and best-model selection."""

import csv
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from typing import ClassVar

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class TetrisMetricsCallback(BaseCallback):
    """Persists per-episode metrics and logs them to TensorBoard.

    Environments put an `episode_summary` dict in `info` when an episode ends
    (done or truncated). Because envs may live in subprocesses, writing the CSV
    here (main process) is the only writer.
    """

    FIELDNAMES: ClassVar[list] = [
        "timestamp", "episode", "steps", "pieces", "lines_cleared",
        "max_height", "avg_bumpiness", "avg_holes", "reward_total",
        "reward_lines", "reward_holes", "reward_height", "reward_bumpiness",
        "rotations", "avg_placement_col", "avg_placement_height",
        "action_distribution", "line_clear_distribution",
    ]

    def __init__(self, log_dir="logs/metrics", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "episodes.csv")
        self.episode_buffer = []

    def _on_step(self):
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            summary = info.get("episode_summary") if info else None
            if summary:
                self.episode_buffer.append(summary)
                self._record_tensorboard(summary)

        if self.episode_buffer:
            self._flush_episodes()

        return True

    def _record_tensorboard(self, summary):
        if self.logger is None:
            return
        self.logger.record("train_ep/lines_cleared", summary["lines_cleared"])
        self.logger.record("train_ep/max_height", summary["max_height"])
        self.logger.record("train_ep/avg_holes", summary["avg_holes"])
        self.logger.record("train_ep/avg_bumpiness", summary["avg_bumpiness"])
        self.logger.record("train_ep/reward_lines", summary["reward_lines"])
        self.logger.record("train_ep/reward_holes", summary["reward_holes"])
        self.logger.record("train_ep/reward_height", summary["reward_height"])
        self.logger.record("train_ep/reward_bumpiness", summary["reward_bumpiness"])
        try:
            actions = json.loads(summary.get("action_distribution", "{}"))
        except (ValueError, TypeError):
            actions = {}
        counts = Counter({int(k): int(v) for k, v in actions.items()})
        total = sum(counts.values())
        if total:
            entropy = -sum(
                (count / total) * math.log(count / total)
                for count in counts.values() if count
            )
            self.logger.record("train_ep/action_entropy", entropy)
            top_action, top_count = counts.most_common(1)[0]
            self.logger.record("train_ep/top_action", top_action)
            self.logger.record("train_ep/top_action_share", top_count / total)

    def _flush_episodes(self):
        if not self.episode_buffer:
            return
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.FIELDNAMES, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerows(self.episode_buffer)
        self.episode_buffer = []

    def on_training_end(self):
        self._flush_episodes()


class NetrisEvalCallback(EvalCallback):
    """Evaluation callback that records lines cleared and protects the starter
    (behavior-cloned) model: a new best is only saved when eval mean lines beat
    the measured baseline.
    """

    def __init__(self, *args, run_number=1,
                 history_path="logs/eval_logs/lines_history.csv",
                 best_model_dir="logs/best_model", baseline_mean_lines=0.0,
                 verbose=0, **kwargs):
        best_model_save_path = kwargs.pop("best_model_save_path", None)
        super().__init__(
            *args, verbose=verbose, best_model_save_path=None, **kwargs
        )
        if best_model_save_path is not None:
            best_model_dir = best_model_save_path
        self.best_model_dir = best_model_dir
        self.baseline_mean_lines = float(baseline_mean_lines)
        self.run_number = run_number
        self.history_path = history_path
        self.best_eval_mean_lines = self.baseline_mean_lines
        self.best_eval_max_lines = 0
        self.last_eval_mean_lines = 0.0
        self.last_eval_max_lines = 0
        self.last_eval_mean_reward = 0.0
        self.last_eval_mean_length = 0.0
        self._eval_episode_lines = []
        self._eval_episode_lengths = []
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        info = locals_.get("info") or {}
        if locals_.get("done"):
            self._eval_episode_lines.append(int(info.get("lines_cleared", 0)))
            index = locals_.get("i")
            lengths = locals_.get("current_lengths")
            if index is not None and lengths is not None:
                self._eval_episode_lengths.append(int(lengths[index]))

    def _on_step(self):
        num_evals_before = len(self.evaluations_timesteps)
        continue_training = super()._on_step()
        if len(self.evaluations_timesteps) > num_evals_before:
            try:
                self._on_eval_complete()
            except Exception as e:  # noqa: BLE001
                print(f"[NetrisEvalCallback] Error reporting eval: {e}")
        return continue_training

    def _on_eval_complete(self):
        episode_rewards = self.evaluations_results[-1]
        mean_reward = float(sum(episode_rewards) / max(1, len(episode_rewards)))
        lines = self._eval_episode_lines
        lengths = self._eval_episode_lengths
        self._eval_episode_lines = []
        self._eval_episode_lengths = []

        mean_lines = float(sum(lines) / max(1, len(lines)))
        max_lines = max(lines) if lines else 0
        mean_length = float(sum(lengths) / max(1, len(lengths))) if lengths else 0.0
        timestep = int(self.evaluations_timesteps[-1])

        self.last_eval_mean_lines = mean_lines
        self.last_eval_max_lines = max_lines
        self.last_eval_mean_reward = mean_reward
        self.last_eval_mean_length = mean_length

        is_new_best = mean_lines > self.best_eval_mean_lines
        if is_new_best:
            self.best_eval_mean_lines = mean_lines
            self.best_eval_max_lines = max_lines
            self._save_best_model()
            print(
                f"[NetrisEvalCallback] New best: {mean_lines:.2f} mean lines "
                f"-> {self.best_model_dir}"
            )

        if self.logger is not None:
            self.logger.record("eval/mean_lines", mean_lines)
            self.logger.record("eval/max_lines", max_lines)
            self.logger.record("eval/mean_length", mean_length)
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/best_mean_lines", self.best_eval_mean_lines)

        self._write_history(timestep, lines, mean_reward, mean_length)

    def _save_best_model(self):
        os.makedirs(self.best_model_dir, exist_ok=True)
        self.model.save(os.path.join(self.best_model_dir, "best_model"))
        from stable_baselines3.common.vec_env import VecNormalize

        training_env = self.training_env
        if isinstance(training_env, VecNormalize):
            training_env.save(
                os.path.join(self.best_model_dir, "vecnormalize.pkl")
            )

    def _write_history(self, timestep, lines, mean_reward, mean_length=0.0):
        write_header = not os.path.exists(self.history_path)
        with open(self.history_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "timestep", "episodes", "mean_lines",
                    "max_lines", "mean_reward", "mean_length",
                    "best_mean_lines",
                ])
            writer.writerow([
                datetime.now(UTC).isoformat(),
                timestep,
                len(lines),
                round(sum(lines) / max(1, len(lines)), 4),
                max(lines) if lines else 0,
                round(mean_reward, 4),
                round(mean_length, 2),
                round(self.best_eval_mean_lines, 4),
            ])
