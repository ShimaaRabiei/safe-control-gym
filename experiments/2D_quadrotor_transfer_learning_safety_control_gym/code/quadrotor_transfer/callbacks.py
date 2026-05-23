
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from stable_baselines3.common.vec_env import sync_envs_normalization
except Exception:
    sync_envs_normalization = None

from quadrotor_transfer.utils import discount_sum, mean_std, write_json

class ActionStdCallback(BaseCallback):

    def __init__(
        self,
        std_mode: str,
        log_std_init: np.ndarray,
        log_std_final: np.ndarray,
        total_timesteps: int,
        verbose: int = 0,
        set_initial_on_training_start: bool = True,
    ) -> None:
        super().__init__(verbose=verbose)
        if std_mode == "trainable":
            std_mode = "learnable"
        if std_mode not in {"learnable", "fixed", "anneal"}:
            raise ValueError("std_mode must be learnable, fixed, or anneal")
        self.std_mode = std_mode
        self.log_std_init = np.asarray(log_std_init, dtype=float).reshape(-1)
        self.log_std_final = np.asarray(log_std_final, dtype=float).reshape(-1)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.set_initial_on_training_start = bool(set_initial_on_training_start)

    def _set_log_std(self, value: np.ndarray) -> None:
        if not hasattr(self.model.policy, "log_std"):
            return
        log_std = self.model.policy.log_std
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr[0], int(log_std.numel()))
        if arr.size != int(log_std.numel()):
            raise ValueError(f"log_std size mismatch: callback has {arr.size}, policy has {log_std.numel()}.")
        tensor = th.as_tensor(arr, dtype=log_std.dtype, device=log_std.device).reshape(log_std.shape)
        with th.no_grad():
            log_std.copy_(tensor)

    def _on_training_start(self) -> None:
        if not hasattr(self.model.policy, "log_std"):
            return
        if self.set_initial_on_training_start:
            self._set_log_std(self.log_std_init)
        if self.std_mode in {"fixed", "anneal"}:
            self.model.policy.log_std.requires_grad_(False)
        else:
            self.model.policy.log_std.requires_grad_(True)

    def _on_rollout_start(self) -> None:
        if self.std_mode == "learnable":
            return
        self._set_log_std(self._current_log_std())

    def _current_log_std(self) -> np.ndarray:
        if self.std_mode == "fixed":
            return self.log_std_init
        progress = min(float(self.num_timesteps) / float(self.total_timesteps), 1.0)
        return (1.0 - progress) * self.log_std_init + progress * self.log_std_final

    def _on_step(self) -> bool:
        if self.std_mode != "learnable":
            self._set_log_std(self._current_log_std())
        if self.verbose and hasattr(self.model.policy, "log_std") and self.num_timesteps % 10000 == 0:
            value = self.model.policy.log_std.detach().cpu().numpy().reshape(-1)
            print(f"[ActionStdCallback] num_timesteps={self.num_timesteps}, log_std={value}, std={np.exp(value)}")
        return True

class RewardOrientedBestModelCallback(BaseCallback):

    def __init__(
        self,
        eval_env_factory: Callable[[], object],
        save_dir: Union[Path, str],
        *,
        eval_freq: int = 25000,
        n_eval_episodes: int = 10,
        max_steps: int = 300,
        deterministic: bool = True,
        metric_name: str = "task_return_mean",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env_factory = eval_env_factory
        self.save_dir = Path(save_dir)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_steps = int(max_steps)
        self.deterministic = bool(deterministic)
        self.metric_name = metric_name
        self.best_metric = -np.inf
        self.last_eval_timestep = -1
        self.eval_csv_path = self.save_dir / "best_model_evaluations.csv"
        self.best_metrics_path = self.save_dir / "best_metrics.json"

    def _make_eval_env(self):
        raw = DummyVecEnv([self.eval_env_factory])
        train_vecnorm = self.model.get_vec_normalize_env()
        if train_vecnorm is not None:
            eval_env = VecNormalize(raw, training=False, norm_obs=True, norm_reward=False, clip_obs=train_vecnorm.clip_obs)
            if sync_envs_normalization is not None:
                try:
                    sync_envs_normalization(self.training_env, eval_env)
                except Exception as exc:
                    if self.verbose:
                        print(f"[BestModelCallback] Could not sync VecNormalize statistics: {exc}")
            else:

                eval_env.obs_rms = train_vecnorm.obs_rms
                eval_env.ret_rms = train_vecnorm.ret_rms
            eval_env.training = False
            eval_env.norm_reward = False
            return eval_env
        return raw

    @staticmethod
    def _done(done) -> bool:
        return bool(np.asarray(done).reshape(-1)[0])

    @staticmethod
    def _reward(value) -> float:
        return float(np.asarray(value, dtype=float).reshape(-1)[0])

    def _append_eval_row(self, row: dict) -> None:
        self.eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.eval_csv_path.exists()
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _evaluate(self) -> dict:
        env = self._make_eval_env()
        gamma = float(getattr(self.model, "gamma", 0.995))
        task_returns = []
        discounted_task_returns = []
        train_returns = []
        lengths = []
        success_flags = []
        final_distances = []
        theta_variation_sums = []
        delta_abs_means = []
        theta_abs_means = []

        obs = env.reset()
        try:
            for _ep in range(self.n_eval_episodes):
                task_rewards = []
                train_rewards = []
                theta_vars = []
                delta_abs = []
                theta_abs = []
                success = False
                final_distance = float("nan")
                for _step in range(self.max_steps):
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, vec_reward, done, infos = env.step(action)
                    info = dict(infos[0])
                    task_reward = float(info.get("task_reward", info.get("raw_r", self._reward(vec_reward))))
                    train_reward = float(info.get("train_reward", self._reward(vec_reward)))
                    task_rewards.append(task_reward)
                    train_rewards.append(train_reward)
                    theta_vars.append(float(info.get("theta_variation", info.get("variation_cost", 0.0))))
                    delta_abs.append(abs(float(info.get("delta_thrust", 0.0))))
                    theta_abs.append(abs(float(info.get("theta_star", 0.0))))
                    success = success or bool(info.get("success", False))
                    final_distance = float(info.get("distance_to_target", final_distance))
                    if self._done(done):
                        break
                task_returns.append(float(np.sum(task_rewards)))
                discounted_task_returns.append(discount_sum(task_rewards, gamma))
                train_returns.append(float(np.sum(train_rewards)))
                lengths.append(len(task_rewards))
                success_flags.append(float(success))
                final_distances.append(final_distance)
                theta_variation_sums.append(float(np.sum(theta_vars)) if theta_vars else 0.0)
                delta_abs_means.append(float(np.mean(delta_abs)) if delta_abs else float("nan"))
                theta_abs_means.append(float(np.mean(theta_abs)) if theta_abs else float("nan"))
        finally:
            env.close()

        task_mean, task_std = mean_std(task_returns)
        disc_mean, disc_std = mean_std(discounted_task_returns)
        train_mean, train_std = mean_std(train_returns)
        len_mean, len_std = mean_std(lengths)
        dist_mean, dist_std = mean_std(final_distances)
        var_mean, var_std = mean_std(theta_variation_sums)
        delta_abs_mean, delta_abs_std = mean_std(delta_abs_means)
        theta_abs_mean, theta_abs_std = mean_std(theta_abs_means)
        return {
            "num_timesteps": int(self.num_timesteps),
            "task_return_mean": task_mean,
            "task_return_std": task_std,
            "discounted_task_return_mean": disc_mean,
            "discounted_task_return_std": disc_std,
            "train_return_mean": train_mean,
            "train_return_std": train_std,
            "episode_length_mean": len_mean,
            "episode_length_std": len_std,
            "success_rate": float(np.mean(success_flags)) if success_flags else float("nan"),
            "final_distance_mean": dist_mean,
            "final_distance_std": dist_std,
            "theta_variation_sum_mean": var_mean,
            "theta_variation_sum_std": var_std,
            "mean_abs_delta_thrust_N": delta_abs_mean,
            "std_abs_delta_thrust_N": delta_abs_std,
            "mean_abs_theta_star_rad": theta_abs_mean,
            "std_abs_theta_star_rad": theta_abs_std,
            "gamma": gamma,
            "n_eval_episodes": int(self.n_eval_episodes),
        }

    def _save_best(self, metrics: dict) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self.save_dir / "best_model.zip")
        train_vecnorm = self.model.get_vec_normalize_env()
        if train_vecnorm is not None:
            train_vecnorm.save(str(self.save_dir / "best_vecnormalize.pkl"))
        write_json(self.best_metrics_path, metrics)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.last_eval_timestep >= 0 and self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = int(self.num_timesteps)
        metrics = self._evaluate()
        metric = float(metrics.get(self.metric_name, metrics["task_return_mean"]))
        metrics["selected_metric_name"] = self.metric_name
        metrics["selected_metric_value"] = metric
        metrics["best_metric_before"] = float(self.best_metric)
        improved = bool(metric > self.best_metric)
        metrics["is_new_best"] = improved
        self._append_eval_row(metrics)

        if self.verbose:
            print(
                "[BestModelCallback] "
                f"t={self.num_timesteps} {self.metric_name}={metric:+.6f}, "
                f"success={metrics['success_rate']:.3f}, "
                f"final_dist={metrics['final_distance_mean']:.4f}, "
                f"theta_var={metrics['theta_variation_sum_mean']:.4f}"
            )
        if improved:
            self.best_metric = metric
            metrics["best_metric_after"] = float(self.best_metric)
            self._save_best(metrics)
            if self.verbose:
                print(f"[BestModelCallback] Saved new reward-oriented best model to {self.save_dir / 'best_model.zip'}")
        return True
