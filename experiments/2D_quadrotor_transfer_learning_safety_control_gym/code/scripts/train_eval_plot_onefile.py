
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback

from quadrotor_transfer.envs import (
    FixedInitialStateWrapper,
    ReducedQuadrotor2DNavigationEnv,
    ReducedQuadrotorConfig,
    sample_initial_states,
)

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def lambda_name(value: float) -> str:
    s = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"lambda_{s}"

def discount_sum(values: List[float], gamma: float) -> float:
    total = 0.0
    power = 1.0
    for v in values:
        total += power * float(v)
        power *= gamma
    return float(total)

def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    arr = np.asarray(xs, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))

def make_cfg(args: argparse.Namespace) -> ReducedQuadrotorConfig:
    return ReducedQuadrotorConfig(
        mass=args.mass,
        gravity=args.gravity,
        dt=args.dt,
        max_steps=args.max_steps,
        x_goal=args.x_goal,
        z_goal=args.z_goal,
        q_x=args.q_x,
        q_z=args.q_z,
        reward_scale=args.reward_scale,
        target_radius=args.target_radius,
        target_bonus=args.target_bonus,
        terminate_on_goal=args.terminate_on_goal,
        theta_max=math.radians(args.theta_max_deg),
        delta_thrust_max=args.delta_thrust_max,
        lambda_penalty=args.lambda_penalty,
        x_bound=args.x_bound,
        z_min=args.z_min,
        z_max=args.z_max,
        x_dot_bound=args.x_dot_bound,
        z_dot_bound=args.z_dot_bound,
        terminate_on_out_of_bounds=not args.no_out_of_bounds_termination,
        init_x_low=args.init_x_low,
        init_x_high=args.init_x_high,
        init_z_low=args.init_z_low,
        init_z_high=args.init_z_high,
        init_x_dot_low=args.init_x_dot_low,
        init_x_dot_high=args.init_x_dot_high,
        init_z_dot_low=args.init_z_dot_low,
        init_z_dot_high=args.init_z_dot_high,
    )

def make_single_env(cfg: ReducedQuadrotorConfig, seed: int | None = None, init_states: np.ndarray | None = None):
    def _make():
        env = ReducedQuadrotor2DNavigationEnv(config=cfg)
        if init_states is not None:
            env = FixedInitialStateWrapper(env, init_states)
        env = Monitor(env)
        if seed is not None:
            try:
                env.reset(seed=seed)
            except Exception:
                pass
        return env
    return _make

def make_train_env(cfg: ReducedQuadrotorConfig, n_envs: int, seed: int, norm_obs: bool, norm_reward: bool, clip_obs: float, clip_reward: float):
    env_fns = [make_single_env(cfg, seed=seed + i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)
    if norm_obs or norm_reward:
        env = VecNormalize(
            env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=cfg.__dict__.get("gamma", 0.99),
        )
    return env

def make_eval_env(cfg: ReducedQuadrotorConfig, init_states: np.ndarray, train_env=None):
    raw = DummyVecEnv([make_single_env(cfg, init_states=init_states)])
    raw = VecMonitor(raw)
    if isinstance(train_env, VecNormalize):
        env = VecNormalize(raw, norm_obs=train_env.norm_obs, norm_reward=False, clip_obs=train_env.clip_obs)
        sync_envs_normalization(train_env, env)
        env.training = False
        env.norm_reward = False
        return env
    return raw

def set_policy_std(model: PPO, std_init: Tuple[float, float]) -> None:
    if not hasattr(model.policy, "log_std"):
        return
    with th.no_grad():
        values = th.as_tensor(np.log(np.asarray(std_init, dtype=np.float32)), device=model.policy.log_std.device)
        if model.policy.log_std.shape == values.shape:
            model.policy.log_std.copy_(values)
        else:

            model.policy.log_std.copy_(values.reshape(model.policy.log_std.shape))

def evaluate_policy_collect(
    model: PPO,
    cfg: ReducedQuadrotorConfig,
    train_env,
    init_states: np.ndarray,
    gamma: float,
    stochastic: bool = False,
    save_rows: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    env = make_eval_env(cfg, init_states, train_env=train_env)
    rows: List[Dict[str, float]] = []

    disc_task_returns: List[float] = []
    disc_train_returns: List[float] = []
    disc_theta_vars: List[float] = []
    theta_var_sums: List[float] = []
    final_distances: List[float] = []
    success_flags: List[float] = []

    obs = env.reset()
    try:
        for ep in range(len(init_states)):
            task_rewards: List[float] = []
            train_rewards: List[float] = []
            theta_vars: List[float] = []
            success = False
            final_distance = float("nan")

            for step in range(cfg.max_steps):
                action, _ = model.predict(obs, deterministic=not stochastic)
                obs, reward_vec, done, infos = env.step(action)
                info = dict(infos[0])

                task_reward = float(info.get("task_reward", np.asarray(reward_vec).reshape(-1)[0]))
                train_reward = float(info.get("train_reward", np.asarray(reward_vec).reshape(-1)[0]))
                theta_var = float(info.get("theta_variation", info.get("variation_cost", 0.0)))
                final_distance = float(info.get("distance_to_target", final_distance))
                success = success or bool(info.get("success", False))

                task_rewards.append(task_reward)
                train_rewards.append(train_reward)
                theta_vars.append(theta_var)

                if save_rows:
                    action_flat = np.asarray(action, dtype=float).reshape(-1)
                    rows.append({
                        "episode": ep,
                        "step": step,
                        "time": step * cfg.dt,
                        "x": float(info.get("x", np.nan)),
                        "x_dot": float(info.get("x_dot", np.nan)),
                        "z": float(info.get("z", np.nan)),
                        "z_dot": float(info.get("z_dot", np.nan)),
                        "action_norm_delta_thrust": float(action_flat[0]) if action_flat.size > 0 else float("nan"),
                        "action_norm_theta_star": float(action_flat[1]) if action_flat.size > 1 else float("nan"),
                        "delta_thrust": float(info.get("delta_thrust", np.nan)),
                        "theta_star": float(info.get("theta_star", np.nan)),
                        "theta_variation": theta_var,
                        "distance_to_target": final_distance,
                        "task_reward": task_reward,
                        "train_reward": train_reward,
                        "success": bool(info.get("success", False)),
                        "out_of_bounds": bool(info.get("out_of_bounds", False)),
                    })

                if bool(np.asarray(done).reshape(-1)[0]):
                    break

            disc_task_returns.append(discount_sum(task_rewards, gamma))
            disc_train_returns.append(discount_sum(train_rewards, gamma))
            disc_theta_vars.append(discount_sum(theta_vars, gamma))
            theta_var_sums.append(float(np.sum(theta_vars)))
            final_distances.append(final_distance)
            success_flags.append(float(success))
    finally:
        env.close()

    task_mean, task_std = mean_std(disc_task_returns)
    train_mean, train_std = mean_std(disc_train_returns)
    dvar_mean, dvar_std = mean_std(disc_theta_vars)
    var_mean, var_std = mean_std(theta_var_sums)
    dist_mean, dist_std = mean_std(final_distances)

    summary = {
        "discounted_task_return_mean": task_mean,
        "discounted_task_return_std": task_std,
        "discounted_train_return_mean": train_mean,
        "discounted_train_return_std": train_std,
        "discounted_theta_variation_mean": dvar_mean,
        "discounted_theta_variation_std": dvar_std,
        "theta_variation_sum_mean": var_mean,
        "theta_variation_sum_std": var_std,
        "success_rate": float(np.mean(success_flags)) if success_flags else float("nan"),
        "final_distance_mean": dist_mean,
        "final_distance_std": dist_std,
    }
    return summary, rows

class EvalCurveCallback(BaseCallback):
    def __init__(self, cfg: ReducedQuadrotorConfig, init_states: np.ndarray, out_dir: Path, eval_every: int, gamma: float, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.init_states = init_states
        self.out_dir = out_dir
        self.eval_every = int(eval_every)
        self.gamma = float(gamma)
        self.records: List[Dict[str, float]] = []
        self._last_eval = -1

    def _on_step(self) -> bool:
        if self.eval_every <= 0:
            return True
        if self.num_timesteps < self.eval_every:
            return True
        if self.num_timesteps - self._last_eval < self.eval_every:
            return True
        self._last_eval = self.num_timesteps
        summary, _ = evaluate_policy_collect(
            self.model,
            self.cfg,
            self.training_env,
            self.init_states,
            gamma=self.gamma,
            stochastic=False,
            save_rows=False,
        )
        row = {"timesteps": int(self.num_timesteps), **summary}
        self.records.append(row)
        write_csv(self.out_dir / "training_eval_curve.csv", self.records)
        if self.verbose:
            print(
                f"[EvalCurve] t={self.num_timesteps} "
                f"disc_task={summary['discounted_task_return_mean']:+.3f} "
                f"disc_theta_var={summary['discounted_theta_variation_mean']:.4f} "
                f"success={summary['success_rate']:.2f} "
                f"final_dist={summary['final_distance_mean']:.4f}"
            )
        return True

def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

def plot_eval_curves(curve_csv: Path, out_dir: Path, lambda_penalty: float | None = None) -> None:
    import pandas as pd
    df = pd.read_csv(curve_csv)
    if df.empty:
        return
    suffix = "" if lambda_penalty is None else f" (lambda={lambda_penalty:g})"

    plt.figure()
    plt.plot(df["timesteps"], df["discounted_task_return_mean"], marker="o")
    plt.xlabel("training timesteps")
    plt.ylabel("discounted task return")
    plt.title(f"Learning curve{suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_discounted_task_return.png", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(df["timesteps"], df["discounted_theta_variation_mean"], marker="o")
    plt.xlabel("training timesteps")
    plt.ylabel("discounted theta-reference variation")
    plt.title(f"Discounted theta-reference variation during training{suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "discounted_theta_variation_during_training.png", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(df["timesteps"], df["theta_variation_sum_mean"], marker="o")
    plt.xlabel("training timesteps")
    plt.ylabel("theta-reference variation sum")
    plt.title(f"Theta-reference variation sum during training{suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "theta_variation_sum_during_training.png", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(df["timesteps"], df["success_rate"], marker="o")
    plt.xlabel("training timesteps")
    plt.ylabel("success rate")
    plt.title(f"Success rate during training{suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate_during_training.png", dpi=220)
    plt.close()

def plot_final_trajectory(traj_csv: Path, out_dir: Path, episode: int = 0, lambda_penalty: float | None = None) -> None:
    import pandas as pd
    df = pd.read_csv(traj_csv)
    ep = df[df["episode"] == episode].copy()
    if ep.empty:
        raise RuntimeError(f"No trajectory rows for episode {episode}")
    suffix = "" if lambda_penalty is None else f" (lambda={lambda_penalty:g})"

    def save(cols: List[str], title: str, ylabel: str, filename: str) -> None:
        plt.figure()
        for col in cols:
            plt.plot(ep["time"], ep[col], label=col)
        plt.xlabel("time [s]")
        plt.ylabel(ylabel)
        plt.title(f"{title}{suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()

    save(["x", "z"], "Position states", "position", "final_position_x_z.png")
    save(["x_dot", "z_dot"], "Velocity states", "velocity", "final_velocity_xdot_zdot.png")
    save(["theta_star"], "Theta reference input", "theta_star [rad]", "final_input_theta_star.png")
    save(["delta_thrust"], "Delta thrust input", "delta_thrust [N]", "final_input_delta_thrust.png")
    save(["distance_to_target"], "Distance to target", "distance", "final_distance_to_target.png")

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(ep["time"], ep["x"], label="x")
    axes[0].plot(ep["time"], ep["z"], label="z")
    axes[0].set_ylabel("position")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep["time"], ep["x_dot"], label="x_dot")
    axes[1].plot(ep["time"], ep["z_dot"], label="z_dot")
    axes[1].set_ylabel("velocity")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(ep["time"], ep["theta_star"], label="theta_star")
    axes[2].set_ylabel("theta ref [rad]")
    axes[2].legend(); axes[2].grid(True)

    axes[3].plot(ep["time"], ep["delta_thrust"], label="delta_thrust")
    axes[3].set_ylabel("Delta T [N]")
    axes[3].legend(); axes[3].grid(True)

    axes[4].plot(ep["time"], ep["distance_to_target"], label="distance")
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("distance")
    axes[4].legend(); axes[4].grid(True)

    fig.suptitle(f"Final evaluation trajectory{suffix}")
    fig.tight_layout()
    fig.savefig(out_dir / "final_evaluation_trajectory_overview.png", dpi=220)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reduced quadrotor and automatically plot learning/variation/final trajectory.")

    parser.add_argument("--lambda_penalty", type=float, default=0.0)
    parser.add_argument("--total_timesteps", type=int, default=800_000)
    parser.add_argument("--eval_every", type=int, default=50_000)
    parser.add_argument("--eval_rollouts", type=int, default=10)
    parser.add_argument("--final_rollouts", type=int, default=1)
    parser.add_argument("--final_init_states_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--out_root", type=Path, default=Path("results") / "onefile_train_eval_plots")
    parser.add_argument("--model_root", type=Path, default=Path("models") / "onefile_train_eval_models")
    parser.add_argument("--run_tag", type=str, default="onefile")

    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--mass", type=float, default=0.027)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--x_goal", type=float, default=1.3)
    parser.add_argument("--z_goal", type=float, default=1.3)
    parser.add_argument("--q_x", type=float, default=1.0)
    parser.add_argument("--q_z", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=100.0)
    parser.add_argument("--target_radius", type=float, default=0.10)
    parser.add_argument("--target_bonus", type=float, default=10.0)
    parser.add_argument("--terminate_on_goal", action="store_true")
    parser.add_argument("--theta_max_deg", type=float, default=35.0)
    parser.add_argument("--delta_thrust_max", type=float, default=None)
    parser.add_argument("--x_bound", type=float, default=2.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--x_dot_bound", type=float, default=30.0)
    parser.add_argument("--z_dot_bound", type=float, default=30.0)
    parser.add_argument("--no_out_of_bounds_termination", action="store_true", default=True)

    parser.add_argument("--init_x_low", type=float, default=-1.3)
    parser.add_argument("--init_x_high", type=float, default=-1.1)
    parser.add_argument("--init_z_low", type=float, default=0.35)
    parser.add_argument("--init_z_high", type=float, default=0.55)
    parser.add_argument("--init_x_dot_low", type=float, default=-1.0)
    parser.add_argument("--init_x_dot_high", type=float, default=1.0)
    parser.add_argument("--init_z_dot_low", type=float, default=-1.0)
    parser.add_argument("--init_z_dot_high", type=float, default=1.0)

    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.994)
    parser.add_argument("--gae_lambda", type=float, default=0.98)
    parser.add_argument("--clip_range", type=float, default=0.10)
    parser.add_argument("--vf_coef", type=float, default=0.8)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--normalize_obs", action="store_true", default=True)
    parser.add_argument("--normalize_reward", action="store_true", default=True)
    parser.add_argument("--clip_obs", type=float, default=10.0)
    parser.add_argument("--clip_reward", type=float, default=10.0)
    parser.add_argument("--std_init_normalized", type=float, nargs=2, default=[0.35, 0.35])
    return parser.parse_args()

def plot_final_trajectory_mean_std(traj_csv, out_dir, lambda_penalty=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    traj_csv = Path(traj_csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(traj_csv)

    if "episode" not in df.columns:
        print("[MeanStdPlot] No episode column found; skipping mean/std trajectory plots.")
        return

    n_ep = int(df["episode"].nunique())
    plot_dir = out_dir / "mean_std_final_trajectory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    mean_df = df.groupby("time").mean(numeric_only=True).reset_index()
    std_df = df.groupby("time").std(numeric_only=True).reset_index().fillna(0.0)

    mean_df.to_csv(plot_dir / "mean_trajectory.csv", index=False)
    std_df.to_csv(plot_dir / "std_trajectory.csv", index=False)

    lam_text = "" if lambda_penalty is None else f", lambda={lambda_penalty:g}"

    def save_plot(cols, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        for col in cols:
            if col not in mean_df.columns:
                continue
            m = mean_df[col]
            s = std_df[col] if col in std_df.columns else 0.0
            plt.plot(mean_df["time"], m, label=f"mean {col}")
            plt.fill_between(mean_df["time"], m - s, m + s, alpha=0.2)
        plt.xlabel("time [s]")
        plt.ylabel(ylabel)
        plt.title(f"{title}{lam_text}, mean?std over {n_ep} episodes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=220)
        plt.close()

    save_plot(["x", "z"], "Position states", "position", "mean_std_position_x_z.png")
    save_plot(["x_dot", "z_dot"], "Velocity states", "velocity", "mean_std_velocity_xdot_zdot.png")
    save_plot(["theta_star"], "Theta-reference input", "theta_star [rad]", "mean_std_theta_star.png")
    save_plot(["delta_thrust"], "Delta thrust input", "Delta T [N]", "mean_std_delta_thrust.png")
    save_plot(["distance_to_target"], "Distance to target", "distance", "mean_std_distance_to_target.png")
    save_plot(["task_reward", "train_reward"], "Reward signals", "reward", "mean_std_rewards.png")

    print(f"[MeanStdPlot] Saved mean?std final trajectory plots to: {plot_dir}")

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    cfg = make_cfg(args)
    stamp = now_stamp()
    run_name = f"reduced_{lambda_name(args.lambda_penalty)}_{args.run_tag}_{stamp}"
    out_dir = Path(args.out_root) / run_name
    model_dir = Path(args.model_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "config.json", {**vars(args), "run_name": run_name, "config": cfg.to_dict(), "action_limits": cfg.action_limit_summary()})

    curve_init_states = sample_initial_states(
        args.eval_rollouts,
        seed=args.seed + 1000,
        x_low=args.init_x_low,
        x_high=args.init_x_high,
        z_low=args.init_z_low,
        z_high=args.init_z_high,
        x_dot_low=args.init_x_dot_low,
        x_dot_high=args.init_x_dot_high,
        z_dot_low=args.init_z_dot_low,
        z_dot_high=args.init_z_dot_high,
    )
    np.save(out_dir / "curve_eval_initial_states.npy", curve_init_states)

    train_env = make_train_env(
        cfg,
        n_envs=args.n_envs,
        seed=args.seed,
        norm_obs=args.normalize_obs,
        norm_reward=args.normalize_reward,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=None,
    )
    set_policy_std(model, tuple(args.std_init_normalized))

    callback = EvalCurveCallback(cfg, curve_init_states, out_dir, eval_every=args.eval_every, gamma=args.gamma, verbose=1)

    print("\n===================================================")
    print(f"run_name       = {run_name}")
    print(f"out_dir        = {out_dir}")
    print(f"model_dir      = {model_dir}")
    print(f"lambda         = {args.lambda_penalty:g}")
    print(f"init velocities= x_dot,z_dot ~ U({args.init_x_dot_low}, {args.init_x_dot_high}) and U({args.init_z_dot_low}, {args.init_z_dot_high})")
    print("reward train   = task_reward - lambda*|Delta theta_star|")
    print("===================================================\n")

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)

    final_model_path = model_dir / "final_model.zip"
    model.save(str(final_model_path))
    if isinstance(train_env, VecNormalize):
        train_env.save(str(model_dir / "vecnormalize.pkl"))
    train_env.close()

    curve_csv = out_dir / "training_eval_curve.csv"
    if curve_csv.exists():
        plot_eval_curves(curve_csv, out_dir, args.lambda_penalty)

    raw_final = DummyVecEnv([make_single_env(cfg, init_states=sample_initial_states(
        args.final_rollouts,
        seed=args.seed + 2000,
        x_low=args.init_x_low,
        x_high=args.init_x_high,
        z_low=args.init_z_low,
        z_high=args.init_z_high,
        x_dot_low=args.init_x_dot_low,
        x_dot_high=args.init_x_dot_high,
        z_dot_low=args.init_z_dot_low,
        z_dot_high=args.init_z_dot_high,
    ))])
    raw_final = VecMonitor(raw_final)
    vec_path = model_dir / "vecnormalize.pkl"
    if vec_path.exists():
        final_env = VecNormalize.load(str(vec_path), raw_final)
        final_env.training = False
        final_env.norm_reward = False
    else:
        final_env = raw_final
    final_model = PPO.load(str(final_model_path), env=final_env, device=args.device)

    if args.final_init_states_path:
        final_init_states = np.load(args.final_init_states_path).astype(np.float32)
        if final_init_states.ndim != 2 or final_init_states.shape[1] < 4:
            raise ValueError(f"final_init_states_path must have shape (N,4) or (N,>=4); got {final_init_states.shape}")
        final_init_states = final_init_states[:, :4]
        args.final_rollouts = int(final_init_states.shape[0])
        print(f"Using fixed final evaluation initial states: {args.final_init_states_path}")
        print(f"Number of final evaluation rollouts: {args.final_rollouts}")
    else:
        final_init_states = sample_initial_states(
            args.final_rollouts,
            seed=args.seed + 2000,
            x_low=args.init_x_low,
            x_high=args.init_x_high,
            z_low=args.init_z_low,
            z_high=args.init_z_high,
            x_dot_low=args.init_x_dot_low,
            x_dot_high=args.init_x_dot_high,
            z_dot_low=args.init_z_dot_low,
            z_dot_high=args.init_z_dot_high,
        )
    final_summary, rows = evaluate_policy_collect(final_model, cfg, final_env, final_init_states, gamma=args.gamma, stochastic=False, save_rows=True)
    final_env.close()

    traj_csv = out_dir / "final_evaluation_trajectory.csv"
    write_csv(traj_csv, rows)
    write_json(out_dir / "final_evaluation_summary.json", final_summary)
    plot_final_trajectory(traj_csv, out_dir, episode=0, lambda_penalty=args.lambda_penalty)

    print("\n===================================================")
    print("DONE")
    print(f"Final model:        {final_model_path}")
    print(f"Final VecNormalize: {model_dir / 'vecnormalize.pkl'}")
    print(f"Training curve CSV: {curve_csv}")
    print(f"Final trajectory:   {traj_csv}")
    print(f"Plots folder:       {out_dir}")
    print(f"Final discounted task return:      {final_summary['discounted_task_return_mean']:+.6f}")
    print(f"Final discounted theta variation:  {final_summary['discounted_theta_variation_mean']:.6f}")
    print(f"Final theta variation sum:         {final_summary['theta_variation_sum_mean']:.6f}")
    print("===================================================\n")

if __name__ == "__main__":
    main()
