# -*- coding: utf-8 -*-
"""Evaluate expected cumulative reward across lambda policies on safe-control-gym.

The plotted quantity is always the unpenalized navigation task reward from
``info['task_reward']``.  The lambda smoothness penalty is logged separately.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadrotor_transfer.envs import (
    FixedInitialStateWrapper,
    ReducedQuadrotor2DNavigationEnv,
    ReducedQuadrotorConfig,
    SafeControlGymQuad2DDeploymentEnv,
    sample_initial_states,
)
from quadrotor_transfer.utils import (
    discount_sum,
    find_policy_artifacts,
    lambda_to_name,
    mean_std,
    safe_float_name,
    scalar,
    unique_floats,
    write_json,
)


def build_config(args: argparse.Namespace, lambda_penalty: float) -> ReducedQuadrotorConfig:
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
        theta_max=np.deg2rad(args.theta_max_deg),
        delta_thrust_max=args.delta_thrust_max,
        lambda_penalty=lambda_penalty,
        x_bound=args.x_bound,
        z_min=args.z_min,
        z_max=args.z_max,
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


def make_reduced_env_factory(cfg: ReducedQuadrotorConfig, init_states: np.ndarray) -> Callable[[], Monitor]:
    def _make():
        env = ReducedQuadrotor2DNavigationEnv(config=cfg)
        env = FixedInitialStateWrapper(env, init_states)
        return Monitor(env)
    return _make


def make_full_env_factory(
    cfg: ReducedQuadrotorConfig,
    init_states: np.ndarray,
    args: argparse.Namespace,
    omega_n: float,
    zeta: float,
) -> Callable[[], Monitor]:
    def _make():
        env = SafeControlGymQuad2DDeploymentEnv(
            config=cfg,
            omega_n=omega_n,
            zeta=zeta,
            theta_star_tau=args.theta_star_tau,
            ctrl_freq=args.ctrl_freq,
            pyb_freq=args.pyb_freq,
            gui=args.gui,
            verbose=args.verbose,
            use_default_constraints=args.use_default_constraints,
            done_on_violation=args.done_on_violation,
            done_on_out_of_bound=not args.no_out_of_bounds_termination,
        )
        env = FixedInitialStateWrapper(env, init_states)
        return Monitor(env)
    return _make


def load_vec_env(env_factory: Callable[[], Monitor], vec_path: Optional[Path]):
    raw = DummyVecEnv([env_factory])
    if vec_path is not None:
        env = VecNormalize.load(str(vec_path), venv=raw)
        env.training = False
        env.norm_reward = False
        return env
    return raw


def info0(infos) -> dict:
    if isinstance(infos, (list, tuple)):
        return dict(infos[0])
    return dict(infos)


def done_bool(done) -> bool:
    return bool(np.asarray(done).reshape(-1)[0])


def evaluate_policy(
    *,
    label: str,
    model_path: Path,
    vec_path: Optional[Path],
    env_factory: Callable[[], Monitor],
    n_rollouts: int,
    max_steps: int,
    device: str,
    deterministic: bool,
    episode_log_every: int,
    trace_dir: Optional[Path] = None,
    trace_prefix: str = "trace",
) -> dict:
    """Evaluate one policy/env pair and return per-episode arrays plus summary."""
    env = load_vec_env(env_factory, vec_path)
    model = PPO.load(str(model_path), env=env, device=device)
    gamma = float(getattr(model, "gamma", 0.995))

    trace_writer = None
    trace_file = None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = open(trace_dir / f"{trace_prefix}.csv", "w", newline="", encoding="utf-8")
        trace_fields = [
            "label", "episode", "step", "task_reward", "train_reward", "distance_to_target", "success",
            "x", "x_dot", "z", "z_dot", "theta", "theta_rate", "theta_star", "theta_star_dot_est",
            "theta_variation", "theta_tracking_error", "delta_thrust", "motor_pair_t1", "motor_pair_t2",
            "motor_pair_clipped", "motor_pair_clip_abs_sum", "action_norm_delta_thrust", "action_norm_theta_star",
        ]
        trace_writer = csv.DictWriter(trace_file, fieldnames=trace_fields)
        trace_writer.writeheader()

    returns: list[float] = []
    train_returns: list[float] = []
    undiscounted_returns: list[float] = []
    lengths: list[int] = []
    success_flags: list[float] = []
    violation_flags: list[float] = []
    theta_variation_sums: list[float] = []
    theta_tracking_abs_means: list[float] = []
    motor_clip_rates: list[float] = []
    motor_clip_abs_sums: list[float] = []
    final_distances: list[float] = []

    obs = env.reset()
    try:
        for ep in range(int(n_rollouts)):
            if episode_log_every and (ep == 0 or ep + 1 == n_rollouts or (ep + 1) % episode_log_every == 0):
                print(f"  {label}: episode {ep + 1}/{n_rollouts}")

            task_rewards: list[float] = []
            train_rewards: list[float] = []
            theta_vars: list[float] = []
            theta_track_abs: list[float] = []
            motor_clipped: list[float] = []
            motor_clip_abs: list[float] = []
            success = False
            violation = False
            final_distance = float("nan")

            for step in range(int(max_steps)):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, vec_reward, done, infos = env.step(action)
                info = info0(infos)
                action_flat = np.asarray(action, dtype=float).reshape(-1)

                task_reward = float(info.get("task_reward", info.get("raw_r", scalar(vec_reward))))
                train_reward = float(info.get("train_reward", scalar(vec_reward)))
                task_rewards.append(task_reward)
                train_rewards.append(train_reward)

                theta_vars.append(float(info.get("theta_variation", info.get("variation_cost", 0.0))))
                if "theta_tracking_error" in info:
                    theta_track_abs.append(abs(float(info["theta_tracking_error"])))
                motor_clipped.append(float(bool(info.get("motor_pair_clipped", False))))
                motor_clip_abs.append(float(info.get("motor_pair_clip_abs_sum", 0.0)))
                success = success or bool(info.get("success", False))
                violation = violation or bool(info.get("constraint_violation", info.get("constraint_violation_rate", 0)))
                final_distance = float(info.get("distance_to_target", final_distance))

                if trace_writer is not None:
                    trace_writer.writerow(
                        {
                            "label": label,
                            "episode": ep,
                            "step": step,
                            "task_reward": task_reward,
                            "train_reward": train_reward,
                            "distance_to_target": final_distance,
                            "success": bool(info.get("success", False)),
                            "x": info.get("x", np.nan),
                            "x_dot": info.get("x_dot", np.nan),
                            "z": info.get("z", np.nan),
                            "z_dot": info.get("z_dot", np.nan),
                            "theta": info.get("theta", np.nan),
                            "theta_rate": info.get("theta_rate", np.nan),
                            "theta_star": info.get("theta_star", np.nan),
                            "theta_star_dot_est": info.get("theta_star_dot_est", np.nan),
                            "theta_variation": info.get("theta_variation", np.nan),
                            "theta_tracking_error": info.get("theta_tracking_error", np.nan),
                            "delta_thrust": info.get("delta_thrust", np.nan),
                            "motor_pair_t1": info.get("motor_pair_t1", np.nan),
                            "motor_pair_t2": info.get("motor_pair_t2", np.nan),
                            "motor_pair_clipped": bool(info.get("motor_pair_clipped", False)),
                            "motor_pair_clip_abs_sum": info.get("motor_pair_clip_abs_sum", np.nan),
                            "action_norm_delta_thrust": action_flat[0] if action_flat.size > 0 else np.nan,
                            "action_norm_theta_star": action_flat[1] if action_flat.size > 1 else np.nan,
                        }
                    )

                if done_bool(done):
                    break

            returns.append(discount_sum(task_rewards, gamma))
            train_returns.append(discount_sum(train_rewards, gamma))
            undiscounted_returns.append(float(np.sum(task_rewards)))
            lengths.append(len(task_rewards))
            success_flags.append(float(success))
            violation_flags.append(float(violation))
            theta_variation_sums.append(float(np.sum(theta_vars)) if theta_vars else 0.0)
            theta_tracking_abs_means.append(float(np.mean(theta_track_abs)) if theta_track_abs else float("nan"))
            motor_clip_rates.append(float(np.mean(motor_clipped)) if motor_clipped else 0.0)
            motor_clip_abs_sums.append(float(np.sum(motor_clip_abs)) if motor_clip_abs else 0.0)
            final_distances.append(final_distance)
    finally:
        env.close()
        if trace_file is not None:
            trace_file.close()

    ret_mean, ret_std = mean_std(returns)
    train_mean, train_std = mean_std(train_returns)
    undiscounted_mean, undiscounted_std = mean_std(undiscounted_returns)
    len_mean, len_std = mean_std(lengths)
    var_mean, var_std = mean_std(theta_variation_sums)
    track_mean, track_std = mean_std(theta_tracking_abs_means)
    dist_mean, dist_std = mean_std(final_distances)
    clip_rate_mean, clip_rate_std = mean_std(motor_clip_rates)
    clip_abs_mean, clip_abs_std = mean_std(motor_clip_abs_sums)

    return {
        "gamma": gamma,
        "returns": np.asarray(returns, dtype=float),
        "train_returns": np.asarray(train_returns, dtype=float),
        "undiscounted_returns": np.asarray(undiscounted_returns, dtype=float),
        "lengths": np.asarray(lengths, dtype=int),
        "success_flags": np.asarray(success_flags, dtype=float),
        "violation_flags": np.asarray(violation_flags, dtype=float),
        "theta_variation_sums": np.asarray(theta_variation_sums, dtype=float),
        "theta_tracking_abs_means": np.asarray(theta_tracking_abs_means, dtype=float),
        "motor_clip_rates": np.asarray(motor_clip_rates, dtype=float),
        "motor_clip_abs_sums": np.asarray(motor_clip_abs_sums, dtype=float),
        "final_distances": np.asarray(final_distances, dtype=float),
        "return_mean": ret_mean,
        "return_std": ret_std,
        "undiscounted_return_mean": undiscounted_mean,
        "undiscounted_return_std": undiscounted_std,
        "train_return_mean": train_mean,
        "train_return_std": train_std,
        "length_mean": len_mean,
        "length_std": len_std,
        "success_rate": float(np.nanmean(success_flags)) if success_flags else float("nan"),
        "violation_rate": float(np.nanmean(violation_flags)) if violation_flags else float("nan"),
        "theta_variation_sum_mean": var_mean,
        "theta_variation_sum_std": var_std,
        "theta_tracking_abs_mean": track_mean,
        "theta_tracking_abs_std": track_std,
        "motor_clip_rate_mean": clip_rate_mean,
        "motor_clip_rate_std": clip_rate_std,
        "motor_clip_abs_sum_mean": clip_abs_mean,
        "motor_clip_abs_sum_std": clip_abs_std,
        "final_distance_mean": dist_mean,
        "final_distance_std": dist_std,
    }


def find_row(rows: list[dict], zeta: float, omega_n: float, lam: float) -> Optional[dict]:
    for row in rows:
        if abs(row["zeta"] - zeta) < 1e-12 and abs(row["omega_n"] - omega_n) < 1e-12 and abs(row["lambda"] - lam) < 1e-12:
            return row
    return None


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], out_dir: Path, zeta: float, omega_values: list[float], lambda_values: list[float], reduced_baseline: dict, std_band: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    omega_arr = np.asarray(omega_values, dtype=float)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, lam in enumerate(lambda_values):
        means = []
        stds = []
        for omega_n in omega_values:
            row = find_row(rows, zeta=zeta, omega_n=omega_n, lam=lam)
            means.append(np.nan if row is None else row["full_expected_cumulative_reward_mean"])
            stds.append(np.nan if row is None else row["full_expected_cumulative_reward_std"])
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        ax.plot(
            omega_arr,
            means_arr,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            label=f"full deploy lambda={lam:g}",
        )
        if std_band:
            ax.fill_between(omega_arr, means_arr - stds_arr, means_arr + stds_arr, alpha=0.18)

    baseline_mean = float(reduced_baseline["return_mean"])
    baseline_std = float(reduced_baseline["return_std"])
    ax.axhline(baseline_mean, linestyle=":", linewidth=2.0, label="reduced lambda=0 baseline")
    if std_band:
        ax.fill_between(omega_arr, baseline_mean - baseline_std, baseline_mean + baseline_std, alpha=0.12)

    ax.set_xlabel("attitude-controller omega_n")
    ax.set_ylabel("expected discounted cumulative task reward")
    ax.set_title(f"Full safe-control-gym deployment across lambdas, zeta={zeta:g}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"expected_cumulative_reward_by_lambda_zeta_{safe_float_name(zeta)}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lambda policies on reduced and full safe-control-gym deployment.")
    parser.add_argument("--lambda_values", type=float, nargs="+", default=[0.0, 0.5, 3.0])
    parser.add_argument("--omega_n_values", type=float, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--zeta_values", type=float, nargs="+", default=[0.7])
    parser.add_argument("--model_root", type=Path, default=Path("models") / "transfer_quadrotor_reduced_policies")
    parser.add_argument("--model_selector", type=str, default="best", choices=["best", "final", "latest"])
    parser.add_argument("--baseline_lambda", type=float, default=0.0)
    parser.add_argument("--baseline_model_selector", type=str, default=None, choices=["best", "final", "latest"])
    parser.add_argument("--out_root", type=Path, default=Path("results") / "transfer_quadrotor_expected_reward_by_lambda")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_rollouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--init_states_path", type=Path, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--std_band", action="store_true")
    parser.add_argument("--episode_log_every", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save_rollout_traces", action="store_true", help="Save per-step full/reduced traces for debugging. Can be large.")

    # Dynamics/task.
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--mass", type=float, default=0.027)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--x_goal", type=float, default=0.0)
    parser.add_argument("--z_goal", type=float, default=1.0)
    parser.add_argument("--q_x", type=float, default=1.0)
    parser.add_argument("--q_z", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--target_radius", type=float, default=0.05)
    parser.add_argument("--target_bonus", type=float, default=2.0)
    parser.add_argument("--terminate_on_goal", action="store_true", help="Default off: fixed horizon even after target is reached.")
    parser.add_argument("--theta_max_deg", type=float, default=22.5)
    parser.add_argument("--delta_thrust_max", type=float, default=None)
    parser.add_argument("--x_bound", type=float, default=2.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--no_out_of_bounds_termination", action="store_true")

    # Sampling range.
    parser.add_argument("--init_x_low", type=float, default=-0.5)
    parser.add_argument("--init_x_high", type=float, default=0.5)
    parser.add_argument("--init_z_low", type=float, default=0.4)
    parser.add_argument("--init_z_high", type=float, default=1.5)
    parser.add_argument("--init_x_dot_low", type=float, default=-0.01)
    parser.add_argument("--init_x_dot_high", type=float, default=0.01)
    parser.add_argument("--init_z_dot_low", type=float, default=-0.01)
    parser.add_argument("--init_z_dot_high", type=float, default=0.01)

    # Full safe-control-gym adapter config.
    parser.add_argument("--ctrl_freq", type=int, default=60)
    parser.add_argument("--pyb_freq", type=int, default=240)
    parser.add_argument("--theta_star_tau", type=float, default=0.10)
    parser.add_argument("--use_default_constraints", action="store_true")
    parser.add_argument("--done_on_violation", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_values = unique_floats(args.lambda_values)
    omega_values = unique_floats(args.omega_n_values)
    zeta_values = unique_floats(args.zeta_values)
    deterministic = not args.stochastic
    baseline_selector = args.baseline_model_selector or args.model_selector

    if args.init_states_path is not None:
        init_states = np.load(str(args.init_states_path)).astype(np.float32)
        if init_states.ndim != 2 or init_states.shape[1] < 4:
            raise ValueError(f"init_states_path must contain shape (N,4) or (N,>=4); got {init_states.shape}")
        init_states = init_states[: args.n_rollouts]
    else:
        init_states = sample_initial_states(
            args.n_rollouts,
            seed=args.seed,
            x_low=args.init_x_low,
            x_high=args.init_x_high,
            z_low=args.init_z_low,
            z_high=args.init_z_high,
            x_dot_low=args.init_x_dot_low,
            x_dot_high=args.init_x_dot_high,
            z_dot_low=args.init_z_dot_low,
            z_dot_high=args.init_z_dot_high,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"expected_cumulative_reward_by_lambda_{args.model_selector}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "eval_initial_states.npy", init_states)
    write_json(out_dir / "config.json", {**vars(args), "lambda_values_resolved": lambda_values, "omega_values_resolved": omega_values, "zeta_values_resolved": zeta_values})

    print("\n===================================================")
    print("Expected cumulative task reward by lambda")
    print(f"model selector = {args.model_selector}")
    print(f"lambda values  = {lambda_values}")
    print(f"zeta values    = {zeta_values}")
    print(f"omega values   = {omega_values}")
    print(f"rollouts       = {len(init_states)}")
    print(f"goal           = ({args.x_goal:g}, {args.z_goal:g})")
    print(f"fixed horizon  = {args.max_steps} steps; terminate_on_goal={args.terminate_on_goal}")
    print(f"output         = {out_dir}")
    print("reward plotted = info['task_reward'] only, not lambda-penalized reward")
    print("===================================================")

    # Reduced baseline.
    baseline_model, baseline_vec, baseline_config = find_policy_artifacts(
        args.model_root,
        args.baseline_lambda,
        selector=baseline_selector,
    )
    reduced_cfg = build_config(args, lambda_penalty=args.baseline_lambda)
    write_json(out_dir / "action_limits.json", reduced_cfg.action_limit_summary())
    trace_dir = out_dir / "rollout_traces" if args.save_rollout_traces else None
    reduced_baseline = evaluate_policy(
        label=f"reduced lambda={args.baseline_lambda:g} baseline",
        model_path=baseline_model,
        vec_path=baseline_vec,
        env_factory=make_reduced_env_factory(reduced_cfg, init_states),
        n_rollouts=len(init_states),
        max_steps=args.max_steps,
        device=args.device,
        deterministic=deterministic,
        episode_log_every=args.episode_log_every,
        trace_dir=trace_dir,
        trace_prefix=f"reduced_baseline_lambda_{safe_float_name(args.baseline_lambda)}",
    )
    print("\nReduced baseline:")
    print(f"  model       = {baseline_model}")
    print(f"  mean reward = {reduced_baseline['return_mean']:+.6f} ± {reduced_baseline['return_std']:.6f}")
    print(f"  success     = {reduced_baseline['success_rate']:.3f}")

    rows: list[dict] = []
    skipped: list[str] = []

    for zeta in zeta_values:
        for lam in lambda_values:
            try:
                model_path, vec_path, config_path = find_policy_artifacts(args.model_root, lam, selector=args.model_selector)
            except FileNotFoundError as exc:
                if args.skip_missing:
                    skipped.append(f"lambda={lam:g}: {exc}")
                    continue
                raise

            for omega_n in omega_values:
                cfg = build_config(args, lambda_penalty=lam)
                trace_prefix = f"full_lambda_{safe_float_name(lam)}_omega_{safe_float_name(omega_n)}_zeta_{safe_float_name(zeta)}"
                result = evaluate_policy(
                    label=f"full lambda={lam:g}, omega={omega_n:g}, zeta={zeta:g}",
                    model_path=model_path,
                    vec_path=vec_path,
                    env_factory=make_full_env_factory(cfg, init_states, args, omega_n=omega_n, zeta=zeta),
                    n_rollouts=len(init_states),
                    max_steps=args.max_steps,
                    device=args.device,
                    deterministic=deterministic,
                    episode_log_every=args.episode_log_every,
                    trace_dir=trace_dir,
                    trace_prefix=trace_prefix,
                )

                if len(result["returns"]) == len(reduced_baseline["returns"]):
                    diff = result["returns"] - reduced_baseline["returns"]
                    diff_mean, diff_std = mean_std(diff)
                else:
                    diff_mean = result["return_mean"] - reduced_baseline["return_mean"]
                    diff_std = float(np.sqrt(result["return_std"] ** 2 + reduced_baseline["return_std"] ** 2))

                row = {
                    "zeta": float(zeta),
                    "omega_n": float(omega_n),
                    "lambda": float(lam),
                    "run_id": lambda_to_name(lam),
                    "model_selector": args.model_selector,
                    "model_path": str(model_path),
                    "vecnormalize_path": str(vec_path) if vec_path else "",
                    "config_path": str(config_path) if config_path else "",
                    "gamma": float(result["gamma"]),
                    "n_rollouts": int(len(init_states)),
                    "full_expected_cumulative_reward_mean": float(result["return_mean"]),
                    "full_expected_cumulative_reward_std": float(result["return_std"]),
                    "full_undiscounted_task_return_mean": float(result["undiscounted_return_mean"]),
                    "full_undiscounted_task_return_std": float(result["undiscounted_return_std"]),
                    "full_expected_cumulative_train_reward_mean": float(result["train_return_mean"]),
                    "full_expected_cumulative_train_reward_std": float(result["train_return_std"]),
                    "reduced_baseline_model_path": str(baseline_model),
                    "reduced_baseline_expected_cumulative_reward_mean": float(reduced_baseline["return_mean"]),
                    "reduced_baseline_expected_cumulative_reward_std": float(reduced_baseline["return_std"]),
                    "full_minus_reduced_baseline_mean": float(diff_mean),
                    "full_minus_reduced_baseline_std": float(diff_std),
                    "full_episode_length_mean": float(result["length_mean"]),
                    "full_episode_length_std": float(result["length_std"]),
                    "reduced_baseline_episode_length_mean": float(reduced_baseline["length_mean"]),
                    "reduced_baseline_episode_length_std": float(reduced_baseline["length_std"]),
                    "full_success_rate": float(result["success_rate"]),
                    "full_constraint_violation_rate": float(result["violation_rate"]),
                    "full_theta_variation_sum_mean": float(result["theta_variation_sum_mean"]),
                    "full_theta_variation_sum_std": float(result["theta_variation_sum_std"]),
                    "full_theta_tracking_abs_mean": float(result["theta_tracking_abs_mean"]),
                    "full_theta_tracking_abs_std": float(result["theta_tracking_abs_std"]),
                    "full_motor_clip_rate_mean": float(result["motor_clip_rate_mean"]),
                    "full_motor_clip_rate_std": float(result["motor_clip_rate_std"]),
                    "full_motor_clip_abs_sum_mean": float(result["motor_clip_abs_sum_mean"]),
                    "full_motor_clip_abs_sum_std": float(result["motor_clip_abs_sum_std"]),
                    "full_final_distance_mean": float(result["final_distance_mean"]),
                    "full_final_distance_std": float(result["final_distance_std"]),
                }
                rows.append(row)
                print(
                    f"\nRESULT | zeta={zeta:g}, omega_n={omega_n:g}, lambda={lam:g} | "
                    f"mean={row['full_expected_cumulative_reward_mean']:+.6f}, "
                    f"minus baseline={row['full_minus_reduced_baseline_mean']:+.6f}, "
                    f"theta-var={row['full_theta_variation_sum_mean']:.4f}, "
                    f"clip-rate={row['full_motor_clip_rate_mean']:.4f}, "
                    f"success={row['full_success_rate']:.3f}"
                )

    csv_path = out_dir / "expected_cumulative_reward_by_lambda_summary.csv"
    save_csv(rows, csv_path)
    write_json(out_dir / "reduced_baseline_summary.json", {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in reduced_baseline.items()
    })

    plot_paths = []
    for zeta in zeta_values:
        plot_paths.append(save_plot(rows, out_dir, zeta, omega_values, lambda_values, reduced_baseline, args.std_band))

    if skipped:
        with open(out_dir / "skipped_policies.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(skipped))

    print("\n===================================================")
    print("Saved outputs")
    print(f"CSV: {csv_path}")
    for path in plot_paths:
        print(f"Plot: {path}")
    if args.save_rollout_traces:
        print(f"Rollout traces: {trace_dir}")
    if skipped:
        print(f"Skipped policies: {out_dir / 'skipped_policies.txt'}")
    print("===================================================\n")


if __name__ == "__main__":
    main()
