
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from quadrotor_transfer.envs import (
    FixedInitialStateWrapper,
    ReducedQuadrotorConfig,
    SafeControlGymQuad2DDeploymentEnv,
    sample_initial_states,
)

def lambda_name(value: float) -> str:
    s = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"lambda_{s}"

def safe_float_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")

def safe_lam_tag(value: float) -> str:
    return f"lam_{str(value).replace('-', 'm').replace('.', 'p')}"

def discounted_sum(values: Sequence[float], gamma: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    powers = float(gamma) ** np.arange(arr.size, dtype=float)
    return float(np.dot(powers, arr))

def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def latest_matching_dir(root: Path, lam: float, run_tag_prefix: str) -> Path:

    run_tag = f"{run_tag_prefix}_{safe_lam_tag(lam)}"
    prefix = f"reduced_{lambda_name(lam)}_{run_tag}_"
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:

        prefix = f"reduced_{lambda_name(lam)}_"
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No model/result folder for lambda={lam:g} under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def make_cfg(args: argparse.Namespace, lam: float) -> ReducedQuadrotorConfig:
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
        lambda_penalty=float(lam),
        x_bound=args.x_bound,
        z_min=args.z_min,
        z_max=args.z_max,
        x_dot_bound=args.x_dot_bound,
        z_dot_bound=args.z_dot_bound,
        terminate_on_out_of_bounds=(not args.no_out_of_bounds_termination),
        init_x_low=args.init_x_low,
        init_x_high=args.init_x_high,
        init_z_low=args.init_z_low,
        init_z_high=args.init_z_high,
        init_x_dot_low=args.init_x_dot_low,
        init_x_dot_high=args.init_x_dot_high,
        init_z_dot_low=args.init_z_dot_low,
        init_z_dot_high=args.init_z_dot_high,
        use_semi_implicit_euler=True,
    )

def build_deployment_vec_env(
    cfg: ReducedQuadrotorConfig,
    init_states: np.ndarray,
    omega_n: float,
    zeta: float,
    vec_path: Path,
    args: argparse.Namespace,
):
    def make_one():
        env = SafeControlGymQuad2DDeploymentEnv(
            config=cfg,
            omega_n=float(omega_n),
            zeta=float(zeta),
            theta_star_tau=args.theta_star_tau,
            ctrl_freq=args.ctrl_freq,
            pyb_freq=args.pyb_freq,
            gui=False,
            verbose=False,
            use_default_constraints=False,
            done_on_violation=False,
            done_on_out_of_bound=(not args.no_out_of_bounds_termination),
        )
        env = FixedInitialStateWrapper(env, init_states)
        return env

    raw_env = DummyVecEnv([make_one])
    raw_env = VecMonitor(raw_env)
    env = VecNormalize.load(str(vec_path), raw_env)
    env.training = False
    env.norm_reward = False
    return env

def evaluate_one_setting(
    model_path: Path,
    vec_path: Path,
    cfg: ReducedQuadrotorConfig,
    init_states: np.ndarray,
    lam: float,
    omega_n: float,
    zeta: float,
    args: argparse.Namespace,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    env = build_deployment_vec_env(cfg, init_states, omega_n, zeta, vec_path, args)
    model = PPO.load(str(model_path), env=env, device=args.device)

    rows: List[Dict[str, Any]] = []

    disc_task_returns: List[float] = []
    disc_train_returns: List[float] = []
    disc_theta_vars: List[float] = []
    theta_var_sums: List[float] = []
    final_distances: List[float] = []
    success_flags: List[float] = []
    mean_abs_tracking_errors: List[float] = []
    mean_motor_clip_sums: List[float] = []

    obs = env.reset()
    try:
        for ep in range(len(init_states)):
            task_rewards: List[float] = []
            train_rewards: List[float] = []
            theta_vars: List[float] = []
            tracking_errors: List[float] = []
            motor_clip_sums: List[float] = []
            success = False
            final_distance = float("nan")

            for step in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward_vec, done_vec, infos = env.step(action)
                done = bool(np.asarray(done_vec).reshape(-1)[0])
                info = dict(infos[0])

                task_reward = float(info.get("task_reward", np.asarray(reward_vec).reshape(-1)[0]))
                train_reward = float(info.get("train_reward", np.asarray(reward_vec).reshape(-1)[0]))
                theta_var = float(info.get("theta_variation", info.get("variation_cost", 0.0)))
                final_distance = float(info.get("distance_to_target", final_distance))
                success = success or bool(info.get("success", False))
                tracking_error = float(info.get("theta_tracking_error", np.nan))
                motor_clip_sum = float(info.get("motor_pair_clip_abs_sum", 0.0))

                task_rewards.append(task_reward)
                train_rewards.append(train_reward)
                theta_vars.append(theta_var)
                if np.isfinite(tracking_error):
                    tracking_errors.append(abs(tracking_error))
                motor_clip_sums.append(motor_clip_sum)

                action_flat = np.asarray(action, dtype=float).reshape(-1)
                rows.append({
                    "lambda_penalty": float(lam),
                    "omega_n": float(omega_n),
                    "zeta": float(zeta),
                    "episode": int(ep),
                    "step": int(step),
                    "time": float(step * cfg.dt),
                    "x": float(info.get("x", np.nan)),
                    "x_dot": float(info.get("x_dot", np.nan)),
                    "z": float(info.get("z", np.nan)),
                    "z_dot": float(info.get("z_dot", np.nan)),
                    "theta": float(info.get("theta", np.nan)),
                    "theta_rate": float(info.get("theta_rate", np.nan)),
                    "theta_star": float(info.get("theta_star", np.nan)),
                    "theta_variation": float(theta_var),
                    "theta_tracking_error": tracking_error,
                    "delta_thrust": float(info.get("delta_thrust", np.nan)),
                    "total_thrust_command": float(info.get("total_thrust_command", np.nan)),
                    "motor_pair_t1": float(info.get("motor_pair_t1", np.nan)),
                    "motor_pair_t2": float(info.get("motor_pair_t2", np.nan)),
                    "motor_pair_clip_abs_sum": motor_clip_sum,
                    "distance_to_target": final_distance,
                    "task_reward": task_reward,
                    "train_reward": train_reward,
                    "success": bool(info.get("success", False)),
                    "normalized_delta_thrust_action": float(action_flat[0]) if action_flat.size > 0 else float("nan"),
                    "normalized_theta_star_action": float(action_flat[1]) if action_flat.size > 1 else float("nan"),
                })

                if done:
                    break

            disc_task_returns.append(discounted_sum(task_rewards, args.gamma))
            disc_train_returns.append(discounted_sum(train_rewards, args.gamma))
            disc_theta_vars.append(discounted_sum(theta_vars, args.gamma))
            theta_var_sums.append(float(np.sum(theta_vars)))
            final_distances.append(final_distance)
            success_flags.append(float(success))
            mean_abs_tracking_errors.append(float(np.mean(tracking_errors)) if tracking_errors else float("nan"))
            mean_motor_clip_sums.append(float(np.mean(motor_clip_sums)) if motor_clip_sums else 0.0)
    finally:
        try:
            env.close()
        except Exception:
            pass

    task_mean, task_std = mean_std(disc_task_returns)
    train_mean, train_std = mean_std(disc_train_returns)
    dvar_mean, dvar_std = mean_std(disc_theta_vars)
    var_mean, var_std = mean_std(theta_var_sums)
    dist_mean, dist_std = mean_std(final_distances)
    track_mean, track_std = mean_std(mean_abs_tracking_errors)
    clip_mean, clip_std = mean_std(mean_motor_clip_sums)

    summary = {
        "lambda_penalty": float(lam),
        "omega_n": float(omega_n),
        "zeta": float(zeta),
        "n_rollouts": int(len(init_states)),
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
        "mean_abs_theta_tracking_error_mean": track_mean,
        "mean_abs_theta_tracking_error_std": track_std,
        "mean_motor_pair_clip_abs_sum_mean": clip_mean,
        "mean_motor_pair_clip_abs_sum_std": clip_std,
    }
    return summary, rows

def mean_std_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_df = df.groupby("time").mean(numeric_only=True).reset_index()
    std_df = df.groupby("time").std(numeric_only=True).reset_index().fillna(0.0)
    return mean_df, std_df

def plot_mean_std(df: pd.DataFrame, cols: List[str], title: str, ylabel: str, out_path: Path) -> None:
    mean_df, std_df = mean_std_by_time(df)
    plt.figure()
    for col in cols:
        if col not in mean_df.columns:
            continue
        plt.plot(mean_df["time"], mean_df[col], label=f"mean {col}")
        if col in std_df.columns:
            lower = mean_df[col] - std_df[col]
            upper = mean_df[col] + std_df[col]
            plt.fill_between(mean_df["time"], lower, upper, alpha=0.2)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def make_trajectory_plots(rows: List[Dict[str, Any]], out_dir: Path, lam: float, omega_n: float, zeta: float, n_rollouts: int) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    mean_df, std_df = mean_std_by_time(df)
    mean_df.to_csv(out_dir / "mean_trajectory.csv", index=False)
    std_df.to_csv(out_dir / "std_trajectory.csv", index=False)

    suffix = f"(lambda={lam:g}, omega_n={omega_n:g}, zeta={zeta:g}, mean±std over {n_rollouts} episodes)"
    plot_mean_std(df, ["x", "z"], f"Full deployment position {suffix}", "position", out_dir / "mean_std_position_x_z.png")
    plot_mean_std(df, ["x_dot", "z_dot"], f"Full deployment velocity {suffix}", "velocity", out_dir / "mean_std_velocity_xdot_zdot.png")
    plot_mean_std(df, ["theta", "theta_star"], f"Actual theta vs reference {suffix}", "angle [rad]", out_dir / "mean_std_theta_actual_vs_ref.png")
    plot_mean_std(df, ["theta_tracking_error"], f"Theta tracking error {suffix}", "theta_star - theta [rad]", out_dir / "mean_std_theta_tracking_error.png")
    plot_mean_std(df, ["delta_thrust"], f"Delta thrust input {suffix}", "delta thrust [N]", out_dir / "mean_std_delta_thrust.png")
    plot_mean_std(df, ["distance_to_target"], f"Distance to target {suffix}", "distance", out_dir / "mean_std_distance_to_target.png")
    plot_mean_std(df, ["task_reward"], f"Task reward {suffix}", "task reward", out_dir / "mean_std_task_reward.png")

def plot_summary_metric(summary_df: pd.DataFrame, metric: str, metric_std: str | None, ylabel: str, title: str, out_path: Path, n_rollouts: int, zeta: float) -> None:
    plt.figure()
    for lam, group in summary_df.groupby("lambda_penalty"):
        group = group.sort_values("omega_n")
        x = group["omega_n"].to_numpy(dtype=float)
        y = group[metric].to_numpy(dtype=float)
        if metric_std and metric_std in group:
            yerr = group[metric_std].to_numpy(dtype=float)
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=fr"$\lambda={lam:g}$")
        else:
            plt.plot(x, y, marker="o", label=fr"$\lambda={lam:g}$")
    plt.xlabel(r"$\omega_n$ [rad/s]")
    plt.ylabel(ylabel)
    plt.title(f"{title} (zeta={zeta:g}, mean over {n_rollouts} episodes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_values", type=float, nargs="+", required=True)
    ap.add_argument("--omega_n_values", "--wn_values", dest="omega_n_values", type=float, nargs="+", required=True)
    ap.add_argument("--zeta", type=float, default=0.7)
    ap.add_argument("--n_rollouts", type=int, default=50)
    ap.add_argument("--init_states_path", type=str, default="")
    ap.add_argument("--model_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--run_tag_prefix", type=str, default="lambda_sweep")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--mass", type=float, default=0.027)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--x_goal", type=float, default=1.3)
    ap.add_argument("--z_goal", type=float, default=1.3)
    ap.add_argument("--q_x", type=float, default=1.0)
    ap.add_argument("--q_z", type=float, default=1.0)
    ap.add_argument("--reward_scale", type=float, default=100.0)
    ap.add_argument("--target_radius", type=float, default=0.10)
    ap.add_argument("--target_bonus", type=float, default=10.0)
    ap.add_argument("--terminate_on_goal", action="store_true")
    ap.add_argument("--theta_max_deg", type=float, default=35.0)
    ap.add_argument("--delta_thrust_max", type=float, default=None)
    ap.add_argument("--x_bound", type=float, default=2.0)
    ap.add_argument("--z_min", type=float, default=0.0)
    ap.add_argument("--z_max", type=float, default=2.0)
    ap.add_argument("--x_dot_bound", type=float, default=30.0)
    ap.add_argument("--z_dot_bound", type=float, default=30.0)
    ap.add_argument("--no_out_of_bounds_termination", action="store_true", default=True)
    ap.add_argument("--init_x_low", type=float, default=-1.3)
    ap.add_argument("--init_x_high", type=float, default=-1.1)
    ap.add_argument("--init_z_low", type=float, default=0.35)
    ap.add_argument("--init_z_high", type=float, default=0.55)
    ap.add_argument("--init_x_dot_low", type=float, default=-1.0)
    ap.add_argument("--init_x_dot_high", type=float, default=1.0)
    ap.add_argument("--init_z_dot_low", type=float, default=-1.0)
    ap.add_argument("--init_z_dot_high", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.994)

    ap.add_argument("--theta_star_tau", type=float, default=0.10)
    ap.add_argument("--ctrl_freq", type=int, default=60)
    ap.add_argument("--pyb_freq", type=int, default=240)

    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.init_states_path:
        init_states = np.load(args.init_states_path).astype(np.float32)
        if init_states.ndim != 2 or init_states.shape[1] < 4:
            raise ValueError(f"init_states_path must have shape (N,4) or (N,>=4); got {init_states.shape}")
        init_states = init_states[:, :4]
        args.n_rollouts = int(init_states.shape[0])
        print(f"Using fixed deployment initial states: {args.init_states_path}")
        print(f"Number of deployment rollouts: {args.n_rollouts}")
    else:
        init_states = sample_initial_states(
            args.n_rollouts,
            seed=args.seed + 7000,
            x_low=args.init_x_low,
            x_high=args.init_x_high,
            z_low=args.init_z_low,
            z_high=args.init_z_high,
            x_dot_low=args.init_x_dot_low,
            x_dot_high=args.init_x_dot_high,
            z_dot_low=args.init_z_dot_low,
            z_dot_high=args.init_z_dot_high,
        )
    np.save(args.out_root / "deployment_initial_states.npy", init_states)

    summary_rows: List[Dict[str, Any]] = []
    trajectory_wn_values = {min(args.omega_n_values), max(args.omega_n_values)}

    print("\n===================================================")
    print(f"lambdas       = {args.lambda_values}")
    print(f"omega_n       = {args.omega_n_values}")
    print(f"zeta          = {args.zeta}")
    print(f"n_rollouts    = {args.n_rollouts}")
    print(f"model_root    = {args.model_root}")
    print(f"out_root      = {args.out_root}")
    print("initial states= same distribution as latest reduced eval")
    print("reward        = task_reward only; lambda penalty reported separately through theta variation")
    print("===================================================\n")

    for lam in args.lambda_values:
        model_dir = latest_matching_dir(args.model_root, lam, args.run_tag_prefix)
        model_path = model_dir / "final_model.zip"
        vec_path = model_dir / "vecnormalize.pkl"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not vec_path.exists():
            raise FileNotFoundError(vec_path)

        cfg = make_cfg(args, lam)

        for wn in args.omega_n_values:
            print(f"\n--- lambda={lam:g}, omega_n={wn:g}, zeta={args.zeta:g} ---")
            summary, rows = evaluate_one_setting(model_path, vec_path, cfg, init_states, lam, wn, args.zeta, args)
            summary["model_dir"] = str(model_dir)
            summary_rows.append(summary)

            combo_dir = args.out_root / f"lambda_{safe_float_name(lam)}" / f"zeta_{safe_float_name(args.zeta)}" / f"omega_n_{safe_float_name(wn)}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(combo_dir / "deployment_trajectories.csv", index=False)
            write_json(combo_dir / "deployment_summary.json", summary)

            if float(wn) in trajectory_wn_values:
                make_trajectory_plots(rows, combo_dir / "mean_std_trajectory_plots", lam, wn, args.zeta, args.n_rollouts)

            print(
                f"task_return={summary['discounted_task_return_mean']:+.3f} ± {summary['discounted_task_return_std']:.3f}, "
                f"disc_theta_var={summary['discounted_theta_variation_mean']:.4f} ± {summary['discounted_theta_variation_std']:.4f}, "
                f"success={summary['success_rate']:.2f}, final_dist={summary['final_distance_mean']:.4f}"
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["lambda_penalty", "omega_n"])
    summary_csv = args.out_root / "full_deployment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_summary_metric(
        summary_df,
        "discounted_task_return_mean",
        "discounted_task_return_std",
        "expected cumulative task reward",
        "Full deployment expected cumulative task reward vs omega_n",
        args.out_root / "summary_expected_cumulative_task_reward_vs_omega_n.png",
        args.n_rollouts,
        args.zeta,
    )
    plot_summary_metric(
        summary_df,
        "discounted_theta_variation_mean",
        "discounted_theta_variation_std",
        "discounted theta-reference variation",
        "Full deployment discounted theta-reference variation vs omega_n",
        args.out_root / "summary_discounted_theta_variation_vs_omega_n.png",
        args.n_rollouts,
        args.zeta,
    )
    plot_summary_metric(
        summary_df,
        "theta_variation_sum_mean",
        "theta_variation_sum_std",
        "theta-reference variation sum",
        "Full deployment theta-reference variation sum vs omega_n",
        args.out_root / "summary_theta_variation_sum_vs_omega_n.png",
        args.n_rollouts,
        args.zeta,
    )
    plot_summary_metric(
        summary_df,
        "final_distance_mean",
        "final_distance_std",
        "final distance",
        "Full deployment final distance vs omega_n",
        args.out_root / "summary_final_distance_vs_omega_n.png",
        args.n_rollouts,
        args.zeta,
    )
    plot_summary_metric(
        summary_df,
        "success_rate",
        None,
        "success rate",
        "Full deployment success rate vs omega_n",
        args.out_root / "summary_success_rate_vs_omega_n.png",
        args.n_rollouts,
        args.zeta,
    )

    write_json(args.out_root / "deployment_config.json", vars(args))

    print("\n===================================================")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Summary plots: {args.out_root}")
    print("Mean±std trajectory plots are saved for min/max omega_n inside each lambda/zeta/omega_n folder.")
    print("===================================================\n")

if __name__ == "__main__":
    main()
