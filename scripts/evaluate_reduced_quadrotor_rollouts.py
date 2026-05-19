# -*- coding: utf-8 -*-
"""Evaluate a reduced lambda policy and save complete state/action trajectories."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadrotor_transfer.envs import (
    FixedInitialStateWrapper,
    ReducedQuadrotor2DNavigationEnv,
    ReducedQuadrotorConfig,
    sample_initial_states,
)
from quadrotor_transfer.utils import discount_sum, find_policy_artifacts, lambda_to_name, mean_std, scalar, write_json


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


def make_eval_env(cfg: ReducedQuadrotorConfig, init_states: np.ndarray):
    def _make():
        env = ReducedQuadrotor2DNavigationEnv(config=cfg)
        env = FixedInitialStateWrapper(env, init_states)
        return Monitor(env)
    return _make


def load_eval_vec_env(cfg: ReducedQuadrotorConfig, init_states: np.ndarray, vec_path: Optional[Path]):
    raw = DummyVecEnv([make_eval_env(cfg, init_states)])
    if vec_path is not None:
        env = VecNormalize.load(str(vec_path), venv=raw)
        env.training = False
        env.norm_reward = False
        return env
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save reduced-model trajectories for one lambda policy.")
    parser.add_argument("--lambda_penalty", type=float, default=0.0)
    parser.add_argument("--model_root", type=Path, default=Path("models") / "transfer_quadrotor_reduced_policies")
    parser.add_argument("--model_selector", type=str, default="best", choices=["best", "final", "latest"])
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions. Default is deterministic.")

    parser.add_argument("--n_rollouts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--init_states_path", type=Path, default=None)

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
    parser.add_argument("--terminate_on_goal", action="store_true")
    parser.add_argument("--theta_max_deg", type=float, default=22.5)
    parser.add_argument("--delta_thrust_max", type=float, default=None)
    parser.add_argument("--x_bound", type=float, default=2.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--no_out_of_bounds_termination", action="store_true")

    parser.add_argument("--init_x_low", type=float, default=-0.5)
    parser.add_argument("--init_x_high", type=float, default=0.5)
    parser.add_argument("--init_z_low", type=float, default=0.4)
    parser.add_argument("--init_z_high", type=float, default=1.5)
    parser.add_argument("--init_x_dot_low", type=float, default=-0.01)
    parser.add_argument("--init_x_dot_high", type=float, default=0.01)
    parser.add_argument("--init_z_dot_low", type=float, default=-0.01)
    parser.add_argument("--init_z_dot_high", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path, vec_path, config_path = find_policy_artifacts(
        args.model_root,
        args.lambda_penalty,
        explicit_model=args.model,
        explicit_vecnormalize=args.vecnormalize,
        selector=args.model_selector,
    )
    cfg = build_config(args, args.lambda_penalty)

    if args.init_states_path is not None:
        init_states = np.load(str(args.init_states_path)).astype(np.float32)
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
    out_dir = args.out_dir or (Path("results") / "transfer_quadrotor_reduced_rollouts" / f"{lambda_to_name(args.lambda_penalty)}_{args.model_selector}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "eval_initial_states.npy", init_states)

    env = load_eval_vec_env(cfg, init_states, vec_path)
    model = PPO.load(str(model_path), env=env, device=args.device)
    gamma = float(getattr(model, "gamma", 0.995))

    rows = []
    episode_returns = []
    episode_train_returns = []
    episode_lengths = []
    success_flags = []
    final_distances = []
    theta_variation_sums = []

    obs = env.reset()
    try:
        for ep in range(args.n_rollouts):
            rewards = []
            train_rewards = []
            success = False
            final_distance = float("nan")
            theta_vars = []
            for step in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, vec_reward, done, infos = env.step(action)
                info = dict(infos[0])
                task_reward = float(info.get("task_reward", scalar(vec_reward)))
                train_reward = float(info.get("train_reward", scalar(vec_reward)))
                rewards.append(task_reward)
                train_rewards.append(train_reward)
                theta_vars.append(float(info.get("theta_variation", info.get("variation_cost", 0.0))))
                success = success or bool(info.get("success", False))
                final_distance = float(info.get("distance_to_target", final_distance))

                action_flat = np.asarray(action, dtype=float).reshape(-1)
                rows.append(
                    {
                        "episode": ep,
                        "step": step,
                        "time": step * cfg.dt,
                        "x": info.get("x", np.nan),
                        "x_dot": info.get("x_dot", np.nan),
                        "z": info.get("z", np.nan),
                        "z_dot": info.get("z_dot", np.nan),
                        "action_norm_delta_thrust": action_flat[0] if action_flat.size > 0 else np.nan,
                        "action_norm_theta_star": action_flat[1] if action_flat.size > 1 else np.nan,
                        "delta_thrust": info.get("delta_thrust", np.nan),
                        "theta_star": info.get("theta_star", np.nan),
                        "theta_variation": info.get("theta_variation", np.nan),
                        "task_reward": task_reward,
                        "train_reward": train_reward,
                        "distance_to_target": final_distance,
                        "success": bool(info.get("success", False)),
                        "out_of_bounds": bool(info.get("out_of_bounds", False)),
                        "delta_thrust_max": cfg.delta_thrust_max,
                        "theta_max": cfg.theta_max,
                    }
                )
                if bool(np.asarray(done).reshape(-1)[0]):
                    break

            episode_returns.append(discount_sum(rewards, gamma))
            episode_train_returns.append(discount_sum(train_rewards, gamma))
            episode_lengths.append(len(rewards))
            success_flags.append(float(success))
            final_distances.append(final_distance)
            theta_variation_sums.append(float(np.sum(theta_vars)) if theta_vars else 0.0)
    finally:
        env.close()

    csv_path = out_dir / "reduced_rollout_trajectories.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ret_mean, ret_std = mean_std(episode_returns)
    train_mean, train_std = mean_std(episode_train_returns)
    len_mean, len_std = mean_std(episode_lengths)
    dist_mean, dist_std = mean_std(final_distances)
    var_mean, var_std = mean_std(theta_variation_sums)
    summary = {
        "lambda_penalty": args.lambda_penalty,
        "model_selector": args.model_selector,
        "model_path": str(model_path),
        "vecnormalize_path": str(vec_path) if vec_path else None,
        "config_path": str(config_path) if config_path else None,
        "gamma": gamma,
        "n_rollouts": args.n_rollouts,
        "expected_cumulative_task_reward_mean": ret_mean,
        "expected_cumulative_task_reward_std": ret_std,
        "expected_cumulative_train_reward_mean": train_mean,
        "expected_cumulative_train_reward_std": train_std,
        "episode_length_mean": len_mean,
        "episode_length_std": len_std,
        "success_rate": float(np.mean(success_flags)) if success_flags else float("nan"),
        "final_distance_mean": dist_mean,
        "final_distance_std": dist_std,
        "theta_variation_sum_mean": var_mean,
        "theta_variation_sum_std": var_std,
        "action_limit_summary": cfg.action_limit_summary(),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "config.json", {**vars(args), "resolved_config": cfg.to_dict()})

    print("\nSaved reduced rollout trajectories")
    print(f"Model: {model_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Mean discounted task reward: {ret_mean:+.6f} ± {ret_std:.6f}")
    print(f"Mean theta variation sum:    {var_mean:.6f} ± {var_std:.6f}")


if __name__ == "__main__":
    main()
