# -*- coding: utf-8 -*-
"""Train a reduced-order quadrotor navigation policy with PPO.

This script is intentionally an overlay script: it does not modify the upstream
``safe_control_gym`` source tree.  It trains the reduced translational policy
that later deploys zero-shot through the full safe-control-gym 2D quadrotor
adapter.

Key features added for debugging/reproducibility:
  * timestamped run folders; no old model is deleted;
  * reward-oriented best_model.zip selected by held-out task return;
  * final_model.zip saved at the end of training;
  * warm-start from best/final/latest/path;
  * PPO std modes: learnable, fixed, anneal;
  * physical action-limit and std interpretation logs;
  * full stdout/stderr tee log in each run directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from quadrotor_transfer.callbacks import ActionStdCallback, RewardOrientedBestModelCallback
from quadrotor_transfer.envs import (
    FixedInitialStateWrapper,
    ReducedQuadrotor2DNavigationEnv,
    ReducedQuadrotorConfig,
    sample_initial_states,
)
from quadrotor_transfer.utils import (
    as_1d_float_array,
    describe_action_std,
    find_policy_artifacts,
    lambda_to_name,
    log_std_to_std,
    make_run_id,
    now_timestamp,
    setup_tee_logging,
    std_to_log_std,
    update_run_index,
    write_json,
)


def build_config(args: argparse.Namespace) -> ReducedQuadrotorConfig:
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
        lambda_penalty=args.lambda_penalty,
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


def make_env_fn(rank: int, seed: int, cfg: ReducedQuadrotorConfig) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = ReducedQuadrotor2DNavigationEnv(config=cfg)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def make_eval_env_fn(seed: int, cfg: ReducedQuadrotorConfig, n_eval_episodes: int) -> Callable[[], gym.Env]:
    init_states = sample_initial_states(
        n_eval_episodes,
        seed=seed,
        x_low=cfg.init_x_low,
        x_high=cfg.init_x_high,
        z_low=cfg.init_z_low,
        z_high=cfg.init_z_high,
        x_dot_low=cfg.init_x_dot_low,
        x_dot_high=cfg.init_x_dot_high,
        z_dot_low=cfg.init_z_dot_low,
        z_dot_high=cfg.init_z_dot_high,
    )

    def _init() -> gym.Env:
        env = ReducedQuadrotor2DNavigationEnv(config=cfg)
        env = FixedInitialStateWrapper(env, init_states)
        return Monitor(env)
    return _init


def parse_std_values(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (log_std_init, log_std_final, std_init, std_final), each length 2."""
    if args.log_std_init is not None:
        log_std_init = as_1d_float_array(args.log_std_init, length=2, name="log_std_init")
        std_init = log_std_to_std(log_std_init, length=2, name="log_std_init")
    else:
        std_init = as_1d_float_array(args.std_init_normalized, length=2, name="std_init_normalized")
        log_std_init = std_to_log_std(std_init, length=2, name="std_init_normalized")

    if args.log_std_final is not None:
        log_std_final = as_1d_float_array(args.log_std_final, length=2, name="log_std_final")
        std_final = log_std_to_std(log_std_final, length=2, name="log_std_final")
    else:
        std_final = as_1d_float_array(args.std_final_normalized, length=2, name="std_final_normalized")
        log_std_final = std_to_log_std(std_final, length=2, name="std_final_normalized")

    return log_std_init, log_std_final, std_init, std_final


def resolve_warm_start(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if args.warm_start == "none":
        return None, None, None
    if args.warm_start == "path":
        if args.warm_start_model_path is None:
            raise ValueError("--warm_start path requires --warm_start_model_path")
        model_path, vec_path, cfg_path = find_policy_artifacts(
            args.model_root,
            args.warm_start_lambda if args.warm_start_lambda is not None else args.lambda_penalty,
            explicit_model=args.warm_start_model_path,
            explicit_vecnormalize=args.warm_start_vecnormalize_path,
            selector="best",
        )
        return model_path, vec_path, cfg_path

    warm_lambda = args.lambda_penalty if args.warm_start_lambda is None else args.warm_start_lambda
    selector = "latest" if args.warm_start == "latest" else args.warm_start
    model_path, vec_path, cfg_path = find_policy_artifacts(
        args.model_root,
        warm_lambda,
        explicit_vecnormalize=args.warm_start_vecnormalize_path,
        selector=selector,
    )
    return model_path, vec_path, cfg_path


def make_train_env(args: argparse.Namespace, cfg: ReducedQuadrotorConfig, warm_vec_path: Optional[Path]):
    raw_env = DummyVecEnv([make_env_fn(rank, args.seed, cfg) for rank in range(args.n_envs)])
    raw_env = VecMonitor(raw_env)
    if args.normalize_obs:
        if warm_vec_path is not None and warm_vec_path.exists():
            env = VecNormalize.load(str(warm_vec_path), venv=raw_env)
            env.training = True
            env.norm_reward = False
            print(f"Loaded warm-start VecNormalize statistics from: {warm_vec_path}")
            return env
        return VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=args.clip_obs)
    return raw_env


def make_new_model(args: argparse.Namespace, env, log_std_init: np.ndarray) -> PPO:
    policy_kwargs = {
        "activation_fn": th.nn.Tanh,
        "net_arch": {"pi": args.policy_hidden_sizes, "vf": args.value_hidden_sizes},
        # SB3 expects a scalar here; the callback sets per-dimension values at training start.
        "log_std_init": float(log_std_init[0]),
    }
    return PPO(
        "MlpPolicy",
        env,
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
        verbose=args.sb3_verbose,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(args.tensorboard_root) if args.tensorboard_root is not None else None,
    )


def load_or_create_model(args: argparse.Namespace, env, log_std_init: np.ndarray, warm_model_path: Optional[Path]) -> PPO:
    if warm_model_path is None:
        return make_new_model(args, env, log_std_init)

    if args.warm_start_optimizer == "reuse":
        print(f"Warm-starting PPO with optimizer state from: {warm_model_path}")
        custom_objects = {
            "learning_rate": args.learning_rate,
            "lr_schedule": lambda _: args.learning_rate,
            "clip_range": lambda _: args.clip_range,
        }
        return PPO.load(str(warm_model_path), env=env, device=args.device, custom_objects=custom_objects)

    print(f"Warm-starting policy weights only from: {warm_model_path}")
    warm_model = PPO.load(str(warm_model_path), device=args.device)
    model = make_new_model(args, env, log_std_init)
    model.policy.load_state_dict(warm_model.policy.state_dict(), strict=True)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reduced-order quadrotor transfer policy.")
    parser.add_argument("--lambda_penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # Dynamics/task.
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Reduced model step time. Use 0.02 to reuse your current fork settings.")
    parser.add_argument("--max_steps", type=int, default=300, help="Fixed horizon. 300 at 60 Hz is 5 seconds.")
    parser.add_argument("--mass", type=float, default=0.027)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--x_goal", type=float, default=0.0)
    parser.add_argument("--z_goal", type=float, default=1.0)
    parser.add_argument("--q_x", type=float, default=1.0)
    parser.add_argument("--q_z", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--target_radius", type=float, default=0.05)
    parser.add_argument("--target_bonus", type=float, default=2.0)
    parser.add_argument("--terminate_on_goal", action="store_true", help="Default off: keep fixed horizon even after target is reached.")
    parser.add_argument("--theta_max_deg", type=float, default=22.5, help="Default pi/8, matching your previous project.")
    parser.add_argument("--delta_thrust_max", type=float, default=None, help="Default None: conservative symmetric limit derived from safe-control-gym CF2X thrust bounds.")
    parser.add_argument("--x_bound", type=float, default=2.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--no_out_of_bounds_termination", action="store_true")

    # Initial condition distribution.
    parser.add_argument("--init_x_low", type=float, default=-0.5)
    parser.add_argument("--init_x_high", type=float, default=0.5)
    parser.add_argument("--init_z_low", type=float, default=0.4)
    parser.add_argument("--init_z_high", type=float, default=1.5)
    parser.add_argument("--init_x_dot_low", type=float, default=-0.01)
    parser.add_argument("--init_x_dot_high", type=float, default=0.01)
    parser.add_argument("--init_z_dot_low", type=float, default=-0.01)
    parser.add_argument("--init_z_dot_high", type=float, default=0.01)

    # PPO.
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sb3_verbose", type=int, default=1)
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--normalize_obs", action="store_true")
    parser.add_argument("--clip_obs", type=float, default=10.0)
    parser.add_argument("--policy_hidden_sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--value_hidden_sizes", type=int, nargs="+", default=[64, 64])

    # PPO std/exploration.
    parser.add_argument("--std_mode", type=str, default="learnable", choices=["learnable", "trainable", "fixed", "anneal"])
    parser.add_argument("--std_init_normalized", type=float, nargs="+", default=[0.35], help="One value or two values [delta_thrust_std, theta_std] in normalized action units.")
    parser.add_argument("--std_final_normalized", type=float, nargs="+", default=[0.08], help="Used only for --std_mode anneal; one value or two values.")
    parser.add_argument("--log_std_init", type=float, nargs="+", default=None, help="Advanced override for log std init; one value or two values.")
    parser.add_argument("--log_std_final", type=float, nargs="+", default=None, help="Advanced override for anneal log std final; one value or two values.")

    # Best model/checkpoints.
    parser.add_argument("--eval_freq", type=int, default=25_000, help="Held-out reduced-env eval frequency in environment timesteps. <=0 disables best-model eval.")
    parser.add_argument("--best_n_eval_episodes", type=int, default=10)
    parser.add_argument("--best_eval_seed", type=int, default=2026)
    parser.add_argument("--best_metric", type=str, default="task_return_mean", choices=["task_return_mean", "discounted_task_return_mean", "success_rate", "train_return_mean"])
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)

    # Warm start.
    parser.add_argument("--warm_start", type=str, default="none", choices=["none", "latest", "best", "final", "path"])
    parser.add_argument("--warm_start_lambda", type=float, default=None, help="Default: current lambda. Useful for lambda continuation/curriculum.")
    parser.add_argument("--warm_start_model_path", type=Path, default=None)
    parser.add_argument("--warm_start_vecnormalize_path", type=Path, default=None)
    parser.add_argument("--warm_start_optimizer", type=str, default="reset", choices=["reset", "reuse"], help="reset copies policy weights into a fresh PPO optimizer; reuse loads PPO optimizer state too.")
    parser.add_argument("--continue_num_timesteps_from_warm_start", action="store_true")
    parser.add_argument("--reset_std_on_warm_start", action="store_true", help="For learnable std warm starts, reset log_std to --std_init_normalized instead of preserving the loaded std.")

    # Output.
    parser.add_argument("--model_root", type=Path, default=Path("models") / "transfer_quadrotor_reduced_policies")
    parser.add_argument("--results_root", type=Path, default=Path("results") / "transfer_quadrotor_reduced_training")
    parser.add_argument("--tensorboard_root", type=Path, default=Path("runs") / "transfer_quadrotor_tensorboard")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--no_log_to_file", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.std_mode == "trainable":
        args.std_mode = "learnable"
    cfg = build_config(args)
    log_std_init, log_std_final, std_init, std_final = parse_std_values(args)

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    timestamp = now_timestamp()
    run_id = make_run_id(
        lambda_penalty=args.lambda_penalty,
        seed=args.seed,
        std_mode=args.std_mode,
        timestamp=timestamp,
        extra=args.run_tag,
    )
    model_run_dir = args.model_root / "runs" / run_id
    result_run_dir = args.results_root / "runs" / run_id
    model_run_dir.mkdir(parents=True, exist_ok=True)
    result_run_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if not args.no_log_to_file:
        log_file = setup_tee_logging(result_run_dir / "training_full.log")

    warm_model_path, warm_vec_path, warm_config_path = resolve_warm_start(args)
    env = make_train_env(args, cfg, warm_vec_path if args.normalize_obs else None)
    model = load_or_create_model(args, env, log_std_init, warm_model_path)

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "lambda_name": lambda_to_name(args.lambda_penalty),
            "model_run_dir": str(model_run_dir),
            "result_run_dir": str(result_run_dir),
            "theta_max_rad": float(cfg.theta_max),
            "config": cfg.to_dict(),
            "action_limit_summary": cfg.action_limit_summary(),
            "std_init_normalized_resolved": std_init.tolist(),
            "std_final_normalized_resolved": std_final.tolist(),
            "log_std_init_resolved": log_std_init.tolist(),
            "log_std_final_resolved": log_std_final.tolist(),
            "std_init_physical": describe_action_std(std_init, cfg.theta_max, cfg.delta_thrust_max),
            "std_final_physical": describe_action_std(std_final, cfg.theta_max, cfg.delta_thrust_max),
            "warm_start_model_resolved": str(warm_model_path) if warm_model_path else None,
            "warm_start_vecnormalize_resolved": str(warm_vec_path) if warm_vec_path else None,
            "warm_start_config_resolved": str(warm_config_path) if warm_config_path else None,
        }
    )
    write_json(result_run_dir / "config.json", config_payload)
    write_json(model_run_dir / "config.json", config_payload)
    write_json(result_run_dir / "action_limits.json", cfg.action_limit_summary())
    write_json(result_run_dir / "std_analysis.json", {
        "std_init": describe_action_std(std_init, cfg.theta_max, cfg.delta_thrust_max),
        "std_final": describe_action_std(std_final, cfg.theta_max, cfg.delta_thrust_max),
    })

    set_initial_std = not (warm_model_path is not None and args.std_mode == "learnable" and not args.reset_std_on_warm_start)
    if warm_model_path is not None and args.std_mode == "learnable" and not args.reset_std_on_warm_start:
        print("Preserving learned log_std from warm-start model. Use --reset_std_on_warm_start to override.")
    callbacks = [
        ActionStdCallback(
            args.std_mode,
            log_std_init,
            log_std_final,
            args.total_timesteps,
            verbose=1,
            set_initial_on_training_start=set_initial_std,
        ),
    ]
    if args.eval_freq > 0:
        callbacks.append(
            RewardOrientedBestModelCallback(
                make_eval_env_fn(args.best_eval_seed, cfg, args.best_n_eval_episodes),
                model_run_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.best_n_eval_episodes,
                max_steps=args.max_steps,
                deterministic=True,
                metric_name=args.best_metric,
                verbose=1,
            )
        )
    if args.checkpoint_freq > 0:
        checkpoint_kwargs = dict(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=str(model_run_dir / "checkpoints"),
            name_prefix="ppo_reduced_quadrotor",
            save_replay_buffer=False,
        )
        try:
            callbacks.append(CheckpointCallback(**checkpoint_kwargs, save_vecnormalize=True))
        except TypeError:
            callbacks.append(CheckpointCallback(**checkpoint_kwargs))
            print("CheckpointCallback in this SB3 version does not support save_vecnormalize=True; final/best VecNormalize files are still saved explicitly.")

    print("\n===================================================")
    print("Training reduced-order 2D quadrotor policy")
    print(f"run_id         = {run_id}")
    print(f"lambda_penalty = {args.lambda_penalty:g}")
    print("obs            = [x, x_dot, z, z_dot, theta_star_prev]")
    print("action         = normalized [Delta thrust, theta_star]")
    print("reward train   = task_reward - lambda*|Delta theta_star|")
    print("best metric    = held-out unpenalized task return unless changed")
    print(f"goal           = ({args.x_goal:g}, {args.z_goal:g})")
    print(f"fixed horizon  = {args.max_steps} steps, terminate_on_goal={args.terminate_on_goal}")
    print(f"theta limit    = +/- {np.rad2deg(cfg.theta_max):.4f} deg ({cfg.theta_max:.6f} rad)")
    print(f"delta limit    = +/- {cfg.delta_thrust_max:.6f} N")
    print(f"std mode       = {args.std_mode}")
    print(f"std init norm  = {std_init}")
    print(f"std init phys  = {describe_action_std(std_init, cfg.theta_max, cfg.delta_thrust_max)}")
    if args.std_mode == "anneal":
        print(f"std final norm = {std_final}")
        print(f"std final phys = {describe_action_std(std_final, cfg.theta_max, cfg.delta_thrust_max)}")
    if warm_model_path is not None:
        print(f"warm start     = {warm_model_path}")
    print(f"model_run_dir  = {model_run_dir}")
    print(f"result_run_dir = {result_run_dir}")
    print("===================================================\n")

    reset_num_timesteps = not bool(warm_model_path and args.continue_num_timesteps_from_warm_start)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=run_id,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=args.progress_bar,
    )

    final_model_path = model_run_dir / "final_model.zip"
    model.save(final_model_path)
    final_vec_path = None
    if isinstance(env, VecNormalize):
        final_vec_path = model_run_dir / "vecnormalize.pkl"
        env.save(str(final_vec_path))

    best_model_path = model_run_dir / "best_model.zip"
    best_vec_path = model_run_dir / "best_vecnormalize.pkl" if (model_run_dir / "best_vecnormalize.pkl").exists() else None
    best_metrics = {}
    if (model_run_dir / "best_metrics.json").exists():
        with open(model_run_dir / "best_metrics.json", "r", encoding="utf-8") as f:
            best_metrics = json.load(f)
    if not best_model_path.exists():
        # If evaluation was disabled, still provide a best_model.zip alias so downstream scripts work.
        shutil.copy2(final_model_path, best_model_path)
        if final_vec_path is not None:
            best_vec_path = model_run_dir / "best_vecnormalize.pkl"
            shutil.copy2(final_vec_path, best_vec_path)
        best_metrics = {"selected_metric_name": args.best_metric, "selected_metric_value": None, "note": "Evaluation disabled or no eval fired; best_model.zip is a copy of final_model.zip."}
        write_json(model_run_dir / "best_metrics.json", best_metrics)

    completion = {
        "run_id": run_id,
        "lambda_penalty": float(args.lambda_penalty),
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "final_vecnormalize_path": str(final_vec_path) if final_vec_path else None,
        "best_vecnormalize_path": str(best_vec_path) if best_vec_path else None,
        "best_metric_name": str(best_metrics.get("selected_metric_name", args.best_metric)),
        "best_metric_value": best_metrics.get("selected_metric_value"),
        "model_run_dir": str(model_run_dir),
        "result_run_dir": str(result_run_dir),
    }
    write_json(model_run_dir / "run_complete.json", completion)
    write_json(result_run_dir / "run_complete.json", completion)
    index_record = update_run_index(
        model_root=args.model_root,
        lambda_penalty=args.lambda_penalty,
        run_id=run_id,
        run_dir=model_run_dir,
        final_model_path=final_model_path,
        best_model_path=best_model_path,
        final_vecnormalize_path=final_vec_path,
        best_vecnormalize_path=best_vec_path,
        config_path=model_run_dir / "config.json",
        best_metric=best_metrics.get("selected_metric_value"),
        best_metric_name=str(best_metrics.get("selected_metric_name", args.best_metric)),
    )
    write_json(result_run_dir / "latest_index_record.json", index_record)

    print("\n===================================================")
    print("Training complete")
    print(f"Saved reward-oriented best model: {best_model_path}")
    print(f"Saved final model:                {final_model_path}")
    if final_vec_path is not None:
        print(f"Saved final VecNormalize:         {final_vec_path}")
    if best_vec_path is not None:
        print(f"Saved best VecNormalize:          {best_vec_path}")
    print(f"By-lambda latest index:           {args.model_root / 'by_lambda' / lambda_to_name(args.lambda_penalty) / 'latest.json'}")
    print(f"Full training log:                {result_run_dir / 'training_full.log'}")
    print("No previous run directory was deleted.")
    print("===================================================\n")

    env.close()
    if log_file is not None:
        log_file.flush()


if __name__ == "__main__":
    main()
