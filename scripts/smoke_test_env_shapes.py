# -*- coding: utf-8 -*-
"""Print reduced-env dimensions and action-limit metadata."""

from __future__ import annotations

import argparse
import numpy as np

from quadrotor_transfer.envs import ReducedQuadrotor2DNavigationEnv, ReducedQuadrotorConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta_max_deg", type=float, default=22.5)
    parser.add_argument("--delta_thrust_max", type=float, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--max_steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReducedQuadrotorConfig(
        theta_max=np.deg2rad(args.theta_max_deg),
        delta_thrust_max=args.delta_thrust_max,
        dt=args.dt,
        max_steps=args.max_steps,
    )
    env = ReducedQuadrotor2DNavigationEnv(config=cfg)
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    print("Reduced quadrotor env smoke test")
    print(f"observation_space: {env.observation_space}")
    print(f"action_space:      {env.action_space}")
    print(f"obs_dim:           {env.observation_space.shape[0]}")
    print(f"action_dim:        {env.action_space.shape[0]}")
    print(f"initial obs:       {obs}")
    print(f"sample action:     {action}")
    print(f"next obs:          {next_obs}")
    print(f"reward:            {reward}")
    print(f"terminated/trunc:  {terminated}/{truncated}")
    print(f"reset info:        {info}")
    print(f"step info keys:    {sorted(step_info.keys())}")
    print("\nAction-limit summary:")
    for key, value in cfg.action_limit_summary().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
