# -*- coding: utf-8 -*-
"""Inspect reduced policy limits and PPO std magnitudes in physical units."""

from __future__ import annotations

import argparse
import numpy as np

from quadrotor_transfer.envs import ReducedQuadrotorConfig, safe_control_gym_cf2x_pair_thrust_bounds
from quadrotor_transfer.utils import as_1d_float_array, describe_action_std


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect safe-control-gym-based action limits and normalized std choices.")
    parser.add_argument("--theta_max_deg", type=float, default=22.5)
    parser.add_argument("--delta_thrust_max", type=float, default=None)
    parser.add_argument("--std_normalized", type=float, nargs="+", default=[0.35], help="One value or two values in normalized action units.")
    parser.add_argument("--mass", type=float, default=0.027)
    parser.add_argument("--gravity", type=float, default=9.81)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReducedQuadrotorConfig(
        mass=args.mass,
        gravity=args.gravity,
        theta_max=np.deg2rad(args.theta_max_deg),
        delta_thrust_max=args.delta_thrust_max,
    )
    pair_low, pair_high = safe_control_gym_cf2x_pair_thrust_bounds(action_dim=2)
    std = as_1d_float_array(args.std_normalized, length=2, name="std_normalized")

    print("safe-control-gym 2D direct-thrust action bounds [T1, T2]")
    print(f"  pair thrust low:          {pair_low:.9f} N")
    print(f"  pair thrust high:         {pair_high:.9f} N")
    print(f"  hover total thrust:       {cfg.mass * cfg.gravity:.9f} N")
    print(f"  hover pair thrust:        {cfg.mass * cfg.gravity / 2.0:.9f} N")
    print("\nReduced policy normalized action limits")
    print(f"  action[0] -> delta thrust in +/- {cfg.delta_thrust_max:.9f} N")
    print(f"  action[1] -> theta_star   in +/- {cfg.theta_max:.9f} rad = +/- {np.rad2deg(cfg.theta_max):.4f} deg")
    print("\nPPO std interpretation")
    for key, value in describe_action_std(std, cfg.theta_max, cfg.delta_thrust_max).items():
        print(f"  {key}: {value}")
    print("\nRule of thumb")
    print("  std_normalized=0.35 means about 35% of the normalized action half-range at initialization.")
    print("  With theta_max=pi/8, that is about 7.9 degrees initial theta_star std.")


if __name__ == "__main__":
    main()
