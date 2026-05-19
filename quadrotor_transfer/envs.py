# -*- coding: utf-8 -*-
"""Reduced and full-deployment environments for quadrotor transfer.

This file intentionally lives outside ``safe_control_gym``.  The reduced model is
used for training.  The deployment adapter wraps the official safe-control-gym
2D quadrotor and exposes the same reduced observation/action interface to the
policy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Default Crazyflie 2.x constants used by safe-control-gym's cf2x.urdf.  These
# are used only to choose conservative reduced-model action limits when the full
# PyBullet environment has not yet been constructed.  The deployment adapter also
# clips using the actual ``full_env.physical_action_bounds`` at runtime.
SCG_CF2X_MASS = 0.027
SCG_GRAVITY = 9.81
SCG_CF2X_KF = 3.16e-10
SCG_CF2X_PWM2RPM_SCALE = 0.2685
SCG_CF2X_PWM2RPM_CONST = 4070.3
SCG_CF2X_MIN_PWM = 20000.0
SCG_CF2X_MAX_PWM = 65535.0


def safe_control_gym_cf2x_pair_thrust_bounds(action_dim: int = 2) -> tuple[float, float]:
    """Return safe-control-gym's default direct-thrust bounds per 2D action.

    For the 2D quadrotor, each action is one motor-pair thrust command.  The
    upstream code computes:
        KF * n_mot * (PWM2RPM_SCALE * PWM + PWM2RPM_CONST)^2
    where n_mot = 4 / action_dim.
    """
    n_mot = 4.0 / float(action_dim)
    low = SCG_CF2X_KF * n_mot * (SCG_CF2X_PWM2RPM_SCALE * SCG_CF2X_MIN_PWM + SCG_CF2X_PWM2RPM_CONST) ** 2
    high = SCG_CF2X_KF * n_mot * (SCG_CF2X_PWM2RPM_SCALE * SCG_CF2X_MAX_PWM + SCG_CF2X_PWM2RPM_CONST) ** 2
    return float(low), float(high)


def default_symmetric_delta_thrust_limit(mass: float = SCG_CF2X_MASS, gravity: float = SCG_GRAVITY) -> float:
    """Conservative symmetric total-thrust offset from upstream physical bounds.

    In the 2D safe-control-gym action space, actions are [T1, T2].  Ignoring
    attitude torque for the moment, each pair receives (m*g + delta_thrust)/2.
    This function returns the largest symmetric |delta_thrust| that keeps both
    pair thrusts inside the upstream physical bounds around hover.
    """
    low, high = safe_control_gym_cf2x_pair_thrust_bounds(action_dim=2)
    hover_total = float(mass) * float(gravity)
    positive = 2.0 * high - hover_total
    negative = hover_total - 2.0 * low
    return float(max(1e-12, min(positive, negative)))


def action_limit_summary(theta_max: float, delta_thrust_max: float, mass: float, gravity: float) -> dict[str, float]:
    """Summarize policy and upstream action limits for logging/debugging."""
    pair_low, pair_high = safe_control_gym_cf2x_pair_thrust_bounds(action_dim=2)
    hover_total = float(mass) * float(gravity)
    return {
        "theta_max_rad": float(theta_max),
        "theta_max_deg": float(math.degrees(theta_max)),
        "delta_thrust_max_N": float(delta_thrust_max),
        "hover_total_thrust_N": float(hover_total),
        "safe_control_gym_pair_thrust_low_N": float(pair_low),
        "safe_control_gym_pair_thrust_high_N": float(pair_high),
        "safe_control_gym_hover_pair_thrust_N": float(hover_total / 2.0),
        "safe_control_gym_symmetric_delta_thrust_limit_N": float(default_symmetric_delta_thrust_limit(mass, gravity)),
    }


@dataclass
class ReducedQuadrotorConfig:
    """Common task and model configuration for reduced/full transfer."""

    # Physical parameters.  The defaults match Crazyflie-like values used by
    # safe-control-gym's quadrotor.
    mass: float = 0.027
    gravity: float = 9.81
    dt: float = 1.0 / 60.0
    max_steps: int = 300

    # Navigation target in the x-z plane.
    x_goal: float = 0.0
    z_goal: float = 1.0
    q_x: float = 1.0
    q_z: float = 1.0
    reward_scale: float = 1.0
    target_radius: float = 0.05
    target_bonus: float = 2.0
    terminate_on_goal: bool = False

    # Policy action scaling.  The policy action is normalized in [-1, 1]^2.
    theta_max: float = math.pi / 8.0
    delta_thrust_max: Optional[float] = None

    # Lambda penalty used for training only.  Evaluation should use info["task_reward"].
    lambda_penalty: float = 0.0

    # State bounds.  These are intentionally close to safe-control-gym 2D bounds:
    # x in [-2, 2], z in [0, 2].
    x_bound: float = 2.0
    z_min: float = 0.0
    z_max: float = 2.0
    x_dot_bound: float = 30.0
    z_dot_bound: float = 30.0
    terminate_on_out_of_bounds: bool = True

    # Initial condition distribution for training/evaluation sampling.
    init_x_low: float = -0.5
    init_x_high: float = 0.5
    init_z_low: float = 0.4
    init_z_high: float = 1.5
    init_x_dot_low: float = -0.01
    init_x_dot_high: float = 0.01
    init_z_dot_low: float = -0.01
    init_z_dot_high: float = 0.01

    use_semi_implicit_euler: bool = True

    def __post_init__(self) -> None:
        # A theta command of +/- pi/8 exactly matches your previous project.
        # It is also well inside safe-control-gym's 2D theta state bound.
        self.theta_max = float(self.theta_max)
        if self.delta_thrust_max is None:
            self.delta_thrust_max = default_symmetric_delta_thrust_limit(self.mass, self.gravity)
        self.delta_thrust_max = float(self.delta_thrust_max)
        if self.theta_max <= 0:
            raise ValueError(f"theta_max must be positive, got {self.theta_max}.")
        if self.delta_thrust_max <= 0:
            raise ValueError(f"delta_thrust_max must be positive, got {self.delta_thrust_max}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def action_limit_summary(self) -> dict[str, float]:
        return action_limit_summary(self.theta_max, self.delta_thrust_max, self.mass, self.gravity)


class ReducedQuadrotor2DNavigationEnv(gym.Env):
    """Reduced 2D quadrotor navigation MDP.

    Observation:
        [x, x_dot, z, z_dot, theta_star_previous]

    Normalized action:
        a[0] -> Delta thrust in [-delta_thrust_max, delta_thrust_max]
        a[1] -> theta_star in [-theta_max, theta_max]

    Returned reward:
        task_reward - lambda_penalty * |theta_star_k - theta_star_{k-1}|

    The unpenalized task reward is always available in ``info["task_reward"]`` and
    ``info["raw_r"]``.  This is the quantity to report for expected cumulative
    reward comparisons across lambdas.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[ReducedQuadrotorConfig] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = ReducedQuadrotorConfig(**kwargs)
        elif kwargs:
            cfg = config.to_dict()
            cfg.update(kwargs)
            config = ReducedQuadrotorConfig(**cfg)
        self.cfg = config
        self.np_random: np.random.Generator = np.random.default_rng()

        obs_low = np.array(
            [
                -self.cfg.x_bound,
                -self.cfg.x_dot_bound,
                self.cfg.z_min,
                -self.cfg.z_dot_bound,
                -self.cfg.theta_max,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self.cfg.x_bound,
                self.cfg.x_dot_bound,
                self.cfg.z_max,
                self.cfg.z_dot_bound,
                self.cfg.theta_max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = np.zeros(4, dtype=np.float32)
        self.prev_theta_star = 0.0
        self.step_count = 0

    @property
    def hover_thrust(self) -> float:
        return float(self.cfg.mass * self.cfg.gravity)

    def scale_action(self, action: np.ndarray) -> tuple[float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}.")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_thrust = float(action[0] * self.cfg.delta_thrust_max)
        theta_star = float(action[1] * self.cfg.theta_max)
        return delta_thrust, theta_star

    def _get_obs(self) -> np.ndarray:
        x, x_dot, z, z_dot = self.state
        return np.array([x, x_dot, z, z_dot, self.prev_theta_star], dtype=np.float32)

    def _distance_terms(self, x: float, z: float) -> tuple[float, float]:
        dx = float(x - self.cfg.x_goal)
        dz = float(z - self.cfg.z_goal)
        weighted_sq = float(self.cfg.q_x * dx * dx + self.cfg.q_z * dz * dz)
        euclidean = float(math.sqrt(dx * dx + dz * dz))
        return weighted_sq, euclidean

    def _task_reward(self, x: float, z: float) -> tuple[float, float, bool]:
        weighted_sq, distance = self._distance_terms(x, z)
        reward = -weighted_sq / max(float(self.cfg.reward_scale), 1e-12)
        success = bool(distance <= self.cfg.target_radius)
        if success and self.cfg.target_bonus:
            reward += float(self.cfg.target_bonus)
        return float(reward), float(distance), success

    def _is_out_of_bounds(self) -> bool:
        x, x_dot, z, z_dot = [float(v) for v in self.state]
        return bool(
            abs(x) > self.cfg.x_bound
            or abs(x_dot) > self.cfg.x_dot_bound
            or z < self.cfg.z_min
            or z > self.cfg.z_max
            or abs(z_dot) > self.cfg.z_dot_bound
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        options = options or {}
        if "init_state" in options:
            init_state = np.asarray(options["init_state"], dtype=np.float32).reshape(-1)
            if init_state.size < 4:
                raise ValueError("options['init_state'] must contain at least [x, x_dot, z, z_dot].")
            self.state = init_state[:4].astype(np.float32).copy()
        else:
            self.state = np.array(
                [
                    self.np_random.uniform(self.cfg.init_x_low, self.cfg.init_x_high),
                    self.np_random.uniform(self.cfg.init_x_dot_low, self.cfg.init_x_dot_high),
                    self.np_random.uniform(self.cfg.init_z_low, self.cfg.init_z_high),
                    self.np_random.uniform(self.cfg.init_z_dot_low, self.cfg.init_z_dot_high),
                ],
                dtype=np.float32,
            )

        self.prev_theta_star = float(options.get("prev_theta_star", 0.0))
        self.step_count = 0
        task_reward, distance, success = self._task_reward(float(self.state[0]), float(self.state[2]))
        info = {
            "task_reward": task_reward,
            "raw_r": task_reward,
            "distance_to_target": distance,
            "success": success,
            "theta_star": self.prev_theta_star,
            "prev_theta_star": self.prev_theta_star,
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        delta_thrust, theta_star = self.scale_action(action)
        x, x_dot, z, z_dot = [float(v) for v in self.state]

        total_thrust = max(0.0, self.hover_thrust + delta_thrust)
        x_ddot = (total_thrust / self.cfg.mass) * math.sin(theta_star)
        z_ddot = (total_thrust / self.cfg.mass) * math.cos(theta_star) - self.cfg.gravity

        if self.cfg.use_semi_implicit_euler:
            x_dot_next = x_dot + x_ddot * self.cfg.dt
            z_dot_next = z_dot + z_ddot * self.cfg.dt
            x_next = x + x_dot_next * self.cfg.dt
            z_next = z + z_dot_next * self.cfg.dt
        else:
            x_next = x + x_dot * self.cfg.dt
            z_next = z + z_dot * self.cfg.dt
            x_dot_next = x_dot + x_ddot * self.cfg.dt
            z_dot_next = z_dot + z_ddot * self.cfg.dt

        variation_cost = abs(theta_star - self.prev_theta_star)
        task_reward, distance, success = self._task_reward(x_next, z_next)
        train_reward = task_reward - float(self.cfg.lambda_penalty) * variation_cost

        prev_theta_star_before_step = self.prev_theta_star
        self.state = np.array([x_next, x_dot_next, z_next, z_dot_next], dtype=np.float32)
        self.prev_theta_star = theta_star
        self.step_count += 1

        out_of_bounds = self._is_out_of_bounds()
        terminated = bool(
            (self.cfg.terminate_on_goal and success)
            or (self.cfg.terminate_on_out_of_bounds and out_of_bounds)
        )
        truncated = bool(self.step_count >= int(self.cfg.max_steps))

        info = {
            "task_reward": float(task_reward),
            "raw_r": float(task_reward),
            "train_reward": float(train_reward),
            "distance_to_target": float(distance),
            "success": bool(success),
            "out_of_bounds": bool(out_of_bounds),
            "variation_cost": float(variation_cost),
            "theta_variation": float(variation_cost),
            "delta_thrust": float(delta_thrust),
            "total_thrust": float(total_thrust),
            "theta_star": float(theta_star),
            "normalized_delta_thrust_action": float(np.asarray(action).reshape(-1)[0]),
            "normalized_theta_star_action": float(np.asarray(action).reshape(-1)[1]),
            "delta_thrust_max": float(self.cfg.delta_thrust_max),
            "theta_max": float(self.cfg.theta_max),
            "prev_theta_star_before_step": float(prev_theta_star_before_step),
            "x": float(self.state[0]),
            "x_dot": float(self.state[1]),
            "z": float(self.state[2]),
            "z_dot": float(self.state[3]),
            "step_count": int(self.step_count),
        }
        return self._get_obs(), float(train_reward), terminated, truncated, info


class FixedInitialStateWrapper(gym.Wrapper):
    """Cycle through fixed initial states on reset.

    Each initial state should be at least [x, x_dot, z, z_dot].  A full 6D state
    [x, x_dot, z, z_dot, theta, theta_dot] may also be supplied for deployment.
    """

    def __init__(self, env: gym.Env, init_states: Sequence[Sequence[float]]) -> None:
        super().__init__(env)
        arr = np.asarray(init_states, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"init_states must have shape (N,4) or (N,>=4), got {arr.shape}")
        self.init_states = arr.copy()
        self._idx = 0

    def reset(self, **kwargs: Any):
        options = dict(kwargs.pop("options", {}) or {})
        options["init_state"] = self.init_states[self._idx % len(self.init_states)].copy()
        self._idx += 1
        return self.env.reset(options=options, **kwargs)


def sample_initial_states(
    n: int,
    seed: int = 0,
    x_low: float = -0.5,
    x_high: float = 0.5,
    z_low: float = 0.4,
    z_high: float = 1.5,
    x_dot_low: float = -0.01,
    x_dot_high: float = 0.01,
    z_dot_low: float = -0.01,
    z_dot_high: float = 0.01,
) -> np.ndarray:
    """Sample fixed reduced initial states [x, x_dot, z, z_dot]."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(x_low, x_high, size=n),
            rng.uniform(x_dot_low, x_dot_high, size=n),
            rng.uniform(z_low, z_high, size=n),
            rng.uniform(z_dot_low, z_dot_high, size=n),
        ]
    ).astype(np.float32)


class SafeControlGymQuad2DDeploymentEnv(gym.Env):
    """Deployment adapter around safe-control-gym's full 2D quadrotor.

    The policy sees the same interface as the reduced model:

        obs = [x, x_dot, z, z_dot, theta_star_previous]
        action = normalized [Delta thrust, theta_star]

    Internally, the adapter maps ``theta_star`` into safe-control-gym motor-pair
    thrusts [T1, T2] using a PD attitude-tracking controller with a filtered
    theta_star derivative estimate.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[ReducedQuadrotorConfig] = None,
        *,
        omega_n: float = 8.0,
        zeta: float = 0.7,
        theta_star_tau: float = 0.10,
        ctrl_freq: int = 60,
        pyb_freq: int = 240,
        gui: bool = False,
        verbose: bool = False,
        use_default_constraints: bool = False,
        done_on_violation: bool = False,
        done_on_out_of_bound: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = ReducedQuadrotorConfig(**kwargs)
        elif kwargs:
            cfg = config.to_dict()
            cfg.update(kwargs)
            config = ReducedQuadrotorConfig(**cfg)
        self.cfg = config
        self.omega_n = float(omega_n)
        self.zeta = float(zeta)
        self.theta_star_tau = float(theta_star_tau)
        self.ctrl_freq = int(ctrl_freq)
        self.pyb_freq = int(pyb_freq)
        self.gui = bool(gui)
        self.verbose = bool(verbose)
        self.use_default_constraints = bool(use_default_constraints)
        self.done_on_violation = bool(done_on_violation)
        self.done_on_out_of_bound = bool(done_on_out_of_bound)

        self.observation_space = spaces.Box(
            low=np.array(
                [-self.cfg.x_bound, -self.cfg.x_dot_bound, self.cfg.z_min, -self.cfg.z_dot_bound, -self.cfg.theta_max],
                dtype=np.float32,
            ),
            high=np.array(
                [self.cfg.x_bound, self.cfg.x_dot_bound, self.cfg.z_max, self.cfg.z_dot_bound, self.cfg.theta_max],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        self.full_env = self._make_full_env()
        self.prev_theta_star = 0.0
        self.theta_star_dot_est = 0.0
        self.step_count = 0
        self.last_safe_control_gym_reward = 0.0
        self.last_full_obs = None

    def _make_constraints(self):
        if not self.use_default_constraints:
            return None
        return [
            {"constraint_form": "default_constraint", "constrained_variable": "state"},
            {"constraint_form": "default_constraint", "constrained_variable": "input"},
        ]

    def _make_full_env(self):
        try:
            from safe_control_gym.envs.benchmark_env import Cost, Task
            from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
            from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
        except Exception as exc:  # pragma: no cover - dependency/runtime setup issue.
            raise ImportError(
                "Could not import safe-control-gym. Run from the safe-control-gym repo root "
                "after `python -m pip install -e .`."
            ) from exc

        # Make the safe-control-gym time limit at least as long as our exact fixed horizon.
        episode_len_sec = max(int(math.ceil(int(self.cfg.max_steps) / float(self.ctrl_freq))), 1)
        task_info = {
            "stabilization_goal": [float(self.cfg.x_goal), float(self.cfg.z_goal)],
            # Zero tolerance prevents the upstream stabilization task from terminating at the goal.
            # We handle target success ourselves through info["success"].
            "stabilization_goal_tolerance": 0.0,
        }
        return Quadrotor(
            quad_type=QuadType.TWO_D,
            gui=self.gui,
            verbose=self.verbose,
            normalized_rl_action_space=False,
            task=Task.STABILIZATION,
            task_info=task_info,
            cost=Cost.RL_REWARD,
            ctrl_freq=self.ctrl_freq,
            pyb_freq=self.pyb_freq,
            episode_len_sec=episode_len_sec,
            init_state=np.zeros(6, dtype=np.float32),
            randomized_init=False,
            constraints=self._make_constraints(),
            done_on_violation=self.done_on_violation,
            use_constraint_penalty=False,
            done_on_out_of_bound=self.done_on_out_of_bound,
            obs_goal_horizon=0,
            rew_exponential=False,
        )

    @property
    def dt(self) -> float:
        return 1.0 / float(self.ctrl_freq)

    def _current_full_state(self) -> np.ndarray:
        state = getattr(self.full_env, "state", None)
        if state is None:
            raise RuntimeError("Full safe-control-gym env has no state yet; call reset first.")
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def _get_obs_from_full_state(self) -> np.ndarray:
        state = self._current_full_state()
        return np.array([state[0], state[1], state[2], state[3], self.prev_theta_star], dtype=np.float32)

    def scale_action(self, action: np.ndarray) -> tuple[float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}.")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return float(action[0] * self.cfg.delta_thrust_max), float(action[1] * self.cfg.theta_max)

    def _set_full_initial_state(self, init_state: np.ndarray) -> None:
        init_state = np.asarray(init_state, dtype=np.float32).reshape(-1)
        if init_state.size < 4:
            raise ValueError("init_state must contain at least [x, x_dot, z, z_dot].")
        full = np.zeros(6, dtype=np.float32)
        full[: min(init_state.size, 6)] = init_state[: min(init_state.size, 6)]
        labels = ["INIT_X", "INIT_X_DOT", "INIT_Z", "INIT_Z_DOT", "INIT_THETA", "INIT_THETA_DOT"]
        for label, value in zip(labels, full):
            setattr(self.full_env, label, float(value))

    def _reset_full_env(self, seed: Optional[int] = None):
        # safe-control-gym versions in this family return (obs, info), but this
        # accepts old-style reset returns too.
        out = self.full_env.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "init_state" in options:
            self._set_full_initial_state(np.asarray(options["init_state"], dtype=np.float32))
        else:
            # Keep training/evaluation distribution aligned with the reduced model.
            rng = np.random.default_rng(seed)
            init = np.array(
                [
                    rng.uniform(self.cfg.init_x_low, self.cfg.init_x_high),
                    rng.uniform(self.cfg.init_x_dot_low, self.cfg.init_x_dot_high),
                    rng.uniform(self.cfg.init_z_low, self.cfg.init_z_high),
                    rng.uniform(self.cfg.init_z_dot_low, self.cfg.init_z_dot_high),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            self._set_full_initial_state(init)

        self.prev_theta_star = float(options.get("prev_theta_star", 0.0))
        self.theta_star_dot_est = 0.0
        self.step_count = 0
        full_obs, full_info = self._reset_full_env(seed=seed)
        self.last_full_obs = full_obs

        state = self._current_full_state()
        task_reward, distance, success = self._task_reward(float(state[0]), float(state[2]))
        info = dict(full_info)
        info.update(
            {
                "task_reward": task_reward,
                "raw_r": task_reward,
                "distance_to_target": distance,
                "success": success,
                "theta_star": self.prev_theta_star,
                "theta_star_dot_est": self.theta_star_dot_est,
                **self.cfg.action_limit_summary(),
            }
        )
        return self._get_obs_from_full_state(), info

    def _estimate_theta_star_dot(self, theta_star: float) -> float:
        if self.step_count == 0:
            raw = 0.0
        else:
            raw = (theta_star - self.prev_theta_star) / max(self.dt, 1e-12)
        alpha = math.exp(-self.dt / self.theta_star_tau) if self.theta_star_tau > 0 else 0.0
        self.theta_star_dot_est = alpha * self.theta_star_dot_est + (1.0 - alpha) * raw
        return float(self.theta_star_dot_est)

    def _make_motor_pair_thrusts(self, delta_thrust: float, theta_star: float) -> tuple[np.ndarray, Dict[str, float]]:
        state = self._current_full_state()
        theta = float(state[4])
        theta_dot = float(state[5])
        theta_star_dot = self._estimate_theta_star_dot(theta_star)

        mass = float(getattr(self.full_env, "OVERRIDDEN_QUAD_MASS", getattr(self.full_env, "MASS", self.cfg.mass)))
        gravity = float(getattr(self.full_env, "GRAVITY_ACC", self.cfg.gravity))
        iyy = float(getattr(self.full_env, "J", np.diag([0.0, 1.4e-5, 0.0]))[1, 1])
        arm_length = float(getattr(self.full_env, "L", 0.0397))

        kp = iyy * self.omega_n * self.omega_n
        kd = 2.0 * iyy * self.zeta * self.omega_n
        theta_ddot_des = (kp * (theta_star - theta) + kd * (theta_star_dot - theta_dot)) / max(iyy, 1e-12)

        total_thrust = max(0.0, mass * gravity + float(delta_thrust))
        thrust_difference = theta_ddot_des * iyy * math.sqrt(2.0) / max(arm_length, 1e-12)
        t1 = 0.5 * (total_thrust - thrust_difference)
        t2 = 0.5 * (total_thrust + thrust_difference)
        motor_pair_thrusts_unclipped = np.array([t1, t2], dtype=np.float32)
        motor_pair_thrusts = motor_pair_thrusts_unclipped.copy()
        action_low = np.full(2, np.nan, dtype=np.float32)
        action_high = np.full(2, np.nan, dtype=np.float32)
        if hasattr(self.full_env, "physical_action_bounds"):
            low, high = self.full_env.physical_action_bounds
            action_low = np.asarray(low, dtype=np.float32).reshape(-1)[:2]
            action_high = np.asarray(high, dtype=np.float32).reshape(-1)[:2]
            motor_pair_thrusts = np.clip(motor_pair_thrusts_unclipped, action_low, action_high).astype(np.float32)
        clip_abs = np.abs(motor_pair_thrusts - motor_pair_thrusts_unclipped)

        details = {
            "theta_actual": theta,
            "theta_dot": theta_dot,
            "theta_star_dot_est": theta_star_dot,
            "theta_tracking_error": float(theta_star - theta),
            "theta_dot_tracking_error": float(theta_star_dot - theta_dot),
            "attitude_kp": float(kp),
            "attitude_kd": float(kd),
            "theta_ddot_des": float(theta_ddot_des),
            "motor_pair_t1": float(motor_pair_thrusts[0]),
            "motor_pair_t2": float(motor_pair_thrusts[1]),
            "motor_pair_t1_unclipped": float(motor_pair_thrusts_unclipped[0]),
            "motor_pair_t2_unclipped": float(motor_pair_thrusts_unclipped[1]),
            "motor_pair_clipped": bool(np.any(clip_abs > 1e-12)),
            "motor_pair_clip_abs_sum": float(np.sum(clip_abs)),
            "safe_control_gym_action_low_t1": float(action_low[0]),
            "safe_control_gym_action_high_t1": float(action_high[0]),
            "safe_control_gym_action_low_t2": float(action_low[1]),
            "safe_control_gym_action_high_t2": float(action_high[1]),
            "total_thrust_command": float(total_thrust),
            "thrust_difference_command": float(thrust_difference),
        }
        return motor_pair_thrusts, details

    def _task_reward(self, x: float, z: float) -> tuple[float, float, bool]:
        dx = float(x - self.cfg.x_goal)
        dz = float(z - self.cfg.z_goal)
        weighted_sq = float(self.cfg.q_x * dx * dx + self.cfg.q_z * dz * dz)
        distance = float(math.sqrt(dx * dx + dz * dz))
        reward = -weighted_sq / max(float(self.cfg.reward_scale), 1e-12)
        success = bool(distance <= self.cfg.target_radius)
        if success and self.cfg.target_bonus:
            reward += float(self.cfg.target_bonus)
        return float(reward), distance, success

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        delta_thrust, theta_star = self.scale_action(action)
        variation_cost = abs(theta_star - self.prev_theta_star)
        motor_pair_thrusts, details = self._make_motor_pair_thrusts(delta_thrust, theta_star)

        out = self.full_env.step(motor_pair_thrusts)
        if isinstance(out, tuple) and len(out) == 5:
            full_obs, safe_rew, terminated, truncated, full_info = out
        elif isinstance(out, tuple) and len(out) == 4:
            full_obs, safe_rew, done, full_info = out
            full_info = dict(full_info)
            truncated = bool(full_info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)
        else:
            raise RuntimeError("Unexpected safe-control-gym step return format.")

        self.last_safe_control_gym_reward = float(np.asarray(safe_rew).reshape(-1)[0])
        self.last_full_obs = full_obs
        self.prev_theta_star = float(theta_star)
        self.step_count += 1

        state = self._current_full_state()
        task_reward, distance, success = self._task_reward(float(state[0]), float(state[2]))
        train_reward = task_reward - float(self.cfg.lambda_penalty) * variation_cost

        # Exact fixed horizon controlled by this adapter.
        if self.step_count >= int(self.cfg.max_steps):
            truncated = True

        if self.cfg.terminate_on_goal and success:
            terminated = True

        info = dict(full_info)
        info.update(details)
        info.update(
            {
                "safe_control_gym_reward": self.last_safe_control_gym_reward,
                "task_reward": float(task_reward),
                "raw_r": float(task_reward),
                "train_reward": float(train_reward),
                "distance_to_target": float(distance),
                "success": bool(success),
                "variation_cost": float(variation_cost),
                "theta_variation": float(variation_cost),
                "delta_thrust": float(delta_thrust),
                "theta_star": float(theta_star),
                "normalized_delta_thrust_action": float(np.asarray(action).reshape(-1)[0]),
                "normalized_theta_star_action": float(np.asarray(action).reshape(-1)[1]),
                "delta_thrust_max": float(self.cfg.delta_thrust_max),
                "theta_max": float(self.cfg.theta_max),
                "x": float(state[0]),
                "x_dot": float(state[1]),
                "z": float(state[2]),
                "z_dot": float(state[3]),
                "theta": float(state[4]),
                "theta_rate": float(state[5]),
                "omega_n": float(self.omega_n),
                "zeta": float(self.zeta),
                "step_count": int(self.step_count),
            }
        )
        return self._get_obs_from_full_state(), float(train_reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        try:
            self.full_env.close()
        except Exception:
            pass
