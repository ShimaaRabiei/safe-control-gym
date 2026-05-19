# Reduced-to-full quadrotor transfer overlay for safe-control-gym

This overlay is meant to be copied into the root of a `safe-control-gym` fork.  It does **not** edit files inside `safe_control_gym/`.  The idea is to work like an academic collaborator on an open-source project: keep the upstream benchmark intact, and add your experiment as a clean external layer.

## Scientific problem

Train a reduced-order 2D quadrotor navigation policy, then deploy it zero-shot on the full safe-control-gym 2D quadrotor.

Reduced training model:

```text
state/obs = [x, x_dot, z, z_dot, theta_star_previous]
action    = normalized [delta_thrust, theta_star]
reward    = task_reward - lambda * |theta_star_k - theta_star_{k-1}|
```

Full deployment model:

```text
safe-control-gym 2D state  = [x, x_dot, z, z_dot, theta, theta_dot]
safe-control-gym 2D action = [T1, T2]
```

During full deployment, the policy still outputs `[delta_thrust, theta_star]`.  The adapter converts `theta_star` into `[T1, T2]` using a PD attitude-tracking inner loop and logs the estimated `theta_star_dot`, tracking error, and motor-pair thrust clipping.

## What is new in this v2 overlay

Compared with the first overlay, this version adds:

1. `best_model.zip` saved by reward-oriented validation.
2. `final_model.zip` saved at the end of training.
3. Warm start from `best`, `final`, `latest`, or an explicit model path.
4. No deletion of old models. Every run receives a unique timestamped directory.
5. A lightweight by-lambda index at `models/transfer_quadrotor_reduced_policies/by_lambda/lambda_X/latest.json`.
6. PPO action-std modes: `learnable`, `fixed`, and `anneal`.
7. Physical interpretation of PPO std in newtons and degrees.
8. Conservative default action limits based on safe-control-gym's Crazyflie thrust bounds.
9. Full stdout/stderr training log in `training_full.log`.
10. Optional per-step rollout traces during evaluation.

## Copy into your safe-control-gym fork

From the downloaded overlay folder, copy these two directories into the root of your safe-control-gym fork:

```text
quadrotor_transfer/
scripts/
```

Your repo should look like:

```text
safe-control-gym/
  safe_control_gym/
  quadrotor_transfer/
  scripts/train_reduced_quadrotor_ppo.py
  scripts/evaluate_reduced_quadrotor_rollouts.py
  scripts/evaluate_expected_cumulative_reward_by_lambda.py
  scripts/inspect_action_limits.py
  scripts/smoke_test_env_shapes.py
```

Then install the repo and RL dependency:

```bash
python -m pip install -e .
python -m pip install stable-baselines3
```

## Action limits

The reduced policy action is normalized in `[-1, 1]^2`.

```text
action[0] -> delta_thrust in [-delta_thrust_max, +delta_thrust_max]
action[1] -> theta_star   in [-theta_max, +theta_max]
```

Defaults:

```text
theta_max = pi/8 = 22.5 deg
```

This matches your previous `[-pi/8, pi/8]` project.

For `delta_thrust_max`, the default is `None`, meaning the code computes a conservative symmetric limit from safe-control-gym's default Crazyflie pair-thrust bounds.  With the default Crazyflie parameters, it is about:

```text
delta_thrust_max ~= 0.152 N
```

This is lower than the old overlay's `0.35 N`, because `0.35 N` can produce reduced-model thrust commands that are not physically feasible for the full safe-control-gym motor-pair action bounds.  You can still override it:

```bash
--delta_thrust_max 0.20
```

but the default is intentionally more conservative and benchmark-aligned.

Inspect limits:

```bash
PYTHONPATH=. python scripts/inspect_action_limits.py
```

Inspect a different std choice:

```bash
PYTHONPATH=. python scripts/inspect_action_limits.py \
  --std_normalized 0.25 0.35
```

The two numbers correspond to:

```text
std for normalized delta_thrust action
std for normalized theta_star action
```

## PPO std modes

The policy acts in normalized action units, so std also starts in normalized units.

Example with default `theta_max = pi/8`:

```text
std_normalized = 0.35
std_theta_star = 0.35 * pi/8 = 0.137 rad ~= 7.9 deg
```

For delta thrust:

```text
std_delta_thrust = std_normalized * delta_thrust_max
```

Modes:

```bash
--std_mode learnable
```

SB3 learns the Gaussian log standard deviation.  The callback only sets the initial value.

```bash
--std_mode fixed
```

The Gaussian std is frozen throughout training.

```bash
--std_mode anneal
```

The Gaussian std is frozen and linearly annealed from `std_init_normalized` to `std_final_normalized`.

Examples:

```bash
# Same std for both action dimensions.
--std_mode learnable --std_init_normalized 0.35

# Different stds for delta_thrust and theta_star.
--std_mode learnable --std_init_normalized 0.25 0.35

# Fixed exploration scale.
--std_mode fixed --std_init_normalized 0.20 0.20

# Annealed exploration scale.
--std_mode anneal --std_init_normalized 0.40 0.35 --std_final_normalized 0.06 0.06
```

## Train reduced policies

Train lambda zero:

```bash
PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 0 \
  --total_timesteps 300000 \
  --std_mode learnable \
  --std_init_normalized 0.35
```

Train a smoother policy:

```bash
PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 0.5 \
  --total_timesteps 300000 \
  --std_mode anneal \
  --std_init_normalized 0.35 0.35 \
  --std_final_normalized 0.08 0.08
```

Another lambda:

```bash
PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 3 \
  --total_timesteps 300000 \
  --std_mode learnable
```

Each run creates:

```text
models/transfer_quadrotor_reduced_policies/runs/<run_id>/
  best_model.zip
  final_model.zip
  best_metrics.json
  best_model_evaluations.csv
  config.json
  run_complete.json
  checkpoints/

results/transfer_quadrotor_reduced_training/runs/<run_id>/
  training_full.log
  config.json
  action_limits.json
  std_analysis.json
  run_complete.json
```

No previous run is deleted.

The by-lambda index is updated here:

```text
models/transfer_quadrotor_reduced_policies/by_lambda/lambda_0/latest.json
models/transfer_quadrotor_reduced_policies/by_lambda/lambda_0p5/latest.json
models/transfer_quadrotor_reduced_policies/by_lambda/lambda_3/latest.json
```

This index points to the latest timestamped run but does not replace the actual saved models.

## Warm start

Warm start from the latest best model for the same lambda:

```bash
PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 0.5 \
  --warm_start best \
  --total_timesteps 150000
```

Warm start lambda 3 from the best lambda 0.5 model:

```bash
PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 3 \
  --warm_start best \
  --warm_start_lambda 0.5 \
  --total_timesteps 150000
```

Warm start from a path:

```bash
PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 0.5 \
  --warm_start path \
  --warm_start_model_path models/transfer_quadrotor_reduced_policies/runs/<run_id>/best_model.zip \
  --total_timesteps 150000
```

By default, warm start copies policy weights into a fresh PPO optimizer:

```bash
--warm_start_optimizer reset
```

For exact continuation, including optimizer state:

```bash
--warm_start_optimizer reuse --continue_num_timesteps_from_warm_start
```

For `--std_mode learnable`, warm start preserves the loaded policy's learned `log_std` by default.  To intentionally reset the exploration std at the start of a warm-start run, add:

```bash
--reset_std_on_warm_start
```

## Reward-oriented best model

The training script periodically evaluates the policy on held-out reduced-model rollouts and saves `best_model.zip` when the selected metric improves.

Default:

```bash
--best_metric task_return_mean
```

That means the best model is chosen using the **unpenalized task reward**, not the lambda-penalized training reward.  This is the right default for lambda sweeps because all lambdas are judged by the same navigation objective.

You can change it:

```bash
--best_metric discounted_task_return_mean
--best_metric success_rate
--best_metric train_return_mean
```

Change evaluation frequency:

```bash
--eval_freq 25000 --best_n_eval_episodes 20
```

Disable best-model validation, while still creating a `best_model.zip` alias to `final_model.zip`:

```bash
--eval_freq 0
```

## Save reduced-model trajectories

Use the reward-oriented best model by default:

```bash
PYTHONPATH=. python scripts/evaluate_reduced_quadrotor_rollouts.py \
  --lambda_penalty 0.5 \
  --model_selector best \
  --n_rollouts 25
```

Use final model instead:

```bash
--model_selector final
```

Outputs:

```text
results/transfer_quadrotor_reduced_rollouts/<lambda>_<selector>_<timestamp>/
  reduced_rollout_trajectories.csv
  summary.json
  config.json
  eval_initial_states.npy
```

## Full safe-control-gym deployment evaluation

Evaluate best models for lambdas `0`, `0.5`, and `3`:

```bash
PYTHONPATH=. python scripts/evaluate_expected_cumulative_reward_by_lambda.py \
  --lambda_values 0 0.5 3 \
  --model_selector best \
  --zeta_values 0.7 \
  --omega_n_values 2 4 6 8 10 12 \
  --n_rollouts 100 \
  --std_band
```

Quick debug run:

```bash
PYTHONPATH=. python scripts/evaluate_expected_cumulative_reward_by_lambda.py \
  --lambda_values 0 \
  --model_selector best \
  --omega_n_values 4 8 \
  --n_rollouts 5 \
  --skip_missing \
  --save_rollout_traces
```

Outputs:

```text
results/transfer_quadrotor_expected_reward_by_lambda/expected_cumulative_reward_by_lambda_best_<timestamp>/
  expected_cumulative_reward_by_lambda_summary.csv
  expected_cumulative_reward_by_lambda_zeta_0p7.png
  reduced_baseline_summary.json
  action_limits.json
  config.json
  eval_initial_states.npy
  rollout_traces/                       # only with --save_rollout_traces
```

Important logged columns:

```text
full_expected_cumulative_reward_mean
full_minus_reduced_baseline_mean
full_theta_variation_sum_mean
full_theta_tracking_abs_mean
full_motor_clip_rate_mean
full_motor_clip_abs_sum_mean
full_success_rate
full_final_distance_mean
```

The motor clipping columns are especially useful.  If `full_motor_clip_rate_mean` is high, the deployment adapter is asking for thrusts outside safe-control-gym's physical action bounds.  In that case, reduce `theta_max_deg`, reduce `delta_thrust_max`, reduce `omega_n`, increase `theta_star_tau`, or train with a larger lambda.

## Suggested first experimental sequence

1. Run action-limit inspection.
2. Train lambda 0 with learnable std.
3. Train lambda 0.5 warm-started from lambda 0.
4. Train lambda 3 warm-started from lambda 0.5.
5. Save reduced rollouts for each lambda.
6. Evaluate full deployment across `omega_n` with `zeta=0.7`.
7. Inspect motor clipping and theta tracking error.
8. Then decide whether to broaden the task distribution or add disturbances/constraints.

Commands:

```bash
PYTHONPATH=. python scripts/inspect_action_limits.py

PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 0 \
  --total_timesteps 300000 \
  --std_mode learnable

PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 0.5 \
  --warm_start best \
  --warm_start_lambda 0 \
  --total_timesteps 200000 \
  --std_mode anneal

PYTHONPATH=. python scripts/train_reduced_quadrotor_ppo.py \
  --lambda_penalty 3 \
  --warm_start best \
  --warm_start_lambda 0.5 \
  --total_timesteps 200000 \
  --std_mode anneal

PYTHONPATH=. python scripts/evaluate_expected_cumulative_reward_by_lambda.py \
  --lambda_values 0 0.5 3 \
  --model_selector best \
  --omega_n_values 2 4 6 8 10 12 \
  --zeta_values 0.7 \
  --n_rollouts 100 \
  --std_band
```

## Notes on not changing upstream

This overlay keeps all new code outside `safe_control_gym/`.  That is the cleanest workflow for your goal:

- upstream benchmark code stays pristine;
- your reduced model and deployment adapter are explicit and reviewable;
- experiment scripts can be committed as your own contribution layer;
- if the upstream repo updates, merge conflicts are much less likely.

If later you decide this task should become a true upstream environment, then you can refactor it into `safe_control_gym` as a formal contribution.  For now, the overlay approach is safer and more academic-project friendly.
