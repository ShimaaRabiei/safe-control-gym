from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

def lambda_name(value: float) -> str:
    s = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"lambda_{s}"

def safe_lam_tag(value: float) -> str:
    return f"lam_{str(value).replace('-', 'm').replace('.', 'p')}"

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def latest_matching_run(out_root: Path, lam: float, run_tag: str) -> Path:
    prefix = f"reduced_{lambda_name(lam)}_{run_tag}_"
    candidates = [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No output folder found with prefix {prefix} in {out_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def plot_metric(rows: List[Dict[str, Any]], y: str, title: str, ylabel: str, out_path: Path) -> None:
    xs = [float(r["lambda_penalty"]) for r in rows]
    ys = [float(r[y]) for r in rows]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    for x, yy in zip(xs, ys):
        plt.annotate(f"{x:g}", (x, yy), textcoords="offset points", xytext=(0, 6), ha="center")
    plt.xlabel("lambda")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_overlay(run_infos: List[Dict[str, Any]], ycol: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    for info in run_infos:
        df = pd.read_csv(info["curve_csv"])
        if df.empty or ycol not in df.columns:
            continue
        lam = info["lambda_penalty"]
        plt.plot(df["timesteps"], df[ycol], marker="o", label=f"lambda={lam:g}")
    plt.xlabel("training timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_values", type=float, nargs="+", default=[0, 10])
    parser.add_argument("--total_timesteps", type=int, default=600_000)
    parser.add_argument("--eval_every", type=int, default=50_000)
    parser.add_argument("--eval_rollouts", type=int, default=10)
    parser.add_argument("--final_rollouts", type=int, default=333)
    parser.add_argument("--final_init_states_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--run_tag_prefix", type=str, default="lambda_sweep")
    parser.add_argument("--train_script", type=Path, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--python_exe", type=str, default=sys.executable)

    parser.add_argument("--theta_max_deg", type=float, default=35.0)
    parser.add_argument("--x_goal", type=float, default=3.0)
    parser.add_argument("--z_goal", type=float, default=7.0)
    parser.add_argument("--init_x_low", type=float, default=-3.5)
    parser.add_argument("--init_x_high", type=float, default=-2.5)
    parser.add_argument("--init_z_low", type=float, default=0.5)
    parser.add_argument("--init_z_high", type=float, default=1.5)
    parser.add_argument("--init_x_dot_low", type=float, default=-1.0)
    parser.add_argument("--init_x_dot_high", type=float, default=1.0)
    parser.add_argument("--init_z_dot_low", type=float, default=-1.0)
    parser.add_argument("--init_z_dot_high", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=100.0)
    parser.add_argument("--target_radius", type=float, default=0.15)
    parser.add_argument("--target_bonus", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--x_bound", type=float, default=4.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=8.0)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = args.train_script or (script_dir / "train_eval_plot_onefile.py")
    plot_script = script_dir / "plot_mean_std_from_trajectory_csv.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Could not find trainer script: {train_script}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.model_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    run_infos: List[Dict[str, Any]] = []

    print("\n===================================================")
    print(f"Trainer:    {train_script}")
    print(f"Out root:   {args.out_root}")
    print(f"Model root: {args.model_root}")
    print(f"Lambdas:    {args.lambda_values}")
    print("===================================================\n")

    for lam in args.lambda_values:
        run_tag = f"{args.run_tag_prefix}_{safe_lam_tag(lam)}"
        existing_run = None

        if args.skip_existing:
            try:
                candidate = latest_matching_run(args.out_root, lam, run_tag)
                if (candidate / "final_evaluation_summary.json").exists():
                    existing_run = candidate
            except FileNotFoundError:
                existing_run = None

        if existing_run is None:
            cmd = [
                args.python_exe, str(train_script),
                "--lambda_penalty", str(lam),
                "--out_root", str(args.out_root),
                "--model_root", str(args.model_root),
                "--run_tag", run_tag,
                "--total_timesteps", str(args.total_timesteps),
                "--eval_every", str(args.eval_every),
                "--eval_rollouts", str(args.eval_rollouts),
                "--final_rollouts", str(args.final_rollouts),
                "--final_init_states_path", str(args.final_init_states_path),
                "--seed", str(args.seed),
                "--theta_max_deg", str(args.theta_max_deg),
                "--x_goal", str(args.x_goal),
                "--z_goal", str(args.z_goal),
                "--init_x_low", str(args.init_x_low),
                "--init_x_high", str(args.init_x_high),
                "--init_z_low", str(args.init_z_low),
                "--init_z_high", str(args.init_z_high),
                "--init_x_dot_low", str(args.init_x_dot_low),
                "--init_x_dot_high", str(args.init_x_dot_high),
                "--init_z_dot_low", str(args.init_z_dot_low),
                "--init_z_dot_high", str(args.init_z_dot_high),
                "--reward_scale", str(args.reward_scale),
                "--target_radius", str(args.target_radius),
                "--target_bonus", str(args.target_bonus),
                "--max_steps", str(args.max_steps),
                "--x_bound", str(args.x_bound),
                "--z_min", str(args.z_min),
                "--z_max", str(args.z_max),
            ]
            print("\n---------------------------------------------------")
            print(f"Training lambda = {lam}")
            print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
            print("---------------------------------------------------\n")
            subprocess.run(cmd, check=True)
            run_dir = latest_matching_run(args.out_root, lam, run_tag)
        else:
            run_dir = existing_run
            print(f"Skipping lambda={lam}; found completed run: {run_dir}")

        traj_csv = run_dir / "final_evaluation_trajectory.csv"
        if plot_script.exists() and traj_csv.exists():
            subprocess.run(
                [args.python_exe, str(plot_script), "--csv", str(traj_csv), "--lambda_value", str(lam)],
                check=False,
            )

        final_summary_path = run_dir / "final_evaluation_summary.json"
        curve_csv = run_dir / "training_eval_curve.csv"

        if not final_summary_path.exists():
            raise FileNotFoundError(f"Missing final summary for lambda={lam}: {final_summary_path}")

        final_summary = read_json(final_summary_path)

        row: Dict[str, Any] = {
            "lambda_penalty": lam,
            "run_dir": str(run_dir),
            "final_model_dir": str(args.model_root / run_dir.name),
        }
        row.update(final_summary)
        summary_rows.append(row)
        run_infos.append({"lambda_penalty": lam, "run_dir": str(run_dir), "curve_csv": str(curve_csv)})

    summary_rows.sort(key=lambda r: float(r["lambda_penalty"]))
    run_infos.sort(key=lambda r: float(r["lambda_penalty"]))

    combined_csv = args.out_root / "lambda_sweep_final_summary.csv"
    write_csv(combined_csv, summary_rows)

    if summary_rows:
        plot_metric(summary_rows, "discounted_task_return_mean", "Final discounted task return vs lambda", "discounted task return", args.out_root / "lambda_vs_final_discounted_task_return.png")
        plot_metric(summary_rows, "discounted_theta_variation_mean", "Final discounted theta variation vs lambda", "discounted theta variation", args.out_root / "lambda_vs_final_discounted_theta_variation.png")
        plot_metric(summary_rows, "theta_variation_sum_mean", "Final theta variation sum vs lambda", "theta variation sum", args.out_root / "lambda_vs_final_theta_variation_sum.png")
        plot_metric(summary_rows, "success_rate", "Final success rate vs lambda", "success rate", args.out_root / "lambda_vs_final_success_rate.png")
        plot_metric(summary_rows, "final_distance_mean", "Final distance vs lambda", "final distance", args.out_root / "lambda_vs_final_distance.png")

    plot_overlay(run_infos, "discounted_task_return_mean", "Learning curve summary across lambda", "discounted task return", args.out_root / "summary_learning_curve_discounted_task_return.png")
    plot_overlay(run_infos, "discounted_theta_variation_mean", "Discounted theta-reference variation during training across lambda", "discounted theta-reference variation", args.out_root / "summary_discounted_theta_variation_during_training.png")
    plot_overlay(run_infos, "theta_variation_sum_mean", "Theta-reference variation sum during training across lambda", "theta-reference variation sum", args.out_root / "summary_theta_variation_sum_during_training.png")
    plot_overlay(run_infos, "success_rate", "Success rate during training across lambda", "success rate", args.out_root / "summary_success_rate_during_training.png")

    print("\n===================================================")
    print(f"Combined summary CSV: {combined_csv}")
    print(f"Combined plots folder: {args.out_root}")
    print("Each lambda keeps its own individual plots inside its run folder.")
    print("Mean/std final trajectory plots are inside each lambda folder.")
    print("===================================================\n")

if __name__ == "__main__":
    main()
