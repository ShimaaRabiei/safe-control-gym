import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--lambda_value", default="")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    if "episode" not in df.columns:
        raise ValueError("CSV must contain an episode column.")

    n_ep = int(df["episode"].nunique())
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "mean_std_final_trajectory_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_df = df.groupby("time").mean(numeric_only=True).reset_index()
    std_df = df.groupby("time").std(numeric_only=True).reset_index().fillna(0.0)

    mean_df.to_csv(out_dir / "mean_trajectory.csv", index=False)
    std_df.to_csv(out_dir / "std_trajectory.csv", index=False)

    lam_text = f", lambda={args.lambda_value}" if args.lambda_value != "" else ""

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
        plt.title(f"{title}{lam_text}, mean±std over {n_ep} episodes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()

    save_plot(["x", "z"], "Position states", "position", "mean_std_position_x_z.png")
    save_plot(["x_dot", "z_dot"], "Velocity states", "velocity", "mean_std_velocity_xdot_zdot.png")
    save_plot(["theta_star"], "Theta-reference input", "theta_star [rad]", "mean_std_theta_star.png")
    save_plot(["delta_thrust"], "Delta thrust input", "Delta T [N]", "mean_std_delta_thrust.png")
    save_plot(["distance_to_target"], "Distance to target", "distance", "mean_std_distance_to_target.png")
    save_plot(["task_reward", "train_reward"], "Reward signals", "reward", "mean_std_rewards.png")

    print(f"[MeanStdPlot] Saved plots to: {out_dir}")
    print(f"[MeanStdPlot] Episodes: {n_ep}")

if __name__ == "__main__":
    main()
