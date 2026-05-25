from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path(r"C:\Users\rabiei\My Research\safe-control-gym")

csv_files = [
    root / "results" / "F3731_full_deployment_zeta03" / "full_deployment_summary.csv",
    root / "results" / "F3731_full_deployment_zeta04" / "full_deployment_summary.csv",
    root / "results" / "F3731_full_deployment_zeta07" / "full_deployment_summary.csv",
    root / "results" / "F3731_full_deployment_zeta1"  / "full_deployment_summary.csv",
]

out_dir = root / "results" / "F3731_best_lambda_heatmap"
out_dir.mkdir(parents=True, exist_ok=True)

zeta_from_folder = {
    "F3731_full_deployment_zeta03": 0.3,
    "F3731_full_deployment_zeta04": 0.4,
    "F3731_full_deployment_zeta07": 0.7,
    "F3731_full_deployment_zeta1": 1.0,
}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def find_col(df, exact_names, required_tokens=None):
    mp = {norm(c): c for c in df.columns}
    for name in exact_names:
        if norm(name) in mp:
            return mp[norm(name)]
    if required_tokens is not None:
        for c in df.columns:
            nc = norm(c)
            if all(t in nc for t in required_tokens):
                return c
    raise KeyError(f"Could not find column. Available columns are: {list(df.columns)}")

frames = []
for p in csv_files:
    if not p.exists():
        raise FileNotFoundError(f"Missing summary CSV: {p}")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if not any(norm(c) == "zeta" for c in df.columns):
        df["zeta"] = zeta_from_folder[p.parent.name]
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

lambda_col = find_col(df, ["lambda", "lambda_penalty", "lambda_value"], ["lambda"])
omega_col  = find_col(df, ["omega_n", "omega", "omega_n_value"], ["omega"])
zeta_col   = find_col(df, ["zeta", "damping_ratio"], ["zeta"])

return_candidates = [
    "discounted_task_return_mean",
    "disc_task_return_mean",
    "task_return_mean",
    "mean_task_return",
    "discounted_return_mean",
    "return_mean",
    "task_return",
]
try:
    ret_col = find_col(df, return_candidates)
except KeyError:
    try:
        ret_col = find_col(df, [], ["task", "return", "mean"])
    except KeyError:
        ret_col = find_col(df, [], ["return", "mean"])

print("Using return column:", ret_col)

for c in [lambda_col, omega_col, zeta_col, ret_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[lambda_col, omega_col, zeta_col, ret_col])

agg = df.groupby([zeta_col, omega_col, lambda_col], as_index=False)[ret_col].mean()
idx = agg.groupby([zeta_col, omega_col])[ret_col].idxmax()
best = agg.loc[idx].sort_values([zeta_col, omega_col])

best_out = best.rename(columns={
    zeta_col: "zeta",
    omega_col: "omega_n",
    lambda_col: "best_lambda",
    ret_col: "best_discounted_task_return",
})

csv_out = out_dir / "best_lambda_by_discounted_task_return.csv"
best_out.to_csv(csv_out, index=False)

heat = best.pivot(index=zeta_col, columns=omega_col, values=lambda_col).sort_index()
heat = heat.reindex(sorted(heat.columns), axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(
    heat.values.astype(float),
    aspect="auto",
    cmap="viridis",
    vmin=float(np.nanmin(df[lambda_col])),
    vmax=float(np.nanmax(df[lambda_col])),
)

ax.set_title("Best lambda by full-deployment discounted task return")
ax.set_xlabel("Natural frequency omega_n [rad/s]")
ax.set_ylabel("Damping ratio zeta")

ax.set_xticks(np.arange(len(heat.columns)))
ax.set_xticklabels([f"{x:g}" for x in heat.columns])
ax.set_yticks(np.arange(len(heat.index)))
ax.set_yticklabels([f"{y:g}" for y in heat.index])

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.iloc[i, j]
        if not pd.isna(val):
            ax.text(j, i, f"{val:g}", ha="center", va="center", color="black", fontsize=12)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Best lambda")
cbar.set_ticks(sorted(df[lambda_col].dropna().unique()))

png_out = out_dir / "best_lambda_by_discounted_task_return_heatmap.png"
fig.tight_layout()
fig.savefig(png_out, dpi=300, bbox_inches="tight")

print("\nSaved:")
print(png_out)
print(csv_out)
print("\nBest lambda table:")
print(best_out.to_string(index=False))
