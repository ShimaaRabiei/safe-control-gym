# Optional Git LFS setup

If the repository should include trained model files, use Git LFS for model artifacts before `git add`:

```powershell
git lfs install
git lfs track "experiments/2D_quadrotor_transfer_learning_safety_control_gym/models/**/*.zip"
git lfs track "experiments/2D_quadrotor_transfer_learning_safety_control_gym/models/**/*.pkl"
git lfs track "experiments/2D_quadrotor_transfer_learning_safety_control_gym/data/**/*.npy"
git add .gitattributes
```

If the repository should contain only code, README, plots, and CSV summaries, run the package script with `-NoModels`.
