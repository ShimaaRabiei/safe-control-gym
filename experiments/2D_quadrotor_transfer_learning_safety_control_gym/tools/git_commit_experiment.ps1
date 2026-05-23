param(
    [string]$Root = "C:\Users\rabiei\My Research\safe-control-gym",
    [string]$ExpName = "2D_quadrotor_transfer_learning_safety_control_gym",
    [string]$Branch = "add-2d-quadrotor-transfer-experiment",
    [string]$Message = "Add 2D quadrotor transfer learning experiment"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $Root
$ExpRel = "experiments/$ExpName"

if (-not (Test-Path (Join-Path $Root "experiments\$ExpName"))) {
    throw "Experiment folder not found: $(Join-Path $Root "experiments\$ExpName")"
}

git status

$current = git branch --show-current
if ($current -ne $Branch) {
    $exists = git branch --list $Branch
    if ($exists) {
        git switch $Branch
    } else {
        git switch -c $Branch
    }
}

git add $ExpRel

git status --short
git diff --cached --stat

Write-Host "If the staged files look right, run:"
Write-Host "git commit -m `"$Message`""
Write-Host "git push -u origin $Branch"
