param(
    [string]$Root = "C:\Users\rabiei\My Research\safe-control-gym",
    [string]$Work = "",
    [string]$ExpName = "2D_quadrotor_transfer_learning_safety_control_gym",
    [switch]$CleanCode,
    [switch]$NoModels,
    [switch]$NoResults
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Work -eq "") {
    $Work = Join-Path $Root "overlays\WORK_FINAL_target37_startm31_fixed333"
}

$Exp = Join-Path (Join-Path $Root "experiments") $ExpName
$Code = Join-Path $Exp "code"
$CodeScripts = Join-Path $Code "scripts"
$Data = Join-Path $Exp "data"
$Tools = Join-Path $Exp "tools"

New-Item -ItemType Directory -Force -Path $Exp, $Code, $CodeScripts, $Data, $Tools | Out-Null

function Copy-DirectorySafe {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path $Source)) {
        Write-Host "Missing source, skipped: $Source"
        return
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    robocopy $Source $Destination /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed for $Source -> $Destination with exit code $LASTEXITCODE"
    }
}

function Copy-FileSafe {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path $Source)) {
        Write-Host "Missing file, skipped: $Source"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

$ThisToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent $ThisToolDir
Copy-FileSafe (Join-Path $PackageRoot "README.md") (Join-Path $Exp "README.md")
Copy-DirectorySafe $ThisToolDir $Tools

$QuadSrc = Join-Path $Work "quadrotor_transfer"
if (-not (Test-Path $QuadSrc)) {
    $QuadSrc = Join-Path $Root "quadrotor_transfer"
}
Copy-DirectorySafe $QuadSrc (Join-Path $Code "quadrotor_transfer")

$ScriptNames = @(
    "train_eval_plot_onefile.py",
    "run_lambda_sweep_onefile.py",
    "deploy_existing_lambda_models_full_meanstd.py",
    "plot_mean_std_from_trajectory_csv.py"
)
foreach ($Name in $ScriptNames) {
    $Src = Join-Path (Join-Path $Work "scripts") $Name
    if (-not (Test-Path $Src)) {
        $Src = Join-Path (Join-Path $Root "scripts") $Name
    }
    Copy-FileSafe $Src (Join-Path $CodeScripts $Name)
}

Copy-FileSafe (Join-Path $Work "data\eval_fixed333_start_m3_1_radius025.npy") (Join-Path $Data "eval_fixed333_start_m3_1_radius025.npy")

if (-not $NoModels) {
    Copy-DirectorySafe (Join-Path $Root "models\F3731") (Join-Path $Exp "models\F3731")
}

if (-not $NoResults) {
    Copy-DirectorySafe (Join-Path $Root "results\F3731") (Join-Path $Exp "results\reduced_training_evaluations")
    Copy-DirectorySafe (Join-Path $Root "results\F3731_full_deployment_zeta03") (Join-Path $Exp "results\deployment_zeta03")
    Copy-DirectorySafe (Join-Path $Root "results\F3731_full_deployment_zeta04") (Join-Path $Exp "results\deployment_zeta04")
    Copy-DirectorySafe (Join-Path $Root "results\F3731_full_deployment_zeta07") (Join-Path $Exp "results\deployment_zeta07")
    Copy-DirectorySafe (Join-Path $Root "results\F3731_full_deployment_zeta1") (Join-Path $Exp "results\deployment_zeta1")
}

if ($CleanCode) {
    python (Join-Path $Tools "clean_code_comments.py") --root $Code
}

Write-Host "Experiment folder created: $Exp"
Write-Host "Check with: Get-ChildItem -Recurse '$Exp' | Select-Object -First 40"
