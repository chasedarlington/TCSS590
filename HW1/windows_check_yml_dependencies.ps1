param(
    [string]$EnvFile = "environment.yml",
    [switch]$InstallMissing = $true
)

Write-Host ""
Write-Host "==============================="
Write-Host "Checking dependencies from: $EnvFile" -ForegroundColor Cyan
Write-Host "==============================="

if (!(Test-Path $EnvFile)) {
    Write-Host "ERROR: File not found: $EnvFile" -ForegroundColor Red
    exit 1
}

Write-Host "Current directory:" -ForegroundColor Cyan
Get-Location
Write-Host ""

Write-Host "Python version:" -ForegroundColor Cyan
python --version
Write-Host ""

Write-Host "Pip version:" -ForegroundColor Cyan
python -m pip --version
Write-Host ""

# Pip packages from YAML
$pipDeps = @(
    @{ Name = "cython"; ImportName = "Cython"; Spec = "cython==0.29.34" },
    @{ Name = "gym"; ImportName = "gym"; Spec = "gym==0.19.0" },
    @{ Name = "gymnasium"; ImportName = "gymnasium"; Spec = "gymnasium==0.29.1" },
    @{ Name = "gymnasium-robotics"; ImportName = "gymnasium_robotics"; Spec = "gymnasium-robotics[mujoco-py]" },
    @{ Name = "gym-notices"; ImportName = "gym_notices"; Spec = "gym-notices==0.0.8" },
    @{ Name = "matplotlib"; ImportName = "matplotlib"; Spec = "matplotlib==3.7.1" },
    @{ Name = "mujoco"; ImportName = "mujoco"; Spec = "mujoco==2.3.7" },
    @{ Name = "mujoco-py"; ImportName = "mujoco_py"; Spec = "mujoco-py==2.1.2.14" },
    @{ Name = "numpy"; ImportName = "numpy"; Spec = "numpy==1.24.3" },
    @{ Name = "packaging"; ImportName = "packaging"; Spec = "packaging==23.1" },
    @{ Name = "pybullet"; ImportName = "pybullet"; Spec = "pybullet==3.2.5" },
    @{ Name = "torch"; ImportName = "torch"; Spec = "torch==1.13.1" },
    @{ Name = "torchvision"; ImportName = "torchvision"; Spec = "torchvision" },
    @{ Name = "torchaudio"; ImportName = "torchaudio"; Spec = "torchaudio" },
    @{ Name = "wheel"; ImportName = "wheel"; Spec = "wheel==0.38.0" }
)

Write-Host ""
Write-Host "==============================="
Write-Host "Checking / Installing Pip packages"
Write-Host "==============================="

foreach ($pkg in $pipDeps) {
    Write-Host ""
    Write-Host "Checking pip package: $($pkg.Name)" -ForegroundColor Yellow

    python -m pip show $($pkg.Name) *> $null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "NOT FOUND: $($pkg.Name)" -ForegroundColor Red

        if ($InstallMissing) {
            Write-Host "Installing: $($pkg.Spec)" -ForegroundColor Cyan
            python -m pip install "$($pkg.Spec)"

            if ($LASTEXITCODE -ne 0) {
                Write-Host "INSTALL FAILED: $($pkg.Spec)" -ForegroundColor Red
            } else {
                Write-Host "INSTALL OK: $($pkg.Spec)" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "FOUND: $($pkg.Name)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "==============================="
Write-Host "Import tests"
Write-Host "==============================="

foreach ($pkg in $pipDeps) {
    $module = $pkg.ImportName

    Write-Host ""
    Write-Host "Importing: $module" -ForegroundColor Yellow

    python -c "import $module; print('OK: $module')" 2>$null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "IMPORT FAILED: $module" -ForegroundColor Red
    } else {
        Write-Host "IMPORT OK: $module" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "==============================="
Write-Host "Version summary"
Write-Host "==============================="

$versionScript = @"
packages = [
    ("Cython", "Cython"),
    ("gym", "gym"),
    ("gymnasium", "gymnasium"),
    ("gymnasium_robotics", "gymnasium_robotics"),
    ("gym_notices", "gym_notices"),
    ("matplotlib", "matplotlib"),
    ("mujoco", "mujoco"),
    ("mujoco_py", "mujoco_py"),
    ("numpy", "numpy"),
    ("packaging", "packaging"),
    ("pybullet", "pybullet"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("wheel", "wheel"),
]

for import_name, display_name in packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__")
        print(f"{display_name}: {version}")
    except Exception as e:
        print(f"{display_name}: FAILED - {e}")
"@

python -c $versionScript