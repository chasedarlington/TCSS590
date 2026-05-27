# check_dependencies.ps1
# Checks whether required Python packages/modules can be imported

Write-Host "Checking Python dependencies..." -ForegroundColor Cyan

# Check Python
try {
    Write-Host ""
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "Python was not found. Try installing Python or using py instead of python." -ForegroundColor Red
    exit 1
}

# here are our Python modules to check 
$modules = @(
    "os",
    "torch",
    "torch.nn",
    "torch.optim",
    "argparse",
    "collections",
    "math",
    "time",
    "typing",
    "gym",
    "copy",
    "numpy",
    "matplotlib.pyplot"
)

$packages = @(
    "torch",
    "gym",
    "numpy",
    "matplotlib"
)

# local project modules to check
$localChecks = @(
    "from utils import mlp",
    "from utils import collect_trajs"
)

Write-Host "`nChecking standard and installed modules..." -ForegroundColor Cyan
foreach ($mod in $modules) {
    # command we use to check modules and packages
    $checkCommand = "import $mod; print($mod.__version__)"
    $modVersion = python -c $checkCommand 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "[OK] $mod version: $modVersion" -ForegroundColor Green
    } else { Write-Host "[MISSING/ERROR] $mod" -ForegroundColor Red }
}

Write-Host "`nChecking key package versions..." -ForegroundColor Cyan
foreach ($pkg in $packages) {
    $checkCommand = "import $pkg; print($pkg.__version__)"
    $pkgVersion = python -c $checkCommand 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "[OK] $pkg version: $pkgVersion" -ForegroundColor Green
    } else { Write-Host "[MISSING/ERROR] $pkg" -ForegroundColor Red }
}

Write-Host "`nChecking local project imports..." -ForegroundColor Cyan
foreach ($check in $localChecks) { 
    python -c $check 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "[OK] $check" -ForegroundColor Green
    } else { Write-Host "[MISSING/ERROR] $check" -ForegroundColor Red }
}

Write-Host "`nDependency check complete." -ForegroundColor Cyan



Write-Host "`nChecking GPU availability..." -ForegroundColor Cyan
python -c @"
import torch

print('CUDA available:', torch.cuda.is_available())
print('CUDA version used by PyTorch:', torch.version.cuda)

if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.current_device())
    print('GPU name:', torch.cuda.get_device_name(0))
    x = torch.rand(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print('Tensor device:', y.device)
else:
    print('GPU check: PyTorch is running on CPU only')
"@