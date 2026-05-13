
$YamlPath = ".\environment.yml"

if (-not (Test-Path $YamlPath)) {throw "YAML file not found: $YamlPath"} 

if (-not (Test-Path ".venv\Scripts\python.exe")) {throw @"
    To create/activate venv, run:

    py -3.10 -m venv .\.venv --without-pip
    .\.venv\Scripts\python.exe -m ensurepip
    . .\.venv\Scripts\Activate.ps1
"@}

# parse the YAML file to extract pip dependencies
$lines = Get-Content $YamlPath
$packages = @()
$insidePipBlock = $false
foreach ($line in $lines) {
    # trim each line to simplify checks
    $trimmed = $line.Trim() 

    # skip empty lines and comments
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) {continue}

    # detect pip block
    if ($trimmed -eq "- pip:") {$insidePipBlock = $true; continue}

    # capture packages inside pip block
    if ($insidePipBlock -and $trimmed.StartsWith("- ")) {
        $spec = ($trimmed.Substring(2) -split "\s+#", 2)[0].Trim()
        if ($spec -ne "") {$packages += $spec}
    } else {$insidePipBlock = $false}
}

if ($packages.Count -eq 0) {
    Write-Host "No pip-installable packages found in $YamlPath" -ForegroundColor Yellow
} else {
    Write-Host "`nInstalling pip packages from $YamlPath..." -ForegroundColor Cyan
    foreach ($pkg in $packages) {
        Write-Host "Installing $pkg"
         & .\.venv\Scripts\python.exe -m pip install -qq $pkg
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to install $pkg" -ForegroundColor Red
            exit 1
        } 
    } Write-Host "`nAll packages installed successfully!" -ForegroundColor Green     
}

# imnport each pkg
# from gym impor utils
