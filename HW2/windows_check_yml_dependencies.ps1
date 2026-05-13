param(
    [string]$YamlPath = ".\environment.yml",
    [string]$PythonExe = "python"
)

if (-not (Test-Path $YamlPath)) {
    throw "YAML file not found: $YamlPath"
}

$lines = Get-Content $YamlPath

$packages = @()
$insidePipBlock = $false

foreach ($line in $lines) {
    $trimmed = $line.Trim()

    # Skip empty lines and comments
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) {
        continue
    }

    # Detect pip block
    if ($trimmed -eq "- pip:") {
        $insidePipBlock = $true
        continue
    }

    # Capture packages inside pip block
    if ($insidePipBlock -and $trimmed.StartsWith("- ")) {
        $spec = $trimmed.Substring(2).Trim()

        if ($spec -ne "") {
            $packages += $spec
        }

        continue
    }

    # Capture top-level conda dependencies
    if (-not $insidePipBlock -and $trimmed.StartsWith("- ")) {
        $spec = $trimmed.Substring(2).Trim()

        # Skip Python itself because pip cannot install/replace the interpreter
        if ($spec -match "^python[=<>!~]") {
            Write-Host "Skipping Python interpreter spec: $spec"
            continue
        }

        # Skip bare pip entry because we handle pip itself separately
        if ($spec -eq "pip") {
            continue
        }

        # Convert conda single equals to pip double equals
        # Example: numpy=1.24.3 -> numpy==1.24.3
        if ($spec -match "^[A-Za-z0-9_.-]+=[^=].*") {
            $spec = $spec -replace "=", "=="
        }

        $packages += $spec
    }
}

if ($packages.Count -eq 0) {
    Write-Host "No pip-installable packages found in $YamlPath"
    exit 0
}

Write-Host "Packages pulled from YAML:"
$packages | ForEach-Object { Write-Host "  $_" }

Write-Host ""
Write-Host "Installing packages..."

foreach ($pkg in $packages) {
    Write-Host "Installing $pkg"
    & $PythonExe -m pip install $pkg
}