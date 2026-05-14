Get-ChildItem .\HW2 -Recurse -Filter *.py |
Where-Object {
    $_.FullName -notmatch '\\\.venv\\' -and
    $_.FullName -notmatch '\\\.vscode\\' -and
    $_.FullName -notmatch '\\__pycache__\\'
} | ForEach-Object {
    $path = $_.FullName
    $content = Get-Content $path -Raw

    $newContent = $content `
        -replace '^\s*import gymnasium as gymnasium as gym\b', 'import gymnasium as gym' `
        -replace '^\s*import gym\b', 'import gymnasium as gym' `
        -replace '^\s*from gym import utils\b', 'from gymnasium import utils'

    if ($newContent -ne $content) {
        Write-Host "Updated $path"
        Set-Content $path $newContent
    }
}