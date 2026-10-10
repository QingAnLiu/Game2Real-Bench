param(
    [string]$Repository = "QingAnLiu/Game2Real-Bench",
    [string]$RepositoryUrl = "",
    [string]$PythonCommand = "python",
    [string]$GhCommand = "gh",
    [string]$CommitMessage = "Prepare Game2Real-Bench private preview",
    [switch]$Publish
)

$ErrorActionPreference = "Stop"
$CodeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DatasetRoot = (Resolve-Path (Join-Path $CodeRoot "..\dataset")).Path

& $PythonCommand (Join-Path $PSScriptRoot "validate_release.py") --dataset-root $DatasetRoot --code-root $CodeRoot
if ($LASTEXITCODE -ne 0) {
    throw "Release validation failed."
}

if (-not $Publish) {
    Write-Host "Validation passed. Re-run with -Publish to publish the private repository $Repository."
    exit 0
}

Push-Location $CodeRoot
try {
    Get-Command git -ErrorAction Stop | Out-Null
    if (-not (Test-Path ".git")) {
        git init
        git branch -M main
    }

    if (-not (git config --local user.name)) {
        git config --local user.name "Game2Real-Bench Contributors"
    }
    if (-not (git config --local user.email)) {
        git config --local user.email "game2real-bench@users.noreply.github.com"
    }

    git add --all
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m $CommitMessage
    }

    $Remote = git remote get-url origin 2>$null
    if (-not $Remote) {
        if ($RepositoryUrl) {
            git remote add origin $RepositoryUrl
            git push -u origin main
        }
        else {
            Get-Command $GhCommand -ErrorAction Stop | Out-Null
            & $GhCommand auth status
            if ($LASTEXITCODE -ne 0) {
                throw "GitHub CLI authentication is required. Run: gh auth login"
            }
            & $GhCommand repo create $Repository --private --source . --remote origin --push
            if ($LASTEXITCODE -ne 0) {
                throw "Private GitHub repository creation failed."
            }
        }
    }
    else {
        git push -u origin main
    }

    Write-Host "Published privately: https://github.com/$Repository"
}
finally {
    Pop-Location
}

