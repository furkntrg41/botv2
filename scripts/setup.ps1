# ═══════════════════════════════════════════════════════════════════════════════
# ALGO TRADING BOT - PROJECT SETUP SCRIPT (POWERSHELL)
# ═══════════════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

$ProjectName = "algo_trading_bot"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Initializing: $ProjectName" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# ─────────────────────────────────────────────────────────────────────────────────
# CREATE DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────────
Write-Host "[1/4] Creating directory structure..." -ForegroundColor Yellow

$Dirs = @(
    "src\data",
    "src\analysis",
    "src\execution",
    "src\models",
    "src\utils",
    "tests",
    "scripts",
    "notebooks"
)

foreach ($dir in $Dirs) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
    Write-Host "  ✓ Created $dir\" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────────────
# CREATE __init__.py FILES
# ─────────────────────────────────────────────────────────────────────────────────
Write-Host "[2/4] Creating Python package files..." -ForegroundColor Yellow

$InitFiles = @(
    "src\__init__.py",
    "src\data\__init__.py",
    "src\analysis\__init__.py",
    "src\execution\__init__.py",
    "src\models\__init__.py",
    "src\utils\__init__.py",
    "tests\__init__.py"
)

foreach ($init in $InitFiles) {
    $fullPath = Join-Path $ProjectRoot $init
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType File -Path $fullPath -Force | Out-Null
    }
    Write-Host "  ✓ Created $init" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────────────
# CREATE .gitkeep FILES
# ─────────────────────────────────────────────────────────────────────────────────
Write-Host "[3/4] Creating placeholder files..." -ForegroundColor Yellow

$GitkeepDirs = @(
    "scripts",
    "notebooks",
    "src\models"
)

foreach ($dir in $GitkeepDirs) {
    $fullPath = Join-Path $ProjectRoot "$dir\.gitkeep"
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType File -Path $fullPath -Force | Out-Null
    }
    Write-Host "  ✓ Created $dir\.gitkeep" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────────────
# VERIFY POETRY & INSTALL
# ─────────────────────────────────────────────────────────────────────────────────
Write-Host "[4/4] Setting up Poetry environment..." -ForegroundColor Yellow

$poetryCmd = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetryCmd) {
    Write-Host "  ⚠ Poetry not found. Install with: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -" -ForegroundColor Red
    exit 1
}

Set-Location $ProjectRoot
poetry config virtualenvs.in-project true
poetry install --no-root

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " ✓ Project initialized successfully!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host " Next steps:" -ForegroundColor White
Write-Host "   cd $ProjectRoot" -ForegroundColor Gray
Write-Host "   poetry shell" -ForegroundColor Gray
Write-Host "   pytest" -ForegroundColor Gray
Write-Host ""
