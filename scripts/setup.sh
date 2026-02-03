#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ALGO TRADING BOT - PROJECT SETUP SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

set -e

PROJECT_NAME="algo_trading_bot"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo " Initializing: ${PROJECT_NAME}"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────────
# CREATE DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────────
echo "[1/4] Creating directory structure..."

DIRS=(
    "src/data"
    "src/analysis"
    "src/execution"
    "src/models"
    "src/utils"
    "tests"
    "scripts"
    "notebooks"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "${PROJECT_ROOT}/${dir}"
    echo "  ✓ Created ${dir}/"
done

# ─────────────────────────────────────────────────────────────────────────────────
# CREATE __init__.py FILES
# ─────────────────────────────────────────────────────────────────────────────────
echo "[2/4] Creating Python package files..."

INIT_FILES=(
    "src/__init__.py"
    "src/data/__init__.py"
    "src/analysis/__init__.py"
    "src/execution/__init__.py"
    "src/models/__init__.py"
    "src/utils/__init__.py"
    "tests/__init__.py"
)

for init in "${INIT_FILES[@]}"; do
    touch "${PROJECT_ROOT}/${init}"
    echo "  ✓ Created ${init}"
done

# ─────────────────────────────────────────────────────────────────────────────────
# CREATE .gitkeep FILES
# ─────────────────────────────────────────────────────────────────────────────────
echo "[3/4] Creating placeholder files..."

GITKEEP_DIRS=(
    "scripts"
    "notebooks"
    "src/models"
)

for dir in "${GITKEEP_DIRS[@]}"; do
    touch "${PROJECT_ROOT}/${dir}/.gitkeep"
    echo "  ✓ Created ${dir}/.gitkeep"
done

# ─────────────────────────────────────────────────────────────────────────────────
# VERIFY POETRY & INSTALL
# ─────────────────────────────────────────────────────────────────────────────────
echo "[4/4] Setting up Poetry environment..."

if ! command -v poetry &> /dev/null; then
    echo "  ⚠ Poetry not found. Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

cd "${PROJECT_ROOT}"
poetry config virtualenvs.in-project true
poetry install --no-root

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo " ✓ Project initialized successfully!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo " Next steps:"
echo "   cd ${PROJECT_ROOT}"
echo "   poetry shell"
echo "   pytest"
echo ""
