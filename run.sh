#!/usr/bin/env bash
# =============================================================================
# run.sh  –  Manage and execute the mixed-layer heat budget computation
#
# Usage:
#   ./run.sh              # Run with defaults
#   ./run.sh --check-only # Only verify the environment, do not run
#   ./run.sh --help       # Show this help
#
# The script:
#   1. Checks for required Python 3 interpreter
#   2. Installs missing Python packages (numpy, netCDF4, cftime, scipy)
#   3. Creates the output directory if it does not exist
#   4. Executes heat_budget_rd_lme.py and logs stdout/stderr
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/heat_budget_rd_lme.py"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/heat_budget_$(date +%Y%m%d_%H%M%S).log"
OUT_DIR="/glade/p/cesm/palwg_dev/LME/proc/samantha/LME_heatbudget"
REQUIRED_PKGS=(numpy netCDF4 cftime scipy)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
CHECK_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=true ;;
        --help|-h)    usage ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Verify Python 3
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.8 or later."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
info "Python version: ${PYTHON_VERSION}"

# ---------------------------------------------------------------------------
# 2. Check / install required packages
# ---------------------------------------------------------------------------
info "Checking Python dependencies: ${REQUIRED_PKGS[*]}"

MISSING=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! python3 -c "import ${pkg}" &>/dev/null; then
        MISSING+=("${pkg}")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Missing packages: ${MISSING[*]}"
    info "Installing missing packages via pip …"
    python3 -m pip install --quiet "${MISSING[@]}" \
        || error "pip install failed. Check your network / permissions."
    info "Installation complete."
else
    info "All dependencies satisfied."
fi

# ---------------------------------------------------------------------------
# 3. Verify the main script exists
# ---------------------------------------------------------------------------
[[ -f "${MAIN_SCRIPT}" ]] || error "Main script not found: ${MAIN_SCRIPT}"

# ---------------------------------------------------------------------------
# 4. Create output / log directories
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"

if [[ -d "$(dirname "${OUT_DIR}")" ]]; then
    mkdir -p "${OUT_DIR}"
    info "Output directory: ${OUT_DIR}"
else
    warn "Parent of output directory does not exist: $(dirname "${OUT_DIR}")"
    warn "The script will attempt to create it at runtime, or may fail if"
    warn "you are not on the NCAR GLADE filesystem."
fi

# ---------------------------------------------------------------------------
# 5. Optionally stop here
# ---------------------------------------------------------------------------
if ${CHECK_ONLY}; then
    info "Environment check complete (--check-only). Exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# 6. Run the main Python script
# ---------------------------------------------------------------------------
info "Starting heat budget computation …"
info "Log file: ${LOG_FILE}"

cd "${SCRIPT_DIR}"

# Tee stdout+stderr to both the terminal and the log file
python3 "${MAIN_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}

if [[ ${STATUS} -eq 0 ]]; then
    info "Computation finished successfully."
else
    error "Computation exited with status ${STATUS}. See ${LOG_FILE} for details."
fi
