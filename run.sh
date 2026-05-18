#!/usr/bin/env bash
# =============================================================================
# run.sh  –  Manage and execute the mixed-layer heat budget computation
#
# Usage:
#   ./run.sh              # Run with defaults
#   ./run.sh --check-only # Only verify the environment, do not run
#   ./run.sh --help       # Show this help
#
# All configuration is managed here as environment variables.  Override any
# variable before calling this script, e.g.:
#
#   BASE_DIR=/my/data OUT_DIR=/my/output ./run.sh
#
# ---- Path / layout configuration ------------------------------------------
#   BASE_DIR            Root of the model archive (contains run-name sub-dirs)
#                       Default: /data4/liuzedong/archive2
#
#   OUT_DIR             Directory where result NetCDF files are written
#                       Default: /lustre/home/liuzedong/fw/heatbudget
#
#   GRID_FILE           NetCDF file used to read TLAT / TLONG
#                       Default: <BASE_DIR>/b.e10.B1850CN.T31_g37.001.rest4101/
#                                ocn/hist/b.e10.B1850CN.T31_g37.001.rest4101.pop.h.5200-01.nc
#
# ---- Data reading options --------------------------------------------------
#   DATA_FORMAT_OPTION  1 = all variables in one file per run/date
#                       2 = one variable per file (CESM monthly time series)
#                       3 = one file per month  (default)
#
# ---- Grid / domain options -------------------------------------------------
#   REGBOX              Space-separated "lat_min lat_max lon_min lon_max"
#                       Default: -10 10 190 240
#
#   NZ                  Number of depth levels to use
#                       Default: 20
#
#   REF_COL             1-indexed reference column used when subsetting
#                       lat/lon bands from the 2-D POP grid
#                       Default: 50
#
# ---- Ensemble / run / date configuration -----------------------------------
#   ENSNAMES_JSON       JSON array of ensemble names
#                       Default: ["CTRL"]
#
#   RUNNAMES_JSON       JSON array-of-arrays of run names (outer = ensemble,
#                       inner = members; use "" for unused slots)
#                       Default: [["b.e10.B1850CN.T31_g37.001.rest4101"]]
#
#   DATES_JSON          JSON array of date-range strings "YYYYMM-YYYYMM"
#                       Default: ["520001-530012"]
#
#   NOTE: when overriding JSON variables from the command line, wrap the value
#   in single quotes to avoid shell interpretation of the double quotes, e.g.:
#     ENSNAMES_JSON='["CTRL","FWFIX"]' ./run.sh
#
# The script:
#   1. Checks for required Python 3 interpreter
#   2. Installs missing Python packages (numpy, netCDF4, cftime)
#   3. Creates the output directory if it does not exist
#   4. Exports all variables so heat_budget_rd_lme.py can read them
#   5. Executes heat_budget_rd_lme.py and logs stdout/stderr
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Script location
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/heat_budget_rd_lme.py"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/heat_budget_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# Global configuration  (all can be overridden by the caller's environment)
# ---------------------------------------------------------------------------
export BASE_DIR="${BASE_DIR:-/data4/liuzedong/archive2}"
export OUT_DIR="${OUT_DIR:-/lustre/home/liuzedong/fw/heatbudget}"
export GRID_FILE="${GRID_FILE:-${BASE_DIR}/b.e10.B1850CN.T31_g37.001.rest4101/ocn/hist/b.e10.B1850CN.T31_g37.001.rest4101.pop.h.5200-01.nc}"

export DATA_FORMAT_OPTION="${DATA_FORMAT_OPTION:-3}"
export REGBOX="${REGBOX:--10 10 190 240}"
export NZ="${NZ:-20}"
export REF_COL="${REF_COL:-50}"

export ENSNAMES_JSON="${ENSNAMES_JSON:-[\"CTRL\"]}"
export RUNNAMES_JSON="${RUNNAMES_JSON:-[[\"b.e10.B1850CN.T31_g37.001.rest4101\"]]}"
export DATES_JSON="${DATES_JSON:-[\"520001-530012\"]}"

# HDF5 file locking can cause problems on shared/lustre filesystems
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

REQUIRED_PKGS=(numpy netCDF4 cftime)

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
# 3. Print active configuration
# ---------------------------------------------------------------------------
info "--- Configuration ---"
info "DATA_FORMAT_OPTION : ${DATA_FORMAT_OPTION}"
info "BASE_DIR           : ${BASE_DIR}"
info "OUT_DIR            : ${OUT_DIR}"
info "GRID_FILE          : ${GRID_FILE}"
info "REGBOX             : ${REGBOX}"
info "NZ                 : ${NZ}"
info "REF_COL            : ${REF_COL}"
info "ENSNAMES_JSON      : ${ENSNAMES_JSON}"
info "RUNNAMES_JSON      : ${RUNNAMES_JSON}"
info "DATES_JSON         : ${DATES_JSON}"
info "---------------------"

# ---------------------------------------------------------------------------
# 4. Verify the main script exists
# ---------------------------------------------------------------------------
[[ -f "${MAIN_SCRIPT}" ]] || error "Main script not found: ${MAIN_SCRIPT}"

# ---------------------------------------------------------------------------
# 5. Create output / log directories
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"

if [[ -d "$(dirname "${OUT_DIR}")" ]]; then
    mkdir -p "${OUT_DIR}"
    info "Output directory: ${OUT_DIR}"
else
    warn "Parent of output directory does not exist: $(dirname "${OUT_DIR}")"
    warn "The script will attempt to create it at runtime."
fi

# ---------------------------------------------------------------------------
# 6. Optionally stop here
# ---------------------------------------------------------------------------
if ${CHECK_ONLY}; then
    info "Environment check complete (--check-only). Exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# 7. Run the main Python script
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
