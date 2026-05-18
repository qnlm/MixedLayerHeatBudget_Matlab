#!/usr/bin/env bash
# =============================================================================
# run.sh  –  Manage and execute the mixed-layer heat budget computation
#
# Usage:
#   ./run.sh              # Run with defaults
#   ./run.sh --check-only # Only verify the environment, do not run
#   ./run.sh --help       # Show this help
#
# All configuration for heat_budget_rd_lme.py is declared here as exported
# environment variables.  Edit this file to change paths, regions, ensembles,
# run names, or date ranges – the Python script itself requires no edits.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration – all values forwarded to heat_budget_rd_lme.py
# ---------------------------------------------------------------------------

# Base directory containing per-variable time-series NetCDF files.
export BASE_PATH="/glade/p/cesm0005/CESM-CAM5-LME/ocn/proc/tseries/monthly"

# Directory where output .mat files will be written.
export OUT_DIR="/glade/p/cesm/palwg_dev/LME/proc/samantha/LME_heatbudget"

# NetCDF file used to read the grid (TLAT / TLONG).
export GRID_FILE="/glade/scratch/samantha/b40.1850.track1.1deg.006/b40.1850.track1.1deg.006.pop.h.HMXL.080001-089912.nc"

# Number of vertical levels to read from each 3-D variable.
export NZ="20"

# 1-indexed reference column used to select the lat/lon bands (MATLAB convention).
export REF_COL="150"

# Region of interest: "lat_min lat_max lon_min lon_max".
export REGBOX="-10 10 90 300"

# Ensemble loop bounds (0-indexed, Python range convention: [EE_START, EE_END) ).
# Default processes only ensemble index 5 (Solar), member index 4 – the same
# single-member subset that matched the original MATLAB script (ee=6, rr=5 in
# 1-indexed notation).  Set EE_START=0 / EE_END=8 to process all ensembles.
export EE_START="5"
export EE_END="6"

# Member loop bounds (0-indexed, Python range convention: [RR_START, RR_END) ).
export RR_START="4"
export RR_END="5"

# Ensemble labels (JSON array).
export ENSNAMES_JSON='["Full","GHG","LULC","Orbital","Solar","Volcanic","OzoneAer","Control"]'

# Run names per ensemble (JSON array-of-arrays; use "" for empty placeholders).
export RUNNAMES_JSON='[
  ["b.e11.BLMTRC5CN.f19_g16.001","b.e11.BLMTRC5CN.f19_g16.002",
   "b.e11.BLMTRC5CN.f19_g16.003","b.e11.BLMTRC5CN.f19_g16.004",
   "b.e11.BLMTRC5CN.f19_g16.005","b.e11.BLMTRC5CN.f19_g16.006",
   "b.e11.BLMTRC5CN.f19_g16.007","b.e11.BLMTRC5CN.f19_g16.008",
   "b.e11.BLMTRC5CN.f19_g16.009","b.e11.BLMTRC5CN.f19_g16.010",
   "b.e11.BLMTRC5CN.f19_g16.011","b.e11.BLMTRC5CN.f19_g16.012",
   "b.e11.BLMTRC5CN.f19_g16.013"],
  ["b.e11.BLMTRC5CN.f19_g16.GHG.001","b.e11.BLMTRC5CN.f19_g16.GHG.002",
   "b.e11.BLMTRC5CN.f19_g16.GHG.003","","","","","","","","","",""],
  ["b.e11.BLMTRC5CN.f19_g16.LULC_HurttPongratz.001",
   "b.e11.BLMTRC5CN.f19_g16.LULC_HurttPongratz.002",
   "b.e11.BLMTRC5CN.f19_g16.LULC_HurttPongratz.003",
   "","","","","","","","","",""],
  ["b.e11.BLMTRC5CN.f19_g16.ORBITAL.001","b.e11.BLMTRC5CN.f19_g16.ORBITAL.002",
   "b.e11.BLMTRC5CN.f19_g16.ORBITAL.003","","","","","","","","","",""],
  ["b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.001","b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.003",
   "b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.004","b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.005",
   "","","","","","","","",""],
  ["b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.001","b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.002",
   "b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.003","b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.004",
   "b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.005","","","","","","","",""],
  ["b.e11.BLMTRC5CN.f19_g16.OZONE_AER.001","b.e11.BLMTRC5CN.f19_g16.OZONE_AER.002",
   "b.e11.BLMTRC5CN.f19_g16.OZONE_AER.003","b.e11.BLMTRC5CN.f19_g16.OZONE_AER.004",
   "b.e11.BLMTRC5CN.f19_g16.OZONE_AER.005","","","","","","","",""],
  ["b.e11.B1850C5CN.f19_g16.0850cntl.001","","","","","","","","","","","",""]
]'

# Date range chunks to process (JSON array).
export DATES_JSON='["085001-089912","090001-099912","100001-109912","110001-119912",
 "120001-129912","130001-139912","140001-149912","150001-159912",
 "160001-169912","170001-179912","180001-184912","185001-200512"]'

# ---------------------------------------------------------------------------
# Internal script settings
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/heat_budget_rd_lme.py"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/heat_budget_$(date +%Y%m%d_%H%M%S).log"
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
