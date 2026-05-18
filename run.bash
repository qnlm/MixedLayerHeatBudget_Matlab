#!/usr/bin/env bash
# =============================================================================
# run.bash - Manage and execute the mixed-layer heat budget computation
#
# Usage:
#   ./run.bash              # Run with defaults
#   ./run.bash --check-only # Only verify the environment, do not run
#   ./run.bash --help       # Show this help
#
# All configuration is declared in this file.  The Python script
# heat_budget_rd_lme.py processes exactly one run/date combination per
# invocation; this script loops over all ensembles, members, and date
# chunks and calls Python once per combination.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration – edit this section to change paths, regions, or run lists
# ---------------------------------------------------------------------------

# File-layout option for reading ocean history files:
#   1 – all variables in one file per run/date
#   2 – one variable per file, CESM standard monthly time-series layout
#   3 – one NetCDF file per calendar month
export DATA_FORMAT_OPTION="2"

# Base directory for input ocean history files.
# For option 2: directory containing per-variable sub-directories
#               (e.g. BASE_DIR/TEMP/runname.pop.h.TEMP.date.nc)
# For option 1: directory containing combined files
#               (e.g. BASE_DIR/runname.pop.h.all.date.nc)
# For option 3: directory containing per-month files
#               (e.g. BASE_DIR/runname.pop.h.YYYY-MM.nc)
export BASE_DIR="/glade/p/cesm0005/CESM-CAM5-LME/ocn/proc/tseries/monthly"

# Directory where output NetCDF files will be written.
export OUT_DIR="/glade/p/cesm/palwg_dev/LME/proc/samantha/LME_heatbudget"

# NetCDF file used to read the grid (TLAT / TLONG).
export GRID_FILE="/glade/scratch/samantha/b40.1850.track1.1deg.006/b40.1850.track1.1deg.006.pop.h.HMXL.080001-089912.nc"

# Number of vertical levels to read from each 3-D variable.
export NZ="20"

# 1-indexed reference column used to select the lat/lon bands (MATLAB convention).
export REF_COL="150"

# Region of interest: "lat_min lat_max lon_min lon_max".
export REGBOX="-10 10 90 300"

# ---------------------------------------------------------------------------
# Ensemble / run name definitions
# Each ensemble has a matching runs_<EnsName> array.  Use "" as a placeholder
# for missing members so the index layout is preserved.
# ---------------------------------------------------------------------------
ensnames=("Full" "GHG" "LULC" "Orbital" "Solar" "Volcanic" "OzoneAer" "Control")

runs_Full=(
    "b.e11.BLMTRC5CN.f19_g16.001" "b.e11.BLMTRC5CN.f19_g16.002"
    "b.e11.BLMTRC5CN.f19_g16.003" "b.e11.BLMTRC5CN.f19_g16.004"
    "b.e11.BLMTRC5CN.f19_g16.005" "b.e11.BLMTRC5CN.f19_g16.006"
    "b.e11.BLMTRC5CN.f19_g16.007" "b.e11.BLMTRC5CN.f19_g16.008"
    "b.e11.BLMTRC5CN.f19_g16.009" "b.e11.BLMTRC5CN.f19_g16.010"
    "b.e11.BLMTRC5CN.f19_g16.011" "b.e11.BLMTRC5CN.f19_g16.012"
    "b.e11.BLMTRC5CN.f19_g16.013"
)
runs_GHG=(
    "b.e11.BLMTRC5CN.f19_g16.GHG.001" "b.e11.BLMTRC5CN.f19_g16.GHG.002"
    "b.e11.BLMTRC5CN.f19_g16.GHG.003"
)
runs_LULC=(
    "b.e11.BLMTRC5CN.f19_g16.LULC_HurttPongratz.001"
    "b.e11.BLMTRC5CN.f19_g16.LULC_HurttPongratz.002"
    "b.e11.BLMTRC5CN.f19_g16.LULC_HurttPongratz.003"
)
runs_Orbital=(
    "b.e11.BLMTRC5CN.f19_g16.ORBITAL.001" "b.e11.BLMTRC5CN.f19_g16.ORBITAL.002"
    "b.e11.BLMTRC5CN.f19_g16.ORBITAL.003"
)
runs_Solar=(
    "b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.001" "b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.003"
    "b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.004" "b.e11.BLMTRC5CN.f19_g16.SSI_VSK_L.005"
)
runs_Volcanic=(
    "b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.001" "b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.002"
    "b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.003" "b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.004"
    "b.e11.BLMTRC5CN.f19_g16.VOLC_GRA.005"
)
runs_OzoneAer=(
    "b.e11.BLMTRC5CN.f19_g16.OZONE_AER.001" "b.e11.BLMTRC5CN.f19_g16.OZONE_AER.002"
    "b.e11.BLMTRC5CN.f19_g16.OZONE_AER.003" "b.e11.BLMTRC5CN.f19_g16.OZONE_AER.004"
    "b.e11.BLMTRC5CN.f19_g16.OZONE_AER.005"
)
runs_Control=(
    "b.e11.B1850C5CN.f19_g16.0850cntl.001"
)

# Date range chunks to process.
dates=(
    "085001-089912" "090001-099912" "100001-109912" "110001-119912"
    "120001-129912" "130001-139912" "140001-149912" "150001-159912"
    "160001-169912" "170001-179912" "180001-184912" "185001-200512"
)

# ---------------------------------------------------------------------------
# Internal script settings
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/heat_budget_rd_lme.py"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/heat_budget_$(date +%Y%m%d_%H%M%S).log"
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
# 6. Loop over ensembles → members → date chunks, invoking Python once each
# ---------------------------------------------------------------------------
info "Starting heat budget computation …"
info "Log file: ${LOG_FILE}"

cd "${SCRIPT_DIR}"

# Redirect all subsequent stdout+stderr to both terminal and the log file.
# This opens the log once rather than on every Python invocation.
exec > >(tee -a "${LOG_FILE}") 2>&1

for ensname in "${ensnames[@]}"; do
    runs_var="runs_${ensname}[@]"
    for runname in "${!runs_var}"; do
        [[ -z "${runname}" ]] && continue
        for date in "${dates[@]}"; do
            info "Ensemble: ${ensname}  |  Run: ${runname}  |  Date: ${date}"
            export ENSNAME="${ensname}"
            export RUNNAME="${runname}"
            export DATE="${date}"
            python3 "${MAIN_SCRIPT}"
            STATUS=$?
            if [[ ${STATUS} -ne 0 ]]; then
                error "Python exited with status ${STATUS}. See ${LOG_FILE}."
            fi
        done
    done
done

info "All computations finished successfully."
