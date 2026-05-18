#!/usr/bin/env python3
"""
Compute the Reynolds-decomposed mixed-layer heat budget for CESM Last
Millennium Ensemble (LME) simulations.

Python translation of heat_budget_rd_lme.m
Original: March 2015 / Sam Stevenson
Uses the formulation of Graham et al.: Climate Dynamics (2014) 43:2399-2414.
"""

import os
import numpy as np
import numpy.ma as ma
import netCDF4 as nc4
import cftime
import scipy.io

from mldavg_varytime import mldavg_varytime
from submld_varytime import submld_varytime
from advection_ml_rd import advection_ml_rd
from vertadv_ml_rd import vertadv_ml_rd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
rho = 1025.0   # Mean density of seawater at 20 °C, 35 psu  [kg/m³]
cp  = 3993.0   # Specific heat of seawater at 20 °C, 35 psu [J/kg/K]

# ---------------------------------------------------------------------------
# Configuration – all values read from environment variables.
# Set these in the calling shell (e.g. run.sh) before invoking this script.
# This script processes exactly one run/date combination per invocation;
# all looping over ensembles, members, and date chunks is done in run.sh.
# ---------------------------------------------------------------------------
BASE_PATH = os.environ['BASE_PATH']
OUT_DIR   = os.environ['OUT_DIR']
GRID_FILE = os.environ['GRID_FILE']

# Number of vertical levels to read
NZ = int(os.environ['NZ'])

# REF_COL: 1-indexed reference column for lat/lon band selection (MATLAB
# convention); converted to 0-indexed here.
REF_COL = int(os.environ['REF_COL']) - 1

# Region of interest: [lat_min, lat_max, lon_min, lon_max]
regbox = [float(x) for x in os.environ['REGBOX'].split()]

# Single run name and date chunk to process (set by the outer bash loop)
runname = os.environ['RUNNAME']
date    = os.environ['DATE']
ensname = os.environ.get('ENSNAME', '')   # optional label for display only; empty string if not set


# ---------------------------------------------------------------------------
# Helper: read a NetCDF variable and return a float array with fill→NaN
# ---------------------------------------------------------------------------
def _read_var(nc, varname, *slices):
    """
    Read *varname* from open Dataset *nc*, apply *slices*, and return a
    float64 ndarray with masked / fill values replaced by NaN.
    """
    v = nc.variables[varname]
    data = v[slices] if slices else v[:]
    if isinstance(data, ma.MaskedArray):
        return ma.filled(data.astype(np.float64), np.nan)
    return np.asarray(data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper: decode the time axis to year/month arrays
# ---------------------------------------------------------------------------
def _decode_time(nc, varname='time'):
    """
    Return (time_vals, yr, mon) where time_vals are the raw float values
    (days since …), yr and mon are integer arrays decoded using cftime.
    """
    tv = nc.variables[varname]
    time_vals = np.asarray(tv[:], dtype=np.float64)
    calendar  = getattr(tv, 'calendar', 'noleap')
    decoded   = cftime.num2date(time_vals, tv.units, calendar=calendar)
    yr  = np.array([d.year  for d in decoded], dtype=int)
    mon = np.array([d.month for d in decoded], dtype=int)
    return time_vals, yr, mon


# ---------------------------------------------------------------------------
# Read grid (lat/lon) information
# ---------------------------------------------------------------------------
print(f"Reading grid from: {GRID_FILE}")
with nc4.Dataset(GRID_FILE) as nc:
    tlat_full = _read_var(nc, 'TLAT')   # (nlat, nlon)
    tlon_full = _read_var(nc, 'TLONG')  # (nlat, nlon)

# Identify the row/column indices that fall within the region of interest.
# REF_COL (0-indexed) is used as the reference row/column for band selection.
mylat = np.where(
    (tlat_full[:, REF_COL] >= regbox[0]) & (tlat_full[:, REF_COL] <= regbox[1])  # REF_COL is 0-indexed
)[0]
mylon = np.where(
    (tlon_full[REF_COL, :] >= regbox[2]) & (tlon_full[REF_COL, :] <= regbox[3])  # REF_COL is 0-indexed
)[0]

tlat = tlat_full[np.ix_(mylat, mylon)]
tlon = tlon_full[np.ix_(mylat, mylon)]

# ---------------------------------------------------------------------------
# Main computation for runname / date
# ---------------------------------------------------------------------------
print(f"\n=== Ensemble: {ensname}  |  Run: {runname}  |  Date: {date} ===")

# ---- Temperature -----------------------------------------------------------
fname = f'{BASE_PATH}/TEMP/{runname}.pop.h.TEMP.{date}.nc'
with nc4.Dataset(fname) as nc:
    temp = _read_var(nc, 'TEMP', slice(None), slice(0, NZ),
                     mylat, slice(None))[:, :, :, mylon]
    z         = np.asarray(nc.variables['z_t'][:NZ], dtype=np.float64) / 100.0  # cm→m
    time_vals, yr, mon = _decode_time(nc)

# ---- Velocities ------------------------------------------------------------
fname = f'{BASE_PATH}/UVEL/{runname}.pop.h.UVEL.{date}.nc'
with nc4.Dataset(fname) as nc:
    uvel = _read_var(nc, 'UVEL', slice(None), slice(0, NZ),
                     mylat, slice(None))[:, :, :, mylon] / 100.0  # cm/s→m/s

fname = f'{BASE_PATH}/VVEL/{runname}.pop.h.VVEL.{date}.nc'
with nc4.Dataset(fname) as nc:
    vvel = _read_var(nc, 'VVEL', slice(None), slice(0, NZ),
                     mylat, slice(None))[:, :, :, mylon] / 100.0

fname = f'{BASE_PATH}/WVEL/{runname}.pop.h.WVEL.{date}.nc'
with nc4.Dataset(fname) as nc:
    wvel = _read_var(nc, 'WVEL', slice(None), slice(0, NZ),
                     mylat, slice(None))[:, :, :, mylon] / 100.0

# ---- Heat fluxes -----------------------------------------------------------
fname = f'{BASE_PATH}/SHF/{runname}.pop.h.SHF.{date}.nc'
with nc4.Dataset(fname) as nc:
    qnet = _read_var(nc, 'SHF', slice(None), mylat,
                     slice(None))[:, :, mylon]   # W/m²

fname = f'{BASE_PATH}/SHF_QSW/{runname}.pop.h.SHF_QSW.{date}.nc'
with nc4.Dataset(fname) as nc:
    qsw = _read_var(nc, 'SHF_QSW', slice(None), mylat,
                    slice(None))[:, :, mylon]   # W/m²

# ---- Mixed-layer depth -----------------------------------------------------
fname = f'{BASE_PATH}/HMXL/{runname}.pop.h.HMXL.{date}.nc'
with nc4.Dataset(fname) as nc:
    mld = _read_var(nc, 'HMXL', slice(None), mylat,
                    slice(None))[:, :, mylon] / 100.0  # cm→m

# ---- Surface heat flux term  (sfcflx) --------------------------------------
nt, nlat_r, nlon_r = mld.shape
qpen   = np.zeros((nt, nlat_r, nlon_r))
sfcflx = np.zeros((nt, nlat_r, nlon_r))

for tt in range(nt):
    qpen[tt] = (qsw[tt]
                * (0.58 * np.exp(-mld[tt] / 0.35)
                   + 0.42 * np.exp(-mld[tt] / 23.0)))
    sfcflx[tt] = (qnet[tt] - qpen[tt]) / (rho * cp * mld[tt])

# ---- Mixed-layer averages / sub-MLD values ---------------------------------
# Broadcast z to (nz, nlat_r, nlon_r)
pacz = np.tile(z[:, np.newaxis, np.newaxis], (1, nlat_r, nlon_r))

Tmld = mldavg_varytime(mld, temp, time_vals, pacz)
Tmld[np.abs(Tmld) > 1e10] = np.nan

umld = mldavg_varytime(mld, uvel, time_vals, pacz)
vmld = mldavg_varytime(mld, vvel, time_vals, pacz)

Tsub = submld_varytime(mld, temp, time_vals, pacz, 'first')
usub = submld_varytime(mld, uvel, time_vals, pacz, 'first')
vsub = submld_varytime(mld, vvel, time_vals, pacz, 'first')
wsub = submld_varytime(mld, wvel, time_vals, pacz, 'first')

# ---- Horizontal advection (Reynolds decomposition) -------------------------
yrclim = [int(yr.min()), int(yr.max())]

(umdTmdx, updTmdx, umdTpdx, updTpdx,
 vmdTmdy, vpdTmdy, vmdTpdy, vpdTpdy,
 dTdt, mnupdTpdx, mnvpdTpdy) = advection_ml_rd(
    Tmld, umld, vmld, tlat, tlon,
    time_vals, yr, mon, yrclim)

# ---- Vertical advection (Reynolds decomposition) ---------------------------
(w_entr, wmdTmdz, wpdTmdz, wmdTpdz, wpdTpdz,
 mnwpdTpdz) = vertadv_ml_rd(
    time_vals, tlat, tlon, mld, usub, vsub,
    Tmld, Tsub, wsub, mon, yr, yrclim)

# Free large arrays to reduce memory footprint
del temp, uvel, vvel, wvel, qnet, qsw, vmld

# ---- Save results ----------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
out_file = os.path.join(
    OUT_DIR,
    f'{runname}_heatbudget_ml_rd_{date}_varyMLD.mat'
)

scipy.io.savemat(out_file, {
    'umdTmdx': umdTmdx, 'updTmdx': updTmdx,
    'umdTpdx': umdTpdx, 'updTpdx': updTpdx,
    'vmdTmdy': vmdTmdy, 'vpdTmdy': vpdTmdy,
    'vmdTpdy': vmdTpdy, 'vpdTpdy': vpdTpdy,
    'dTdt':    dTdt,
    'mnupdTpdx': mnupdTpdx, 'mnvpdTpdy': mnvpdTpdy,
    'w_entr':    w_entr,
    'wmdTmdz':   wmdTmdz,   'wpdTmdz': wpdTmdz,
    'wmdTpdz':   wmdTpdz,   'wpdTpdz': wpdTpdz,
    'mnwpdTpdz': mnwpdTpdz,
    'sfcflx': sfcflx, 'qpen': qpen,
    'mld':    mld,    'Tmld': Tmld, 'Tsub': Tsub,
    'umld':   umld,   'usub': usub, 'vsub': vsub, 'wsub': wsub,
    'tlat':   tlat,   'tlon': tlon,
    'time':   time_vals, 'yr': yr, 'mon': mon, 'z': z,
    'runname': runname,  'date': date,
})
print(f"Saved: {out_file}")
