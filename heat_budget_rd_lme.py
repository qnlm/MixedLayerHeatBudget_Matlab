#!/usr/bin/env python3
"""
Compute the Reynolds-decomposed mixed-layer heat budget for CESM Last
Millennium Ensemble (LME) simulations.

Python translation of heat_budget_rd_lme.m
Original: March 2015 / Sam Stevenson
Uses the formulation of Graham et al.: Climate Dynamics (2014) 43:2399-2414.

Runtime configuration is read from environment variables set by run.bash
(see that file for full documentation of every variable).

Three file-layout options are supported (DATA_FORMAT_OPTION):
  1 – all variables in one file per run/date
  2 – one variable per file, CESM standard monthly time-series layout
  3 – one NetCDF file per calendar month
"""

import os
import numpy as np
import numpy.ma as ma
import netCDF4 as nc4
import cftime

from mldavg_varytime import mldavg_varytime
from submld_varytime import submld_varytime
from advection_ml_rd import advection_ml_rd
from vertadv_ml_rd import vertadv_ml_rd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
rho    = 1025.0   # Mean density of seawater at 20 °C, 35 psu  [kg/m³]
cp     = 3993.0   # Specific heat of seawater at 20 °C, 35 psu [J/kg/K]
rho_cp = rho * cp

# ---------------------------------------------------------------------------
# Configuration – all values read from environment variables.
# Set these in the calling shell (run.bash) before invoking this script.
# This script processes exactly one run/date combination per invocation;
# all looping over ensembles, members, and date chunks is done in run.bash.
# ---------------------------------------------------------------------------
DATA_FORMAT_OPTION = int(os.environ.get('DATA_FORMAT_OPTION', '2'))
BASE_DIR  = os.environ['BASE_DIR']
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
ensname = os.environ.get('ENSNAME', '')   # optional label for display only


# ---------------------------------------------------------------------------
# Helper: read a NetCDF variable and return a float64 array with fill→NaN
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
# Helper: clean fill values  (CESM land/mask fill is ≈ 1e36)
# ---------------------------------------------------------------------------
def _clean(arr, threshold=1e30):
    """Replace values with |x| > threshold with NaN in-place."""
    arr[np.abs(arr) > threshold] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Helper: parse a date string "YYYYMM-YYYYMM" into start/end year and month
# ---------------------------------------------------------------------------
def _parse_date_str(date_str):
    yr_start = int(date_str[0:4])
    mo_start = int(date_str[4:6])
    yr_end   = int(date_str[7:11])
    mo_end   = int(date_str[11:13])
    return yr_start, mo_start, yr_end, mo_end


# ---------------------------------------------------------------------------
# Helper: write all budget variables to a NetCDF4 file
# ---------------------------------------------------------------------------
def write_budget_nc(filename, data, time_vals):
    """
    Write all budget variables to a NetCDF4 file.

    Parameters
    ----------
    filename  : path to output file (overwritten if it exists)
    data      : dict of arrays; must contain 'tlat' and 'tlon' (nlat × nlon)
    time_vals : 1-D array of raw time values (days since …)
    """
    nt, nlat, nlon = data['Tmld'].shape

    if os.path.exists(filename):
        os.remove(filename)

    with nc4.Dataset(filename, 'w', format='NETCDF4') as ds:
        # Dimensions
        ds.createDimension('time',  None)   # unlimited
        ds.createDimension('lat',   nlat)
        ds.createDimension('lon',   nlon)
        ds.createDimension('month', 12)

        # Coordinate: time
        tv          = ds.createVariable('time', 'f8', ('time',))
        tv[:]       = time_vals
        tv.units    = 'days since 0000-01-01 00:00:00'
        tv.calendar = 'noleap'

        # Coordinate: month
        mv    = ds.createVariable('month', 'i4', ('month',))
        mv[:] = np.arange(1, 13)

        # Grid
        latv       = ds.createVariable('TLAT',  'f8', ('lat', 'lon'))
        latv[:]    = data['tlat']
        latv.units = 'degrees_north'

        lonv       = ds.createVariable('TLONG', 'f8', ('lat', 'lon'))
        lonv[:]    = data['tlon']
        lonv.units = 'degrees_east'

        # All other variables — choose dimensions by shape
        for vname, arr in data.items():
            if vname in ('tlat', 'tlon'):
                continue
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape == (nt, nlat, nlon):
                v = ds.createVariable(vname, 'f8', ('time', 'lat', 'lon'),
                                      fill_value=np.nan)
                v[:] = arr
            elif arr.ndim == 3 and arr.shape == (12, nlat, nlon):
                v = ds.createVariable(vname, 'f8', ('month', 'lat', 'lon'),
                                      fill_value=np.nan)
                v[:] = arr
            elif arr.shape == (nlat, nlon):
                v = ds.createVariable(vname, 'f8', ('lat', 'lon'),
                                      fill_value=np.nan)
                v[:] = arr
            else:
                print(f'  [write_budget_nc] Warning: {vname} shape {arr.shape} '
                      f'unrecognised, skipped.')


# ---------------------------------------------------------------------------
# Read grid (lat/lon) information
# ---------------------------------------------------------------------------
print(f"Reading grid from: {GRID_FILE}")
with nc4.Dataset(GRID_FILE) as nc:
    tlat_full = _read_var(nc, 'TLAT')   # (nlat, nlon)
    tlon_full = _read_var(nc, 'TLONG')  # (nlat, nlon)

# Identify the row/column indices that fall within the region of interest.
# REF_COL (0-indexed) is used as the reference row/column for band selection.
# A small epsilon is added to the bounds to match MATLAB's inclusive boundary
# behaviour: float32 grid values may differ from the exact bound by a tiny
# amount after float32→float64 conversion, which can cause Python to exclude
# one boundary row/column that MATLAB includes.
boundary_tolerance = 1e-6
mylat = np.where(
    (tlat_full[:, REF_COL] >= regbox[0] - boundary_tolerance) & (tlat_full[:, REF_COL] <= regbox[1] + boundary_tolerance)
)[0]
mylon = np.where(
    (tlon_full[REF_COL, :] >= regbox[2] - boundary_tolerance) & (tlon_full[REF_COL, :] <= regbox[3] + boundary_tolerance)
)[0]

tlat = tlat_full[np.ix_(mylat, mylon)]
tlon = tlon_full[np.ix_(mylat, mylon)]

# ---------------------------------------------------------------------------
# Main computation for runname / date
# ---------------------------------------------------------------------------
print(f"\n=== Ensemble: {ensname}  |  Run: {runname}  |  Date: {date} ===")

# ----------------------------------------------------------------
# Data reading — supports three file-layout options
# ----------------------------------------------------------------
if DATA_FORMAT_OPTION == 1:
    # Format 1: all variables in one file per run/date
    filepath = os.path.join(BASE_DIR, f'{runname}.pop.h.all.{date}.nc')
    with nc4.Dataset(filepath) as nc:
        time_vals, yr, mon = _decode_time(nc)
        z    = _read_var(nc, 'z_t', slice(NZ)) / 100.0
        temp = _read_var(nc, 'TEMP',    slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon]
        uvel = _read_var(nc, 'UVEL',    slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon] / 100.0
        vvel = _read_var(nc, 'VVEL',    slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon] / 100.0
        wvel = _read_var(nc, 'WVEL',    slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon] / 100.0
        qnet = _read_var(nc, 'SHF',     slice(None), mylat, slice(None))[:, :, mylon]
        qsw  = _read_var(nc, 'SHF_QSW', slice(None), mylat, slice(None))[:, :, mylon]
        mld  = _read_var(nc, 'HMXL',    slice(None), mylat, slice(None))[:, :, mylon] / 100.0

elif DATA_FORMAT_OPTION == 2:
    # Format 2: one variable per file (CESM standard monthly time-series layout)
    def _fpath(var):
        return os.path.join(BASE_DIR, var, f'{runname}.pop.h.{var}.{date}.nc')

    with nc4.Dataset(_fpath('TEMP')) as nc:
        time_vals, yr, mon = _decode_time(nc)
        z    = _read_var(nc, 'z_t', slice(NZ)) / 100.0
        temp = _read_var(nc, 'TEMP', slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon]
    with nc4.Dataset(_fpath('UVEL')) as nc:
        uvel = _read_var(nc, 'UVEL', slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon] / 100.0
    with nc4.Dataset(_fpath('VVEL')) as nc:
        vvel = _read_var(nc, 'VVEL', slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon] / 100.0
    with nc4.Dataset(_fpath('WVEL')) as nc:
        wvel = _read_var(nc, 'WVEL', slice(None), slice(NZ), mylat, slice(None))[:, :, :, mylon] / 100.0
    with nc4.Dataset(_fpath('SHF')) as nc:
        qnet = _read_var(nc, 'SHF',     slice(None), mylat, slice(None))[:, :, mylon]
    with nc4.Dataset(_fpath('SHF_QSW')) as nc:
        qsw  = _read_var(nc, 'SHF_QSW', slice(None), mylat, slice(None))[:, :, mylon]
    with nc4.Dataset(_fpath('HMXL')) as nc:
        mld  = _read_var(nc, 'HMXL',    slice(None), mylat, slice(None))[:, :, mylon] / 100.0

elif DATA_FORMAT_OPTION == 3:
    # Format 3: one NetCDF file per calendar month (runname.pop.h.YYYY-MM.nc)
    yr_start, mo_start, yr_end, mo_end = _parse_date_str(date)
    n_months = (yr_end - yr_start) * 12 + (mo_end - mo_start) + 1

    time_vals = np.zeros(n_months)
    yr        = np.zeros(n_months, dtype=int)
    mon       = np.zeros(n_months, dtype=int)
    z         = None

    count = 0
    for y in range(yr_start, yr_end + 1):
        for m in range(1, 13):
            if (y == yr_start and m < mo_start) or \
               (y == yr_end   and m > mo_end):
                continue
            fpath = os.path.join(BASE_DIR,
                                 f'{runname}.pop.h.{y:04d}-{m:02d}.nc')
            with nc4.Dataset(fpath) as nc:
                if z is None:
                    z      = _read_var(nc, 'z_t', slice(NZ)) / 100.0
                    nlat_r = len(mylat)
                    nlon_r = len(mylon)
                    temp = np.zeros((n_months, NZ,   nlat_r, nlon_r))
                    uvel = np.zeros_like(temp)
                    vvel = np.zeros_like(temp)
                    wvel = np.zeros_like(temp)
                    qnet = np.zeros((n_months, nlat_r, nlon_r))
                    qsw  = np.zeros_like(qnet)
                    mld  = np.zeros_like(qnet)
                time_vals[count] = _read_var(nc, 'time')[0]
                temp[count] = _read_var(nc, 'TEMP',    0, slice(NZ), mylat, slice(None))[:, :, mylon]
                uvel[count] = _read_var(nc, 'UVEL',    0, slice(NZ), mylat, slice(None))[:, :, mylon] / 100.0
                vvel[count] = _read_var(nc, 'VVEL',    0, slice(NZ), mylat, slice(None))[:, :, mylon] / 100.0
                wvel[count] = _read_var(nc, 'WVEL',    0, slice(NZ), mylat, slice(None))[:, :, mylon] / 100.0
                qnet[count] = _read_var(nc, 'SHF',     0, mylat, slice(None))[:, mylon]
                qsw[count]  = _read_var(nc, 'SHF_QSW', 0, mylat, slice(None))[:, mylon]
                mld[count]  = _read_var(nc, 'HMXL',    0, mylat, slice(None))[:, mylon] / 100.0
            yr[count]  = y
            mon[count] = m
            count += 1

else:
    raise ValueError(f'Unknown DATA_FORMAT_OPTION: {DATA_FORMAT_OPTION}')

# ---- Fill-value cleanup  (CESM land/mask fill ≈ 1e36) ---------------------
_clean(temp); _clean(uvel); _clean(vvel); _clean(wvel)
_clean(qnet); _clean(qsw);  _clean(mld)

# ---- Surface heat flux term  (sfcflx) — vectorised ------------------------
# Penetrative shortwave: two-band approximation (Paulson & Simpson 1977).
# 0.58 / 0.35 m: visible band fraction / e-folding depth
# 0.42 / 23.0 m: near-IR band fraction / e-folding depth
qpen   = qsw * (0.58 * np.exp(-mld / 0.35)
                + 0.42 * np.exp(-mld / 23.0))
sfcflx = (qnet - qpen) / (rho_cp * mld)

# ---- Mixed-layer averages / sub-MLD values ---------------------------------
nt, nlat_r, nlon_r = mld.shape
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

# dTdt from advection_ml_rd is in K/day → convert to K/s
dTdt = dTdt / 86400.0

# ---- Vertical advection (Reynolds decomposition) ---------------------------
(w_entr, wmdTmdz, wpdTmdz, wmdTpdz, wpdTpdz,
 mnwpdTpdz) = vertadv_ml_rd(
    time_vals, tlat, tlon, mld, usub, vsub,
    Tmld, Tsub, wsub, mon, yr, yrclim)

# Free large arrays to reduce memory footprint
del temp, uvel, vvel, wvel, qnet, qsw, vmld

# ---- Save results as NetCDF ------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
nc_path = os.path.join(OUT_DIR,
                       f'{runname}_heatbudget_ml_rd_{date}_varyMLD.nc')

out_data = {
    'tlat':  tlat,   'tlon': tlon,
    'mld':   mld,    'Tmld': Tmld,   'Tsub': Tsub,
    'umld':  umld,   'usub': usub,   'vsub': vsub,   'wsub': wsub,
    'sfcflx': sfcflx, 'qpen': qpen,
    'dTdt':  dTdt,
    'umdTmdx': umdTmdx, 'updTmdx': updTmdx,
    'umdTpdx': umdTpdx, 'updTpdx': updTpdx,
    'vmdTmdy': vmdTmdy, 'vpdTmdy': vpdTmdy,
    'vmdTpdy': vmdTpdy, 'vpdTpdy': vpdTpdy,
    'mnupdTpdx': mnupdTpdx, 'mnvpdTpdy': mnvpdTpdy,
    'w_entr':    w_entr,
    'wmdTmdz':   wmdTmdz,   'wpdTmdz': wpdTmdz,
    'wmdTpdz':   wmdTpdz,   'wpdTpdz': wpdTpdz,
    'mnwpdTpdz': mnwpdTpdz,
}

write_budget_nc(nc_path, out_data, time_vals)
print(f"Saved: {nc_path}")
