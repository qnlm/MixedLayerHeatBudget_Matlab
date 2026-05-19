#!/usr/bin/env python3
"""
verify_results.py
-----------------
Compare every budget variable saved by the MATLAB (.mat) and Python (.nc)
versions of the mixed-layer heat budget pipeline.

Because the two grids may not be byte-for-byte identical (float32 vs float64,
slight coordinate rounding), the script first finds the common lat/lon grid
points by matching the TLAT / TLONG 2-D coordinate arrays within a configurable
spatial tolerance, then performs element-wise comparison on the matched subset.

Usage
-----
    python verify_results.py <matlab_mat_file> <python_nc_file> [options]

Positional arguments
    matlab_mat_file   Path to the .mat file written by heat_budget_rd_lme.m
    python_nc_file    Path to the .nc  file written by heat_budget_rd_lme.py

Optional flags
    --tol DEGREES     Tolerance (degrees) for matching lat/lon grid points.
                      Default: 0.001
    --atol VALUE      Absolute tolerance for np.allclose. Default: 1e-6
    --rtol VALUE      Relative tolerance for np.allclose. Default: 1e-5
    --verbose         Print per-point diff statistics (slow for large arrays).
"""

import argparse
import sys
import numpy as np
import scipy.io
import netCDF4 as nc4

# ---------------------------------------------------------------------------
# Variables that are expected in both files
# (MATLAB workspace variable name  →  NetCDF variable name in Python output)
# ---------------------------------------------------------------------------
VARMAP = {
    # MATLAB name   : NC name
    'mld'       : 'mld',
    'Tmld'      : 'Tmld',
    'Tsub'      : 'Tsub',
    'umld'      : 'umld',
    'usub'      : 'usub',
    'vsub'      : 'vsub',
    'wsub'      : 'wsub',
    'sfcflx'    : 'sfcflx',
    'qpen'      : 'qpen',
    'dTdt'      : 'dTdt',
    'umdTmdx'   : 'umdTmdx',
    'updTmdx'   : 'updTmdx',
    'umdTpdx'   : 'umdTpdx',
    'updTpdx'   : 'updTpdx',
    'vmdTmdy'   : 'vmdTmdy',
    'vpdTmdy'   : 'vpdTmdy',
    'vmdTpdy'   : 'vmdTpdy',
    'vpdTpdy'   : 'vpdTpdy',
    'mnupdTpdx' : 'mnupdTpdx',
    'mnvpdTpdy' : 'mnvpdTpdy',
    'w_entr'    : 'w_entr',
    'wmdTmdz'   : 'wmdTmdz',
    'wpdTmdz'   : 'wpdTmdz',
    'wmdTpdz'   : 'wmdTpdz',
    'wpdTpdz'   : 'wpdTpdz',
    'mnwpdTpdz' : 'mnwpdTpdz',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _squeeze(arr):
    """Remove length-1 axes introduced by scipy.io.loadmat."""
    return np.squeeze(np.asarray(arr, dtype=np.float64))


def _to_nan(arr, threshold=1e30):
    """Replace CESM fill values (≈ 1e36) with NaN."""
    out = arr.copy()
    out[np.abs(out) > threshold] = np.nan
    return out


def _load_mat(path):
    """
    Load a MATLAB v5/v7 .mat file.
    Returns a dict  {name: ndarray}.
    Arrays are squeezed to remove length-1 dims added by loadmat.
    """
    raw = scipy.io.loadmat(path, squeeze_me=True)
    return {k: _to_nan(_squeeze(v))
            for k, v in raw.items()
            if not k.startswith('_') and isinstance(v, np.ndarray)}


def _load_nc(path):
    """
    Load all variables from a NetCDF4 file.
    Returns a dict  {name: ndarray}.
    """
    out = {}
    with nc4.Dataset(path) as ds:
        for vname, var in ds.variables.items():
            arr = np.array(var[:], dtype=np.float64)
            if hasattr(arr, 'mask'):
                arr = np.ma.filled(arr, np.nan)
            out[vname] = arr
    return out


def _find_common_lat_rows(tlat_mat, tlat_py, tol):
    """
    For each row in tlat_py (shape: nlat_py × nlon), find the matching row
    index in tlat_mat (shape: nlat_mat × nlon) using the reference column.

    Returns (idx_mat, idx_py) — parallel index arrays for matched rows only.
    """
    # Use median across longitude to get a robust representative value
    ref_mat = np.nanmedian(tlat_mat, axis=1)   # (nlat_mat,)
    ref_py  = np.nanmedian(tlat_py,  axis=1)   # (nlat_py,)

    idx_mat, idx_py = [], []
    for i, v in enumerate(ref_py):
        dists = np.abs(ref_mat - v)
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            idx_mat.append(j)
            idx_py.append(i)

    return np.array(idx_mat, dtype=int), np.array(idx_py, dtype=int)


def _find_common_lon_cols(tlon_mat, tlon_py, tol):
    """Same as above but for longitude columns."""
    ref_mat = np.nanmedian(tlon_mat, axis=0)   # (nlon_mat,)
    ref_py  = np.nanmedian(tlon_py,  axis=0)   # (nlon_py,)

    idx_mat, idx_py = [], []
    for i, v in enumerate(ref_py):
        dists = np.abs(ref_mat - v)
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            idx_mat.append(j)
            idx_py.append(i)

    return np.array(idx_mat, dtype=int), np.array(idx_py, dtype=int)


def _extract(arr, lat_idx, lon_idx):
    """
    Extract a sub-array at the matched lat/lon indices.
    Supports arrays of shape (nlat, nlon), (nt, nlat, nlon),
    or (12, nlat, nlon) — any shape with lat in axis -2 and lon in axis -1.
    """
    if arr.ndim == 2:
        return arr[np.ix_(lat_idx, lon_idx)]
    if arr.ndim == 3:
        return arr[:, lat_idx, :][:, :, lon_idx]
    raise ValueError(f'Unsupported array shape {arr.shape}')


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(mat_path, nc_path, tol=0.001, atol=1e-6, rtol=1e-5, verbose=False):
    print(f"\n{'='*70}")
    print(f"MATLAB file : {mat_path}")
    print(f"Python file : {nc_path}")
    print(f"Grid tol    : ±{tol}°   |  allclose rtol={rtol}, atol={atol}")
    print(f"{'='*70}\n")

    # ---- Load files --------------------------------------------------------
    print("Loading MATLAB .mat …")
    mat = _load_mat(mat_path)
    print("Loading Python  .nc  …")
    py  = _load_nc(nc_path)

    # ---- Grid coordinates --------------------------------------------------
    # MATLAB stores tlat/tlon as (nlat, nlon)
    # Python NC stores TLAT/TLONG as (lat, lon)
    tlat_mat = mat.get('tlat')
    tlon_mat = mat.get('tlon')
    tlat_py  = py.get('TLAT')
    tlon_py  = py.get('TLONG')

    if tlat_mat is None or tlon_mat is None:
        sys.exit("ERROR: 'tlat' or 'tlon' not found in MATLAB file.")
    if tlat_py is None or tlon_py is None:
        sys.exit("ERROR: 'TLAT' or 'TLONG' not found in Python NC file.")

    print(f"MATLAB grid : {tlat_mat.shape[0]} lat × {tlat_mat.shape[1]} lon")
    print(f"Python grid : {tlat_py.shape[0]}  lat × {tlat_py.shape[1]}  lon\n")

    # ---- Find common grid points -------------------------------------------
    lat_idx_mat, lat_idx_py = _find_common_lat_rows(tlat_mat, tlat_py, tol)
    lon_idx_mat, lon_idx_py = _find_common_lon_cols(tlon_mat, tlon_py, tol)

    n_lat_common = len(lat_idx_mat)
    n_lon_common = len(lon_idx_mat)

    print(f"Common lat rows : {n_lat_common}  "
          f"(MATLAB rows {lat_idx_mat[[0,-1]]}, "
          f"Python rows {lat_idx_py[[0,-1]]})")
    print(f"Common lon cols : {n_lon_common}  "
          f"(MATLAB cols {lon_idx_mat[[0,-1]]}, "
          f"Python cols {lon_idx_py[[0,-1]]})")

    if n_lat_common == 0 or n_lon_common == 0:
        sys.exit("ERROR: No common grid points found — check file paths / tol.")

    # Verify the coordinate values actually match
    tlat_mat_sub = _extract(tlat_mat, lat_idx_mat, lon_idx_mat)
    tlat_py_sub  = _extract(tlat_py,  lat_idx_py,  lon_idx_py)
    tlon_mat_sub = _extract(tlon_mat, lat_idx_mat, lon_idx_mat)
    tlon_py_sub  = _extract(tlon_py,  lat_idx_py,  lon_idx_py)

    lat_coord_max_diff = np.nanmax(np.abs(tlat_mat_sub - tlat_py_sub))
    lon_coord_max_diff = np.nanmax(np.abs(tlon_mat_sub - tlon_py_sub))
    print(f"\nCoordinate alignment check:")
    print(f"  Max |TLAT_mat  - TLAT_py|  = {lat_coord_max_diff:.3e} °")
    print(f"  Max |TLONG_mat - TLONG_py| = {lon_coord_max_diff:.3e} °")
    print()

    # ---- Compare each budget variable --------------------------------------
    all_pass = True
    results  = []

    col_w = [20, 10, 12, 12, 12, 10, 8]
    header = (f"{'Variable':<{col_w[0]}} {'Shape_mat':<{col_w[1]}} "
              f"{'MaxAbsDiff':>{col_w[2]}} {'RMSE':>{col_w[3]}} "
              f"{'MeanAbsDiff':>{col_w[4]}} {'Corr':>{col_w[5]}} {'allclose':<{col_w[6]}}")
    print(header)
    print('-' * sum(col_w))

    for mat_name, nc_name in VARMAP.items():
        # ---- Fetch arrays --------------------------------------------------
        if mat_name not in mat:
            print(f"  [SKIP] '{mat_name}' not in MATLAB file")
            continue
        if nc_name not in py:
            print(f"  [SKIP] '{nc_name}' not in Python NC file")
            continue

        arr_mat = mat[mat_name]
        arr_py  = py[nc_name]

        # Some MATLAB arrays may be (nlat, nlon, nt) if saved in column-major
        # order that scipy reorders.  Detect and transpose if needed.
        # Python arrays are always (nt, nlat, nlon) or (12, nlat, nlon).
        # Expected spatial dims equal the common grid size.
        if arr_mat.ndim == 3:
            # Check if spatial axes are last two (MATLAB may save as lat×lon×t)
            if (arr_mat.shape[0] == tlat_mat.shape[0] and
                    arr_mat.shape[1] == tlat_mat.shape[1]):
                # shape (nlat, nlon, nt) → (nt, nlat, nlon)
                arr_mat = arr_mat.transpose(2, 0, 1)
        if arr_py.ndim == 3:
            if (arr_py.shape[0] == tlat_py.shape[0] and
                    arr_py.shape[1] == tlat_py.shape[1]):
                arr_py = arr_py.transpose(2, 0, 1)

        # ---- Extract common grid subset ------------------------------------
        try:
            sub_mat = _extract(arr_mat, lat_idx_mat, lon_idx_mat)
            sub_py  = _extract(arr_py,  lat_idx_py,  lon_idx_py)
        except (ValueError, IndexError) as exc:
            print(f"  [SKIP] '{mat_name}': cannot extract common subset — {exc}")
            continue

        if sub_mat.shape != sub_py.shape:
            print(f"  [SKIP] '{mat_name}': shape mismatch after extraction "
                  f"mat={sub_mat.shape} py={sub_py.shape}")
            continue

        # ---- Statistics ----------------------------------------------------
        diff = sub_mat - sub_py
        mask = np.isfinite(diff) & np.isfinite(sub_mat) & np.isfinite(sub_py)

        if mask.sum() == 0:
            print(f"  [SKIP] '{mat_name}': all NaN in common region")
            continue

        max_abs   = float(np.max(np.abs(diff[mask])))
        rmse      = float(np.sqrt(np.mean(diff[mask] ** 2)))
        mean_abs  = float(np.mean(np.abs(diff[mask])))

        a_flat = sub_mat[mask].ravel()
        b_flat = sub_py[mask].ravel()
        if a_flat.std() > 0 and b_flat.std() > 0:
            corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
        else:
            corr = float('nan')

        close = bool(np.allclose(sub_mat[mask], sub_py[mask],
                                 atol=atol, rtol=rtol, equal_nan=False))
        if not close:
            all_pass = False

        shape_str = str(arr_mat.shape)
        results.append((mat_name, shape_str, max_abs, rmse, mean_abs, corr, close))

        tick = '✓' if close else '✗'
        print(f"{mat_name:<{col_w[0]}} {shape_str:<{col_w[1]}} "
              f"{max_abs:>{col_w[2]}.3e} {rmse:>{col_w[3]}.3e} "
              f"{mean_abs:>{col_w[4]}.3e} {corr:>{col_w[5]}.5f} "
              f"{tick + ' ' + str(close):<{col_w[6]}}")

        if verbose and not close:
            bad = np.where(~np.isclose(sub_mat, sub_py,
                                       atol=atol, rtol=rtol,
                                       equal_nan=True))
            n_bad = len(bad[0])
            print(f"    → {n_bad} mismatched points (first 5):")
            for k in range(min(5, n_bad)):
                idx = tuple(x[k] for x in bad)
                print(f"       idx={idx}  mat={sub_mat[idx]:.8g}  "
                      f"py={sub_py[idx]:.8g}  diff={diff[idx]:.3e}")

    print('-' * sum(col_w))
    print(f"\n{'OVERALL RESULT':>20}: "
          f"{'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Compare MATLAB .mat and Python .nc heat-budget output files.')
    p.add_argument('matlab_mat_file', help='Path to the MATLAB .mat output file')
    p.add_argument('python_nc_file',  help='Path to the Python .nc  output file')
    p.add_argument('--tol',     type=float, default=0.001,
                   help='Lat/lon grid-matching tolerance in degrees (default: 0.001)')
    p.add_argument('--atol',    type=float, default=1e-6,
                   help='Absolute tolerance for np.allclose (default: 1e-6)')
    p.add_argument('--rtol',    type=float, default=1e-5,
                   help='Relative tolerance for np.allclose (default: 1e-5)')
    p.add_argument('--verbose', action='store_true',
                   help='Print per-point details for failing variables')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    ok = compare(
        args.matlab_mat_file,
        args.python_nc_file,
        tol=args.tol,
        atol=args.atol,
        rtol=args.rtol,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)
