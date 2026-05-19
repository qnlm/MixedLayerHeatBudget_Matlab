#!/usr/bin/env python3
"""
verify_results.py
-----------------
Compare every budget variable saved by the MATLAB and Python versions of the
mixed-layer heat budget pipeline.  Both inputs must be NetCDF4 files.

Because the two grids may not be byte-for-byte identical (float32 vs float64,
slight coordinate rounding), the script first finds the common lat/lon grid
points by matching the TLAT / TLONG 2-D coordinate arrays within a configurable
spatial tolerance, then performs element-wise comparison on the matched subset.

All variables that exist in *both* files and have a spatial (lat × lon) shape
are compared automatically — no hard-coded variable list is needed.

Usage
-----
    python verify_results.py <nc_file_a> <nc_file_b> [options]

Positional arguments
    nc_file_a   First  NetCDF4 file  (e.g. MATLAB output)
    nc_file_b   Second NetCDF4 file  (e.g. Python output)

Optional flags
    --tol DEGREES   Tolerance (degrees) for matching lat/lon grid points.
                    Default: 0.001
    --atol VALUE    Absolute tolerance for np.allclose. Default: 1e-6
    --rtol VALUE    Relative tolerance for np.allclose. Default: 1e-5
    --verbose       Print per-point diff statistics for failing variables.
"""

import argparse
import sys
import numpy as np
import netCDF4 as nc4

# ---------------------------------------------------------------------------
# Known candidate names for the 2-D lat/lon coordinate grids.
# The script tries each pair in order and uses the first match found.
# ---------------------------------------------------------------------------
_TLAT_CANDIDATES  = ('TLAT',  'tlat',  'lat',  'latitude')
_TLONG_CANDIDATES = ('TLONG', 'tlong', 'lon',  'longitude')

# Coordinate / dimension-only variables — skip from data comparison.
_SKIP_VARS = frozenset({
    'TLAT', 'tlat', 'TLONG', 'tlong',
    'lat', 'lon', 'latitude', 'longitude',
    'time', 'month', 'depth', 'z_t',
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_nan(arr, threshold=1e30):
    """Replace CESM fill values (≈ 1e36) with NaN, return float64 ndarray."""
    out = np.array(arr, dtype=np.float64)
    if isinstance(out, np.ma.MaskedArray):
        out = np.ma.filled(out, np.nan)
    out[np.abs(out) > threshold] = np.nan
    return out


def _load_nc(path):
    """
    Load all variables from a NetCDF4 file.
    Returns (data_dict, dims_dict) where
      data_dict : {varname: float64 ndarray with fill→NaN}
      dims_dict : {varname: tuple of dimension names}
    """
    data, dims = {}, {}
    with nc4.Dataset(path) as ds:
        for vname, var in ds.variables.items():
            data[vname] = _to_nan(var[:])
            dims[vname] = var.dimensions
    return data, dims


def _find_coord(data, candidates):
    """Return the first key from *candidates* that exists in *data*, or None."""
    for name in candidates:
        if name in data:
            return name
    return None


def _find_common_rows(coord_a, coord_b, tol):
    """
    For each row in coord_b (shape nlat_b × nlon), find the closest matching
    row in coord_a (shape nlat_a × nlon) by the median value across columns.

    Returns (idx_a, idx_b) — parallel index arrays for matched rows only.
    """
    ref_a = np.nanmedian(coord_a, axis=1)
    ref_b = np.nanmedian(coord_b, axis=1)
    idx_a, idx_b = [], []
    for i, v in enumerate(ref_b):
        dists = np.abs(ref_a - v)
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            idx_a.append(j)
            idx_b.append(i)
    return np.array(idx_a, dtype=int), np.array(idx_b, dtype=int)


def _find_common_cols(coord_a, coord_b, tol):
    """Same as _find_common_rows but along columns (longitude axis)."""
    ref_a = np.nanmedian(coord_a, axis=0)
    ref_b = np.nanmedian(coord_b, axis=0)
    idx_a, idx_b = [], []
    for i, v in enumerate(ref_b):
        dists = np.abs(ref_a - v)
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            idx_a.append(j)
            idx_b.append(i)
    return np.array(idx_a, dtype=int), np.array(idx_b, dtype=int)


def _extract(arr, lat_idx, lon_idx):
    """
    Extract a spatial subset from *arr* at the given lat/lon indices.
    Handles shapes: (nlat, nlon), (nt, nlat, nlon), (12, nlat, nlon).
    Returns the extracted sub-array.
    """
    if arr.ndim == 2:
        return arr[np.ix_(lat_idx, lon_idx)]
    if arr.ndim == 3:
        return arr[:, lat_idx, :][:, :, lon_idx]
    raise ValueError(f'Unsupported array ndim={arr.ndim}, shape={arr.shape}')


def _spatial_shape(arr, nlat, nlon):
    """
    Return *arr* reshaped / transposed so that the last two axes are
    (nlat, nlon).  Returns None if the array has no recognisable spatial axes.
    """
    if arr.ndim == 2 and arr.shape == (nlat, nlon):
        return arr
    if arr.ndim == 3:
        if arr.shape[-2:] == (nlat, nlon):
            return arr                          # already (t, lat, lon)
        if arr.shape[:2] == (nlat, nlon):
            return arr.transpose(2, 0, 1)       # (lat, lon, t) → (t, lat, lon)
    return None                                 # not a spatial array


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(path_a, path_b, tol=0.001, atol=1e-6, rtol=1e-5, verbose=False):
    print(f"\n{'='*72}")
    print(f"File A (MATLAB) : {path_a}")
    print(f"File B (Python) : {path_b}")
    print(f"Grid tol : ±{tol}°   |   allclose rtol={rtol}, atol={atol}")
    print(f"{'='*72}\n")

    # ---- Load both NC files ------------------------------------------------
    print("Loading File A …")
    data_a, dims_a = _load_nc(path_a)
    print("Loading File B …")
    data_b, dims_b = _load_nc(path_b)

    # ---- Locate coordinate grids ------------------------------------------
    tlat_name_a = _find_coord(data_a, _TLAT_CANDIDATES)
    tlon_name_a = _find_coord(data_a, _TLONG_CANDIDATES)
    tlat_name_b = _find_coord(data_b, _TLAT_CANDIDATES)
    tlon_name_b = _find_coord(data_b, _TLONG_CANDIDATES)

    for label, name in [('TLAT in A', tlat_name_a), ('TLONG in A', tlon_name_a),
                        ('TLAT in B', tlat_name_b), ('TLONG in B', tlon_name_b)]:
        if name is None:
            sys.exit(f"ERROR: cannot find {label} — tried {_TLAT_CANDIDATES}")

    tlat_a = data_a[tlat_name_a]
    tlon_a = data_a[tlon_name_a]
    tlat_b = data_b[tlat_name_b]
    tlon_b = data_b[tlon_name_b]

    print(f"Grid A : {tlat_a.shape[0]} lat × {tlat_a.shape[1]} lon  "
          f"(coords: {tlat_name_a!r}, {tlon_name_a!r})")
    print(f"Grid B : {tlat_b.shape[0]} lat × {tlat_b.shape[1]} lon  "
          f"(coords: {tlat_name_b!r}, {tlon_name_b!r})\n")

    # ---- Find common grid points ------------------------------------------
    lat_idx_a, lat_idx_b = _find_common_rows(tlat_a, tlat_b, tol)
    lon_idx_a, lon_idx_b = _find_common_cols(tlon_a, tlon_b, tol)

    n_lat = len(lat_idx_a)
    n_lon = len(lon_idx_a)

    if n_lat == 0 or n_lon == 0:
        sys.exit("ERROR: No common grid points found — check file paths / --tol.")

    print(f"Common lat rows : {n_lat}  "
          f"(A rows {lat_idx_a[[0,-1]]}, B rows {lat_idx_b[[0,-1]]})")
    print(f"Common lon cols : {n_lon}  "
          f"(A cols {lon_idx_a[[0,-1]]}, B cols {lon_idx_b[[0,-1]]})")

    # Verify coordinate residuals
    tlat_a_sub = _extract(tlat_a, lat_idx_a, lon_idx_a)
    tlat_b_sub = _extract(tlat_b, lat_idx_b, lon_idx_b)
    tlon_a_sub = _extract(tlon_a, lat_idx_a, lon_idx_a)
    tlon_b_sub = _extract(tlon_b, lat_idx_b, lon_idx_b)

    print(f"\nCoordinate residuals after alignment:")
    print(f"  max |TLAT_A  - TLAT_B|  = {np.nanmax(np.abs(tlat_a_sub - tlat_b_sub)):.3e} °")
    print(f"  max |TLONG_A - TLONG_B| = {np.nanmax(np.abs(tlon_a_sub - tlon_b_sub)):.3e} °\n")

    # ---- Discover variables present in both files --------------------------
    skip = _SKIP_VARS | {tlat_name_a, tlon_name_a, tlat_name_b, tlon_name_b}
    vars_a = set(data_a.keys()) - skip
    vars_b = set(data_b.keys()) - skip

    common_vars  = sorted(vars_a & vars_b)
    only_in_a    = sorted(vars_a - vars_b)
    only_in_b    = sorted(vars_b - vars_a)

    if only_in_a:
        print(f"Variables only in A (skipped): {only_in_a}")
    if only_in_b:
        print(f"Variables only in B (skipped): {only_in_b}")
    print(f"Variables to compare: {len(common_vars)}\n")

    # ---- Compare each variable --------------------------------------------
    all_pass = True

    col_w = [22, 16, 12, 12, 12, 10, 14]
    header = (f"{'Variable':<{col_w[0]}} {'Shape_A':<{col_w[1]}} "
              f"{'MaxAbsDiff':>{col_w[2]}} {'RMSE':>{col_w[3]}} "
              f"{'MeanAbsDiff':>{col_w[4]}} {'Corr':>{col_w[5]}} "
              f"{'allclose':>{col_w[6]}}")
    print(header)
    print('-' * sum(col_w))

    for vname in common_vars:
        arr_a = data_a[vname]
        arr_b = data_b[vname]

        nlat_a, nlon_a = tlat_a.shape
        nlat_b, nlon_b = tlat_b.shape

        # Normalise to (..., lat, lon) layout
        arr_a = _spatial_shape(arr_a, nlat_a, nlon_a)
        arr_b = _spatial_shape(arr_b, nlat_b, nlon_b)

        if arr_a is None or arr_b is None:
            print(f"  [SKIP] '{vname}': no spatial (lat×lon) axes detected")
            continue

        # Extract common grid subset
        try:
            sub_a = _extract(arr_a, lat_idx_a, lon_idx_a)
            sub_b = _extract(arr_b, lat_idx_b, lon_idx_b)
        except (ValueError, IndexError) as exc:
            print(f"  [SKIP] '{vname}': cannot extract common subset — {exc}")
            continue

        if sub_a.shape != sub_b.shape:
            print(f"  [SKIP] '{vname}': shape mismatch after extraction "
                  f"A={sub_a.shape} B={sub_b.shape}")
            continue

        # Statistics on finite, matched values
        diff = sub_a - sub_b
        mask = np.isfinite(sub_a) & np.isfinite(sub_b)

        if mask.sum() == 0:
            print(f"  [SKIP] '{vname}': all NaN / fill in common region")
            continue

        max_abs  = float(np.max(np.abs(diff[mask])))
        rmse     = float(np.sqrt(np.mean(diff[mask] ** 2)))
        mean_abs = float(np.mean(np.abs(diff[mask])))

        a_flat = sub_a[mask].ravel()
        b_flat = sub_b[mask].ravel()
        corr = (float(np.corrcoef(a_flat, b_flat)[0, 1])
                if a_flat.std() > 0 and b_flat.std() > 0 else float('nan'))

        close = bool(np.allclose(sub_a[mask], sub_b[mask],
                                 atol=atol, rtol=rtol, equal_nan=False))
        if not close:
            all_pass = False

        tick = '✓' if close else '✗'
        shape_str = str(arr_a.shape)
        print(f"{vname:<{col_w[0]}} {shape_str:<{col_w[1]}} "
              f"{max_abs:>{col_w[2]}.3e} {rmse:>{col_w[3]}.3e} "
              f"{mean_abs:>{col_w[4]}.3e} {corr:>{col_w[5]}.5f} "
              f"{tick + ' ' + str(close):>{col_w[6]}}")

        if verbose and not close:
            bad = np.where(~np.isclose(sub_a, sub_b,
                                       atol=atol, rtol=rtol, equal_nan=True))
            n_bad = len(bad[0])
            print(f"    → {n_bad} mismatched points (first 5):")
            for k in range(min(5, n_bad)):
                idx = tuple(x[k] for x in bad)
                print(f"       idx={idx}  A={sub_a[idx]:.8g}  "
                      f"B={sub_b[idx]:.8g}  diff={diff[idx]:.3e}")

    print('-' * sum(col_w))
    verdict = 'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'
    print(f"\n{'OVERALL RESULT':>20}: {verdict}")
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Compare MATLAB and Python heat-budget NetCDF output files.')
    p.add_argument('nc_file_a', help='First  NC file (e.g. MATLAB output)')
    p.add_argument('nc_file_b', help='Second NC file (e.g. Python output)')
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
        args.nc_file_a,
        args.nc_file_b,
        tol=args.tol,
        atol=args.atol,
        rtol=args.rtol,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)
