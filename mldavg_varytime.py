"""
Average a three-dimensional field over the mixed layer depth.

Python translation of mldavg_varytime.m
Original: February 7, 2013 / Sam Stevenson
Updated:  September 19, 2013 (time-varying MLD)

Parameters
----------
mld   : ndarray, shape (nt, nlat, nlon)
    Mixed-layer depth in metres.
field : ndarray, shape (nt, nz, nlat, nlon)
    Field to average over the mixed layer.
time  : ndarray, shape (nt,)
    Measurement times (unused directly, kept for API compatibility).
z     : ndarray, shape (nz, nlat, nlon)
    Depth levels in metres (positive downward).

Returns
-------
fldint : ndarray, shape (nt, nlat, nlon)
    Field values averaged over the mixed layer (NaN-mean).
"""

import numpy as np


def mldavg_varytime(mld, field, time, z):
    # Work on a float copy so we can write NaN without touching the caller's data
    field = np.array(field, dtype=float, copy=True)

    # Build a mask: True wherever a depth level lies *below* the mixed layer
    # z   shape: (nz, nlat, nlon)  -> broadcast to (1, nz, nlat, nlon)
    # mld shape: (nt, nlat, nlon)  -> broadcast to (nt, 1, nlat, nlon)
    below_mld = z[np.newaxis, :, :, :] > mld[:, np.newaxis, :, :]  # (nt, nz, nlat, nlon)
    field[below_mld] = np.nan

    # NaN-mean over the depth axis (axis=1)
    fldint = np.nanmean(field, axis=1)
    return fldint
