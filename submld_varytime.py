"""
Find the value of a three-dimensional field just below the mixed layer.

Python translation of submld_varytime.m
Original: March 2015 / Sam Stevenson

Parameters
----------
mld         : ndarray, shape (nt, nlat, nlon)
    Mixed-layer depth in metres.
field       : ndarray, shape (nt, nz, nlat, nlon)
    Field whose sub-MLD value is needed.
time        : ndarray, shape (nt,)
    Measurement times (unused directly, kept for API compatibility).
z           : ndarray, shape (nz, nlat, nlon)
    Depth levels in metres (positive downward).
search_type : str, 'first' or 'last'
    'first'  – use the shallowest depth below the MLD (CESM/POP convention).
    'last'   – use the deepest depth below the MLD (ROMS convention).

Returns
-------
fldint : ndarray, shape (nt, nlat, nlon)
    Field value at the level immediately below the mixed layer.
    Zero where no valid sub-MLD level exists.
"""

import numpy as np


def submld_varytime(mld, field, time, z, search_type='first'):
    nz = field.shape[1]

    # Boolean mask: True where a depth level is below the mixed layer
    # z   -> (1, nz, nlat, nlon),  mld -> (nt, 1, nlat, nlon)
    below_mld = z[np.newaxis, :, :, :] > mld[:, np.newaxis, :, :]  # (nt, nz, nlat, nlon)

    # Whether any sub-MLD level exists at each (t, lat, lon)
    has_valid = below_mld.any(axis=1)  # (nt, nlat, nlon)

    if search_type == 'first':
        # argmax returns the index of the first True along the depth axis.
        # For all-False rows it returns 0; we mask those out later.
        depth_idx = np.argmax(below_mld, axis=1)  # (nt, nlat, nlon)
    else:
        # 'last': flip along depth, find first True, convert back to original index
        depth_idx = nz - 1 - np.argmax(below_mld[:, ::-1, :, :], axis=1)

    # Gather field values using the depth index via take_along_axis
    # Expand depth_idx to (nt, 1, nlat, nlon) for take_along_axis
    fldint = np.take_along_axis(
        np.array(field, dtype=float),
        depth_idx[:, np.newaxis, :, :],
        axis=1
    ).squeeze(axis=1)  # -> (nt, nlat, nlon)

    # Zero out positions where no sub-MLD level was found
    fldint[~has_valid] = 0.0
    return fldint
