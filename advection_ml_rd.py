"""
Reynolds-decomposed horizontal advection of a mixed-layer tracer.

Python translation of advection_ml_rd.m
Original: November 3, 2013 / Sam Stevenson
Validated: January 2021

Inputs (all arrays use C-order, i.e. time-first):
    salt   : (nt, nlat, nlon)  – mixed-layer tracer (temperature or salinity)
    uvel   : (nt, nlat, nlon)  – mixed-layer zonal velocity [m/s]
    vvel   : (nt, nlat, nlon)  – mixed-layer meridional velocity [m/s]
    lat    : (nlat, nlon)      – latitude  [degrees]
    lon    : (nlat, nlon)      – longitude [degrees]
    time   : (nt,)             – time in days (used for dT/dt)
    yr     : (nt,)             – year of each time step
    mon    : (nt,)             – month of each time step (1-12)
    yrclim : sequence of 2     – [yr_min, yr_max] for climatology window

Returns (all shape nt x nlat x nlon unless noted):
    umdSmdx   – mean-u × mean-dS/dx
    updSmdx   – u′ × mean-dS/dx
    umdSpdx   – mean-u × dS′/dx
    updSpdx   – u′ × dS′/dx
    vmdSmdy   – mean-v × mean-dS/dy
    vpdSmdy   – v′ × mean-dS/dy
    vmdSpdy   – mean-v × dS′/dy
    vpdSpdy   – v′ × dS′/dy
    dSpdt     – time-derivative of tracer anomaly [units/day]
    mnupdSpdx – monthly-climatological mean of updSpdx
    mnvpdSpdy – monthly-climatological mean of vpdSpdy
"""

import numpy as np

RE = 6378.0e3  # Earth radius [m]


def _pad_idx(n):
    """
    Return the two index vectors that implement the centred-difference
    padding scheme used in the original MATLAB code.

    MATLAB: A([2 2:end end]) and A([1 1:(end-1) (end-1)])  (1-indexed)
    Python equivalent (0-indexed), both of length n+1:
        fwd: [1, 1, 2, ..., n-1, n-1]
        bwd: [0, 0, 1, ..., n-2, n-2]
    """
    fwd = np.concatenate([[1], np.arange(1, n), [n - 1]])
    bwd = np.concatenate([[0], np.arange(0, n - 1), [n - 2]])
    return fwd, bwd


def _grad_x(field, latarr, lonarr):
    """
    Zonal gradient on a sphere.  Works for any leading batch dimensions.
    field  : (..., nlat, nlon)
    latarr : (nlat, nlon)  radians
    lonarr : (nlat, nlon)  radians
    Returns (..., nlat, nlon).
    """
    nlon = field.shape[-1]
    fwd, bwd = _pad_idx(nlon)

    dS     = field[..., fwd] - field[..., bwd]                        # (..., nlat, nlon+1)
    dStmp  = dS * np.cos(latarr[..., bwd]) / (lonarr[..., fwd] - lonarr[..., bwd])
    return 0.5 * (dStmp[..., :-1] + dStmp[..., 1:]) / RE             # (..., nlat, nlon)


def _grad_y(field, latarr):
    """
    Meridional gradient on a sphere.  Works for any leading batch dimensions.
    field  : (..., nlat, nlon)
    latarr : (nlat, nlon)  radians
    Returns (..., nlat, nlon).
    """
    nlat = field.shape[-2]
    fwd, bwd = _pad_idx(nlat)

    dlat   = latarr[fwd, :] - latarr[bwd, :]                         # (nlat+1, nlon)
    dStmp  = (field[..., fwd, :] - field[..., bwd, :]) / dlat
    return 0.5 * (dStmp[..., :-1, :] + dStmp[..., 1:, :]) / RE      # (..., nlat, nlon)


def advection_ml_rd(salt, uvel, vvel, lat, lon, time, yr, mon, yrclim):
    salt = np.array(salt, dtype=float, copy=True)   # local copy (modified later)
    nt, nlat, nlon = salt.shape

    latarr = lat * np.pi / 180.0
    lonarr = lon * np.pi / 180.0

    # ------------------------------------------------------------------
    # 1. Instantaneous horizontal gradients for all time steps at once
    # ------------------------------------------------------------------
    dSdx = _grad_x(salt, latarr, lonarr)   # (nt, nlat, nlon)
    dSdy = _grad_y(salt, latarr)            # (nt, nlat, nlon)

    # ------------------------------------------------------------------
    # 2. Monthly climatologies over the yrclim window
    # ------------------------------------------------------------------
    Sm    = np.zeros((12, nlat, nlon))
    um    = np.zeros((12, nlat, nlon))
    vm    = np.zeros((12, nlat, nlon))

    for mm in range(1, 13):
        thism = np.where((mon == mm) & (yr >= yrclim[0]) & (yr <= yrclim[1]))[0]
        if thism.size:
            Sm[mm - 1] = np.nanmean(salt[thism], axis=0)
            um[mm - 1] = np.nanmean(uvel[thism], axis=0)
            vm[mm - 1] = np.nanmean(vvel[thism], axis=0)

    # Gradients of the monthly climatologies
    dSmdx = _grad_x(Sm, latarr, lonarr)    # (12, nlat, nlon)
    dSmdy = _grad_y(Sm, latarr)            # (12, nlat, nlon)

    # ------------------------------------------------------------------
    # 3–6. Reynolds-decomposed advection terms
    #      Use mi = mon-1  for vectorised month-index lookup.
    #      um[mi] / vm[mi] etc. all broadcast to (nt, nlat, nlon).
    # ------------------------------------------------------------------
    mi = mon - 1   # 0-based month indices, shape (nt,)

    # 3. Mean advection of mean gradient
    umdSmdx = um[mi] * dSmdx[mi]
    vmdSmdy = vm[mi] * dSmdy[mi]

    # 4. Anomalous advection of mean gradient
    updSmdx = (uvel - um[mi]) * dSmdx[mi]
    vpdSmdy = (vvel - vm[mi]) * dSmdy[mi]

    # 5. Mean advection of anomalous gradient
    umdSpdx = um[mi] * (dSdx - dSmdx[mi])
    vmdSpdy = vm[mi] * (dSdy - dSmdy[mi])

    # 6. Anomalous advection of anomalous gradient
    updSpdx = (uvel - um[mi]) * (dSdx - dSmdx[mi])
    vpdSpdy = (vvel - vm[mi]) * (dSdy - dSmdy[mi])

    # ------------------------------------------------------------------
    # 7. Monthly-climatological mean of the eddy–eddy cross-terms
    #    For each month mm, compute the time-mean of all matching time
    #    steps and broadcast the result back to those same steps.
    # ------------------------------------------------------------------
    mnupdSpdx = np.zeros_like(salt)
    mnvpdSpdy = np.zeros_like(salt)

    for mm in range(1, 13):
        thism = np.where(mon == mm)[0]
        if thism.size:
            mnupdSpdx[thism] = np.nanmean(updSpdx[thism], axis=0)
            mnvpdSpdy[thism] = np.nanmean(vpdSpdy[thism], axis=0)

    # ------------------------------------------------------------------
    # 8. Time derivative of the tracer anomaly
    # ------------------------------------------------------------------
    # Subtract monthly climatology to get anomaly (vectorised)
    salt -= Sm[mi]   # salt[tt] -= Sm[mon[tt]-1] for every tt

    dSpdt = np.zeros_like(salt)
    if nt > 1:
        fwd_t, bwd_t = _pad_idx(nt)
        dt    = (time[fwd_t] - time[bwd_t])[:, np.newaxis, np.newaxis]  # (nt+1,1,1)
        dStmp = (salt[fwd_t] - salt[bwd_t]) / dt                        # (nt+1, nlat, nlon)
        dSpdt = 0.5 * (dStmp[:-1] + dStmp[1:])                          # (nt,   nlat, nlon)

    return (umdSmdx, updSmdx, umdSpdx, updSpdx,
            vmdSmdy, vpdSmdy, vmdSpdy, vpdSpdy,
            dSpdt, mnupdSpdx, mnvpdSpdy)
