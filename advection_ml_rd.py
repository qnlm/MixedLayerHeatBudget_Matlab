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
    dSpdt     – time-derivative of tracer anomaly
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


def _grad_x(field2d, latarr, lonarr):
    """Zonal gradient on a sphere, shape (nlat, nlon)."""
    nlon = field2d.shape[1]
    fwd, bwd = _pad_idx(nlon)

    dS = field2d[:, fwd] - field2d[:, bwd]                          # (nlat, nlon+1)
    dStmp = dS * np.cos(latarr[:, bwd]) / (lonarr[:, fwd] - lonarr[:, bwd])
    return 0.5 * (dStmp[:, :-1] + dStmp[:, 1:]) / RE               # (nlat, nlon)


def _grad_y(field2d, latarr):
    """Meridional gradient on a sphere, shape (nlat, nlon)."""
    nlat = field2d.shape[0]
    fwd, bwd = _pad_idx(nlat)

    dStmp = (field2d[fwd, :] - field2d[bwd, :]) / (latarr[fwd, :] - latarr[bwd, :])
    return 0.5 * (dStmp[:-1, :] + dStmp[1:, :]) / RE               # (nlat, nlon)


def advection_ml_rd(salt, uvel, vvel, lat, lon, time, yr, mon, yrclim):
    salt = np.array(salt, dtype=float, copy=True)   # local copy (modified later)
    nt, nlat, nlon = salt.shape

    latarr = lat * np.pi / 180.0
    lonarr = lon * np.pi / 180.0

    # ------------------------------------------------------------------
    # 1. Instantaneous horizontal gradients at every time step
    # ------------------------------------------------------------------
    dSdx = np.zeros_like(salt)
    dSdy = np.zeros_like(salt)

    for tt in range(nt):
        dSdx[tt] = _grad_x(salt[tt], latarr, lonarr)
        dSdy[tt] = _grad_y(salt[tt], latarr)

    # ------------------------------------------------------------------
    # 2. Monthly climatologies over the yrclim window
    # ------------------------------------------------------------------
    Sm     = np.zeros((12, nlat, nlon))
    dSmdx  = np.zeros((12, nlat, nlon))
    dSmdy  = np.zeros((12, nlat, nlon))
    um     = np.zeros((12, nlat, nlon))
    vm     = np.zeros((12, nlat, nlon))

    for mm in range(1, 13):
        thism = np.where((mon == mm) & (yr >= yrclim[0]) & (yr <= yrclim[1]))[0]
        Sm[mm - 1]  = np.nanmean(salt[thism],  axis=0)
        um[mm - 1]  = np.nanmean(uvel[thism],  axis=0)
        vm[mm - 1]  = np.nanmean(vvel[thism],  axis=0)

    for mm in range(1, 13):
        dSmdx[mm - 1] = _grad_x(Sm[mm - 1], latarr, lonarr)
        dSmdy[mm - 1] = _grad_y(Sm[mm - 1], latarr)

    # ------------------------------------------------------------------
    # 3. Mean advection of mean gradient  (size: nt x nlat x nlon)
    # ------------------------------------------------------------------
    umdSmdx = np.zeros_like(salt)
    vmdSmdy = np.zeros_like(salt)

    for mm in range(1, 13):
        thism = np.where(mon == mm)[0]
        umdSmdx[thism] = um[mm - 1] * dSmdx[mm - 1]   # broadcast over time
        vmdSmdy[thism] = vm[mm - 1] * dSmdy[mm - 1]

    # ------------------------------------------------------------------
    # 4. Anomalous advection of mean gradient
    # ------------------------------------------------------------------
    updSmdx = np.zeros_like(salt)
    vpdSmdy = np.zeros_like(salt)

    for tt in range(nt):
        mi = mon[tt] - 1   # 0-based month index
        updSmdx[tt] = (uvel[tt] - um[mi]) * dSmdx[mi]
        vpdSmdy[tt] = (vvel[tt] - vm[mi]) * dSmdy[mi]

    # ------------------------------------------------------------------
    # 5. Mean advection of anomalous gradient
    # ------------------------------------------------------------------
    umdSpdx = np.zeros_like(salt)
    vmdSpdy = np.zeros_like(salt)

    for tt in range(nt):
        mi = mon[tt] - 1
        umdSpdx[tt] = um[mi] * (dSdx[tt] - dSmdx[mi])
        vmdSpdy[tt] = vm[mi] * (dSdy[tt] - dSmdy[mi])

    # ------------------------------------------------------------------
    # 6. Anomalous advection of anomalous gradient
    # ------------------------------------------------------------------
    updSpdx = np.zeros_like(salt)
    vpdSpdy = np.zeros_like(salt)

    for tt in range(nt):
        mi = mon[tt] - 1
        updSpdx[tt] = (uvel[tt] - um[mi]) * (dSdx[tt] - dSmdx[mi])
        vpdSpdy[tt] = (vvel[tt] - vm[mi]) * (dSdy[tt] - dSmdy[mi])

    # ------------------------------------------------------------------
    # 7. Monthly-climatological mean of the eddy–eddy cross-terms
    # ------------------------------------------------------------------
    mnupdSpdx = np.zeros_like(salt)
    mnvpdSpdy = np.zeros_like(salt)

    for mm in range(1, 13):
        thism = np.where(mon == mm)[0]
        mnupdSpdx[thism] = np.nanmean(updSpdx[thism], axis=0)
        mnvpdSpdy[thism] = np.nanmean(vpdSpdy[thism], axis=0)

    # ------------------------------------------------------------------
    # 8. Time derivative of the tracer anomaly
    # ------------------------------------------------------------------
    # Convert salt to anomaly in-place (local copy)
    for tt in range(nt):
        salt[tt] -= Sm[mon[tt] - 1]

    dSpdt = np.zeros_like(salt)
    if nt > 1:
        fwd_t, bwd_t = _pad_idx(nt)
        dt = (time[fwd_t] - time[bwd_t])[:, np.newaxis, np.newaxis]   # (nt+1, 1, 1)
        dStmp = (salt[fwd_t] - salt[bwd_t]) / dt                      # (nt+1, nlat, nlon)
        dSpdt = 0.5 * (dStmp[:-1] + dStmp[1:])                        # (nt,   nlat, nlon)

    return (umdSmdx, updSmdx, umdSpdx, updSpdx,
            vmdSmdy, vpdSmdy, vmdSpdy, vpdSpdy,
            dSpdt, mnupdSpdx, mnvpdSpdy)
