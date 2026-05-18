"""
Reynolds-decomposed vertical advection terms for the mixed-layer heat budget.

Python translation of vertadv_ml_rd.m
Original: March 2015 / Sam Stevenson

Inputs
------
time   : (nt,)             – time in days
lat    : (nlat, nlon)      – latitude  [degrees]
lon    : (nlat, nlon)      – longitude [degrees]
mld    : (nt, nlat, nlon)  – mixed-layer depth [m]
umld   : (nt, nlat, nlon)  – MLD-averaged zonal velocity [m/s]
vmld   : (nt, nlat, nlon)  – MLD-averaged meridional velocity [m/s]
Tmld   : (nt, nlat, nlon)  – MLD-averaged temperature [°C]
Tsub   : (nt, nlat, nlon)  – temperature just below the MLD [°C]
wsub   : (nt, nlat, nlon)  – vertical velocity just below the MLD [m/s]
mon    : (nt,)             – month of each time step (1-12)
yr     : (nt,)             – year of each time step
yrclim : sequence of 2     – [yr_min, yr_max] for climatology window

Returns (all shape nt x nlat x nlon unless noted)
-------
w_entr    – entrainment velocity [m/s]
wmdTmdz   – mean-w × mean-dT/dz (climatological entrainment of mean gradient)
wpdTmdz   – w′ × mean-dT/dz    (anomalous entrainment of mean gradient)
wmdTpdz   – mean-w × dT′/dz    (climatological entrainment of anomalous gradient)
wpdTpdz   – w′ × dT′/dz        (anomalous entrainment of anomalous gradient)
mnwpdTpdz – (12, nlat, nlon)   monthly climatology of wpdTpdz
"""

import numpy as np

RE = 6378.0e3  # Earth radius [m]


def _pad_idx(n):
    """Same padding scheme as in advection_ml_rd.py."""
    fwd = np.concatenate([[1], np.arange(1, n), [n - 1]])
    bwd = np.concatenate([[0], np.arange(0, n - 1), [n - 2]])
    return fwd, bwd


def vertadv_ml_rd(time, lat, lon, mld, umld, vmld, Tmld, Tsub, wsub, mon, yr, yrclim):
    nt, nlat, nlon = mld.shape
    latarr = lat * np.pi / 180.0
    lonarr = lon * np.pi / 180.0

    # ------------------------------------------------------------------
    # 1.  Time derivative of MLD  [m/day]
    # ------------------------------------------------------------------
    fwd_t, bwd_t = _pad_idx(nt)
    dt = (time[fwd_t] - time[bwd_t])[:, np.newaxis, np.newaxis]   # (nt+1, 1, 1)
    dHtmp = (mld[fwd_t] - mld[bwd_t]) / dt                        # (nt+1, nlat, nlon)
    dHdt  = 0.5 * (dHtmp[:-1] + dHtmp[1:])                        # (nt,   nlat, nlon)

    # ------------------------------------------------------------------
    # 2.  Horizontal derivatives of MLD (dimensionless: m/m)
    # ------------------------------------------------------------------
    dHdx = np.zeros_like(mld)
    dHdy = np.zeros_like(mld)

    fwd_lon, bwd_lon = _pad_idx(nlon)
    fwd_lat, bwd_lat = _pad_idx(nlat)

    for tt in range(nt):
        mld_tt = mld[tt]                                            # (nlat, nlon)
        dH = mld_tt[:, fwd_lon] - mld_tt[:, bwd_lon]               # (nlat, nlon+1)
        dHtmp2 = dH * np.cos(latarr[:, bwd_lon]) / (lonarr[:, fwd_lon] - lonarr[:, bwd_lon])
        dHdx[tt] = 0.5 * (dHtmp2[:, :-1] + dHtmp2[:, 1:]) / RE

        dHtmp2 = (mld_tt[fwd_lat, :] - mld_tt[bwd_lat, :]) / (latarr[fwd_lat, :] - latarr[bwd_lat, :])
        dHdy[tt] = 0.5 * (dHtmp2[:-1, :] + dHtmp2[1:, :]) / RE

    # ------------------------------------------------------------------
    # 3.  Entrainment velocity  [m/s]
    #     w_entr = dH/dt / 86400 + u·∇H + w_sub
    # ------------------------------------------------------------------
    w_entr = dHdt / 86400.0 + umld * dHdx + vmld * dHdy + wsub

    # ------------------------------------------------------------------
    # 4.  Cross-MLD temperature difference normalised by MLD depth
    # ------------------------------------------------------------------
    dTdz = (Tmld - Tsub) / mld    # [°C / m]

    # ------------------------------------------------------------------
    # 5.  Monthly climatologies over the yrclim window
    # ------------------------------------------------------------------
    Tm_mld = np.zeros((12, nlat, nlon))
    Tm_sub = np.zeros((12, nlat, nlon))
    wm     = np.zeros((12, nlat, nlon))

    for mm in range(1, 13):
        thism = np.where((mon == mm) & (yr >= yrclim[0]) & (yr <= yrclim[1]))[0]
        Tm_mld[mm - 1] = np.nanmean(Tmld[thism],   axis=0)
        Tm_sub[mm - 1] = np.nanmean(Tsub[thism],   axis=0)
        wm[mm - 1]     = np.nanmean(w_entr[thism], axis=0)

    # Heaviside step function on climatological mean entrainment:
    # only entrain (positive w_entr means upward motion into the layer)
    wsgn = np.where(wm > 0, 1.0, 0.0)   # (12, nlat, nlon)

    # Climatological mean temperature difference (no depth normalisation yet;
    # the normalisation by MLD is done per time step below)
    dTm = Tm_mld - Tm_sub   # (12, nlat, nlon)

    # ------------------------------------------------------------------
    # 6.  Reynolds-decomposed vertical advection terms
    # ------------------------------------------------------------------
    wmdTmdz = np.zeros_like(Tmld)
    wpdTmdz = np.zeros_like(Tmld)
    wmdTpdz = np.zeros_like(Tmld)
    wpdTpdz = np.zeros_like(Tmld)

    for tt in range(nt):
        mi = mon[tt] - 1   # 0-based month index

        # Mean entrainment of mean gradient
        wmdTmdz[tt] = wsgn[mi] * wm[mi] * dTm[mi] / mld[tt]

        # Anomalous entrainment of mean gradient
        wp = w_entr[tt] - wm[mi]
        wpdTmdz[tt] = wsgn[mi] * wp * (Tm_mld[mi] - Tm_sub[mi]) / mld[tt]

        # Mean entrainment of anomalous gradient
        tmp1 = Tmld[tt] - Tm_mld[mi]
        tmp2 = Tsub[tt] - Tm_sub[mi]
        wmdTpdz[tt] = wsgn[mi] * wm[mi] * (tmp1 - tmp2) / mld[tt]

        # Anomalous entrainment of anomalous gradient
        wpdTpdz[tt] = wsgn[mi] * wp * (tmp1 - tmp2) / mld[tt]

    # ------------------------------------------------------------------
    # 7.  Monthly climatology of the eddy–eddy term
    # ------------------------------------------------------------------
    mnwpdTpdz = np.zeros((12, nlat, nlon))
    for mm in range(1, 13):
        thism = np.where((mon == mm) & (yr >= yrclim[0]) & (yr <= yrclim[1]))[0]
        mnwpdTpdz[mm - 1] = np.nanmean(wpdTpdz[thism], axis=0)

    return w_entr, wmdTmdz, wpdTmdz, wmdTpdz, wpdTpdz, mnwpdTpdz
