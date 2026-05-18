% Function to perform averaging of a three-dimensional field over a mixed
% layer depth taken from an input field
% February 7, 2013
% Sam Stevenson
% Updated September 19, 2013 to use a time-varying mixed layer depth
%   (appropriate for ROMS model output)
% May 2026, Z. Liu: vectorised – replaced triple nested loop with
%   broadcasting (bsxfun) to mask sub-MLD levels in one operation.

% Inputs:
% mld   -   matrix containing mixed-layer depth field, dimensions
%           nt x nlat x nlon, units of m
% field -   matrix containing desired field to integrate, dimensions nt x
%           nz x nlat x nlon
% time  -   array containing measurement times for field, dimensions nt
%           units should be in days since 0000-01-01 for compatibility with
%           the Matlab "datenum" function
% z     -   array of depths, units of m

% Outputs:
% fldint -  matrix containing the values of the input field averaged over
%           the mixed layer, dimensions nt x nlat x nlon

function [fldint] = mldavg_varytime(mld, field, time, z)

    nt   = size(field, 1);
    nlat = size(field, 3);
    nlon = size(field, 4);

    % Expand z:   (nz, nlat, nlon) -> (1,  nz, nlat, nlon)
    % Expand mld: (nt, nlat, nlon) -> (nt,  1, nlat, nlon)
    z_4d   = reshape(z,   [1,  size(z,1),   nlat, nlon]);
    mld_4d = reshape(mld, [nt, 1,           nlat, nlon]);

    % Mask levels below the mixed layer in a single vectorised step
    field = double(field);
    field(bsxfun(@gt, z_4d, mld_4d)) = NaN;

    % NaN-mean over depth axis (dim 2); reshape to drop the singleton
    fldint = reshape(nanmean(field, 2), [nt, nlat, nlon]);
end