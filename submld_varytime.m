% Function to find the value of a three-dimensional field just below a mixed
% layer depth taken from an input field
% March 2015
% Sam Stevenson
% May 2026, Z. Liu: vectorised – replaced triple nested loop with
%   broadcasting and a cumsum-based pick of the first/last sub-MLD level.

% Inputs:
% mld   -   matrix containing mixed-layer depth field, dimensions
%           nt x nlat x nlon, units of m
% field -   matrix containing desired field to integrate, dimensions nt x
%           nz x nlat x nlon
% time  -   array containing measurement times for field, dimensions nt
%           units should be in days since 0000-01-01 for compatibility with
%           the Matlab "datenum" function
% z     -   array of depths, units of m
% type  -   type of searching to do on depth array. For CESM/POP: should
%           use 'first', for ROMS 'last' since depths are ordered backwards

% Outputs:
% fldint -  matrix containing the values of the input field averaged over
%           the mixed layer, dimensions nt x nlat x nlon

function [fldint] = submld_varytime(mld, field, time, z, type)

    nt   = size(field, 1);
    nz   = size(field, 2);
    nlat = size(field, 3);
    nlon = size(field, 4);

    % Boolean mask: True where depth level > MLD  -> (nt, nz, nlat, nlon)
    z_4d   = reshape(z,   [1,  nz, nlat, nlon]);
    mld_4d = reshape(mld, [nt,  1, nlat, nlon]);
    below  = bsxfun(@gt, z_4d, mld_4d);

    % Whether any sub-MLD level exists at each (t, lat, lon)
    has_valid = reshape(any(below, 2), [nt, nlat, nlon]);

    % Select the target depth level using a cumsum trick:
    %   cumsum(below, 2) == 1  marks exactly the first True along depth.
    if strcmp(type, 'first')
        pick = (cumsum(below, 2) == 1) & below;          % (nt, nz, nlat, nlon)
    else  % 'last'
        flipped = flip(below, 2);
        pick    = flip((cumsum(flipped, 2) == 1) & flipped, 2);
    end

    % Sum field over the single selected level -> (nt, nlat, nlon)
    fldint = reshape(sum(double(field) .* pick, 2), [nt, nlat, nlon]);

    % Zero out positions where no sub-MLD level was found
    fldint(~has_valid) = 0;
end