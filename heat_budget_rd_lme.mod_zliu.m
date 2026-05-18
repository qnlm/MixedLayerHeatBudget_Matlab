% Compute mixed-layer heat budget for LME simulations, figure out why there
% is a change in the type of El Nino event
% March 2015
% Sam Stevenson
%
% May 2026, zliu

clc;clear all;close all

setenv('HDF5_USE_FILE_LOCKING', 'FALSE'); 

% =========================================================================
% 用户设置选项
% =========================================================================
% 选择数据存放格式：
% 1 = 某一时段的所有变量整合在同一个文件中 (All-in-one file)
% 2 = 一个变量一个文件 (CESM standard monthly time series)
% 3 = 每月一个文件，包含所有变量 (Monthly history files)
data_format_option = 3; 

% 统一的基础路径 (不包含 runname)
base_dir = '/data4/liuzedong/archive2/'; 

% 保存结果的路径 (请根据您的实际需求修改)
out_dir = '/lustre/home/liuzedong/fw/heatbudget/';

% =========================================================================
% 初始化与网格读取
% =========================================================================
cm = 1-[[1*(1:-.1:0)' 1*(1:-.1:0)' 0*(1:-.1:0)'];[0*(0:.1:1)' 1*(0:.1:1)' 1*(0:.1:1)']]; 
nz = 20; % depth-index choosed
regbox = [-10, 10, 190, 240];     % Region of interest
RE = 6378.e3;
rho = 1025;       % Mean density of seawater at 20C, 35 psu: kg/m^3
cp = 3993;        % Specific heat of seawater at 20C, 35 psu, J/kg/K
rho_cp = rho * cp; 

% choose lat-lon region (注意：如果网格文件也在新路径，请同步修改这里)
gridFile = fullfile(base_dir, 'b.e10.B1850CN.T31_g37.001.rest4101', 'ocn', 'hist',...
                    'b.e10.B1850CN.T31_g37.001.rest4101.pop.h.5200-01.nc');
gridnc = netcdf(gridFile);
tlat = gridnc{'TLAT'}(:,:);
tlon = gridnc{'TLONG'}(:,:);
mylat = find(tlat(:,50) >= regbox(1) & tlat(:,50) <= regbox(2));
mylon = find(tlon(50,:) >= regbox(3) & tlon(50,:) <= regbox(4));
tlat = tlat(mylat,mylon);
tlon = tlon(mylat,mylon);

nlat = length(mylat);
nlon = length(mylon);

% % CESM control
% ensnames={'CTRL','FWFIX'};
% runnames={'b40.1850.track1.1deg.006';'b40.1850.track1.1deg.006.fwfix'};
% dates={'080001-089912','080001-089912'};

ensnames={'CTRL'};
runnames={'b.e10.B1850CN.T31_g37.001.rest4101'};
dates={'520001-530012'};

% =========================================================================
% 主循环
% =========================================================================
%%%% Loop through each simulation
for ee=1:length(ensnames)
    for rr=1:size(runnames,2)
        runname=runnames{ee,rr}

        for dd = 1:length(dates)
            date_str = dates{dd}
            
            % -------------------------------------------------------------
            % 数据读取模块
            % -------------------------------------------------------------
            if data_format_option == 1
                % 格式1路径
                data_dir_1 = fullfile(base_dir, runname, 'ocn', 'proc');
                filepath = fullfile(data_dir_1, sprintf('%s.pop.h.all.%s.nc', runname, date_str));
                nc = netcdf(filepath);

                time = nc{'time'}(:);
                z    = nc{'z_t'}(1:nz) / 100.;
                [yr, mon, ~] = datenumnoleap(time - 29, [0 1 1]);

                temp = nc{'TEMP'}(:, 1:nz, mylat, mylon);
                uvel = nc{'UVEL'}(:, 1:nz, mylat, mylon) / 100.;
                vvel = nc{'VVEL'}(:, 1:nz, mylat, mylon) / 100.;
                wvel = nc{'WVEL'}(:, 1:nz, mylat, mylon) / 100.;
                qnet = nc{'SHF'}(:, mylat, mylon);
                qsw  = nc{'SHF_QSW'}(:, mylat, mylon);
                mld  = nc{'HMXL'}(:, mylat, mylon) / 100.;

            elseif data_format_option == 2
                % 格式2路径
                data_dir_2 = fullfile(base_dir, runname, 'ocn', 'proc', 'tseries', 'monthly');
                ncpath = @(var) fullfile(data_dir_2, var, sprintf('%s.pop.h.%s.%s.nc', runname, var, date_str));

                nc = netcdf(ncpath('TEMP'));
                time = nc{'time'}(:);
                z    = nc{'z_t'}(1:nz) / 100.;
                temp = nc{'TEMP'}(:, 1:nz, mylat, mylon);
                [yr, mon, ~] = datenumnoleap(time - 29, [0 1 1]);

                nc = netcdf(ncpath('UVEL')); uvel = nc{'UVEL'}(:, 1:nz, mylat, mylon) / 100.;
                nc = netcdf(ncpath('VVEL')); vvel = nc{'VVEL'}(:, 1:nz, mylat, mylon) / 100.;
                nc = netcdf(ncpath('WVEL')); wvel = nc{'WVEL'}(:, 1:nz, mylat, mylon) / 100.;
                nc = netcdf(ncpath('SHF'));     qnet = nc{'SHF'}(:, mylat, mylon);
                nc = netcdf(ncpath('SHF_QSW')); qsw  = nc{'SHF_QSW'}(:, mylat, mylon);
                nc = netcdf(ncpath('HMXL'));    mld  = nc{'HMXL'}(:, mylat, mylon) / 100.;

            elseif data_format_option == 3
                % 格式3路径：解析年月，预分配内存
                data_dir_3 = fullfile(base_dir, runname, 'ocn', 'hist');

                yr_start = str2double(date_str(1:4));
                mo_start = str2double(date_str(5:6));
                yr_end   = str2double(date_str(8:11));
                mo_end   = str2double(date_str(12:13));
                
                n_months = (yr_end - yr_start)*12 + (mo_end - mo_start) + 1;
                
                time = zeros(n_months, 1);
                temp = zeros(n_months, nz, nlat, nlon);
                uvel = zeros(n_months, nz, nlat, nlon);
                vvel = zeros(n_months, nz, nlat, nlon);
                wvel = zeros(n_months, nz, nlat, nlon);
                qnet = zeros(n_months, nlat, nlon);
                qsw  = zeros(n_months, nlat, nlon);
                mld  = zeros(n_months, nlat, nlon);
                
                count = 1;
                for y = yr_start:yr_end
                    for m = 1:12
                        if (y == yr_start && m < mo_start) || (y == yr_end && m > mo_end)
                            continue;
                        end
                        
                        % 读取各月文件
                        month_file = sprintf('%s.pop.h.%04d-%02d.nc', runname, y, m);
                        filepath = fullfile(data_dir_3, month_file); 
                        
                        nc = netcdf(filepath);
                        
                        if count == 1
                            z = nc{'z_t'}(1:nz) / 100.;
                        end
                        
                        time(count) = nc{'time'}(1);
                        temp(count, :, :, :) = nc{'TEMP'}(1, 1:nz, mylat, mylon);
                        uvel(count, :, :, :) = nc{'UVEL'}(1, 1:nz, mylat, mylon) / 100.;
                        vvel(count, :, :, :) = nc{'VVEL'}(1, 1:nz, mylat, mylon) / 100.;
                        wvel(count, :, :, :) = nc{'WVEL'}(1, 1:nz, mylat, mylon) / 100.;
                        qnet(count, :, :)    = nc{'SHF'}(1, mylat, mylon);
                        qsw(count, :, :)     = nc{'SHF_QSW'}(1, mylat, mylon);
                        mld(count, :, :)     = nc{'HMXL'}(1, mylat, mylon) / 100.;
                        
                        count = count + 1;
                    end
                end
                [yr, mon, ~] = datenumnoleap(time - 29, [0 1 1]);

            else
                error('未知的读取选项');
            end

            % -------------------------------------------------------------
            % 异常缺测值清洗模块 (非常关键！)
            % CESM 的海陆掩码/缺测值通常是 1e36，必须在运算前转为 NaN
            % -------------------------------------------------------------
            temp(abs(temp) > 1e30) = NaN;
            uvel(abs(uvel) > 1e30) = NaN;
            vvel(abs(vvel) > 1e30) = NaN;
            wvel(abs(wvel) > 1e30) = NaN;
            qnet(abs(qnet) > 1e30) = NaN;
            qsw(abs(qsw) > 1e30)   = NaN;
            mld(abs(mld) > 1e30)   = NaN;

            % -------------------------------------------------------------
            % 物理量计算模块
            % -------------------------------------------------------------
            qpen = qsw .* (0.58 .* exp(-mld ./ 0.35) + 0.42 .* exp(-mld ./ 23.0));
            sfcflx = (qnet - qpen) ./ (rho_cp .* mld);

            pacz = repmat(z, [1, size(temp,3), size(temp,4)]);
            
            Tmld = mldavg_varytime(mld, temp, time, pacz);
            Tmld(abs(Tmld) > 1e10) = NaN; 

            umld = mldavg_varytime(mld, uvel, time, pacz);
            vmld = mldavg_varytime(mld, vvel, time, pacz);

            Tsub = submld_varytime(mld, temp, time, pacz, 'first');
            usub = submld_varytime(mld, uvel, time, pacz, 'first');
            vsub = submld_varytime(mld, vvel, time, pacz, 'first');
            wsub = submld_varytime(mld, wvel, time, pacz, 'first');

            [umdTmdx,updTmdx,umdTpdx,updTpdx,vmdTmdy,vpdTmdy,vmdTpdy,vpdTpdy,dTdt,mnupdTpdx,mnvpdTpdy] = ...
                advection_ml_rd(Tmld, umld, vmld, tlat, tlon, time, yr, mon, [min(yr), max(yr)]);

            %dTdt : K/day ==> K/s：
            dTdt = dTdt ./ 86400.0;

            [w_entr,wmdTmdz,wpdTmdz,wmdTpdz,wpdTpdz,mnwpdTpdz] = ...
                vertadv_ml_rd(time, tlat, tlon, mld, usub, vsub, Tmld, Tsub, wsub, mon, yr, [min(yr), max(yr)]);

            % -------------------------------------------------------------
            % 清理与保存模块 (保存为 .nc 文件)
            % -------------------------------------------------------------
            clear temp uvel vvel wvel qnet qsw vmld 
            
            if ~exist(out_dir, 'dir'), mkdir(out_dir); end

            % mat_name = sprintf('%s_heatbudget_ml_rd_%s_varyMLD.mat', runname, date_str);
            % mat_path = fullfile(out_dir, mat_name);
            % save(mat_path);
            % fprintf('已完成并保存: %s\n', mat_name);

            nc_name = sprintf('%s_heatbudget_ml_rd_%s_varyMLD.nc', runname, date_str);
            nc_path = fullfile(out_dir, nc_name);

            out_data = struct('time', time, 'tlat', tlat, 'tlon', tlon, 'mld', mld, ...
                              'sfcflx', sfcflx, 'Tmld', Tmld, 'dTdt', dTdt, ...
                              'umdTmdx', umdTmdx, 'updTmdx', updTmdx, 'umdTpdx', umdTpdx, 'updTpdx', updTpdx, ...
                              'vmdTmdy', vmdTmdy, 'vpdTmdy', vpdTmdy, 'vmdTpdy', vmdTpdy, 'vpdTpdy', vpdTpdy, ...
                              'w_entr', w_entr, 'wmdTmdz', wmdTmdz, 'wpdTmdz', wpdTmdz, 'wmdTpdz', wmdTpdz, 'wpdTpdz', wpdTpdz, ...
                              'mnupdTpdx', mnupdTpdx, 'mnvpdTpdy', mnvpdTpdy, 'mnwpdTpdz', mnwpdTpdz);

            write_budget_nc(nc_path, out_data);
            
            fprintf('已完成并保存: %s\n', nc_name);
        end
    end
end



% =========================================================================
% 局部函数 (Local Functions) - 脚本文件的最底部
% =========================================================================

function [yr, mon] = get_noleap_date(time_days)
    % GET_NOLEAP_DATE 用于解析 CESM 的无闰年(noleap)日历
    % 输入: time_days (单位: days since 0000-01-01)
    % 输出: yr (年份), mon (月份)
    
    % 将时间戳回退29天，使原本月末的时间戳落入当前月份，防止跨月计算错误
    t_shift = time_days - 29; 
    
    % 每年固定 365 天
    yr = floor(t_shift / 365);
    doy = mod(t_shift, 365); 
    
    % noleap 日历下每个月开始对应的天数 (Day of Year)
    month_bounds = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    
    mon = zeros(size(doy));
    for m = 1:12
        mon(doy >= month_bounds(m)) = m;
    end
end


function [yr, mon, day] = datenumnoleap(time_days, ref_date)
    % DATENUMNOLEAP 根据无闰年(noleap)日历计算对应的年月日
    % 兼容标准的 datenumnoleap 函数接口
    % 输入: 
    %   time_days - 从参考日期起算的绝对天数 (数组或标量)
    %   ref_date  - 参考日期向量，格式为 [年, 月, 日]，通常为 [0 1 1]
    % 输出:
    %   yr, mon, day - 分别为年、月、日数组
    
    if nargin < 2
        ref_date = [0 1 1]; % 默认参考时间为 0000-01-01
    end
    
    % noleap 日历下每个月开始对应的累积天数
    month_bounds = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    
    % 计算参考日期距离 年初(X年-01-01) 的偏差天数
    ref_days = ref_date(1)*365 + month_bounds(ref_date(2)) + (ref_date(3) - 1);
    
    % 计算总绝对天数
    abs_days = time_days + ref_days;
    
    % 提取年份，每年固定 365 天
    yr = floor(abs_days / 365);
    
    % 提取当年内天数 Day of Year (范围：0~364)
    doy = mod(abs_days, 365); 
    
    % 计算所属月份和日期
    mon = zeros(size(doy));
    day = zeros(size(doy));
    for m = 1:12
        idx = (doy >= month_bounds(m));
        mon(idx) = m;
        day(idx) = doy(idx) - month_bounds(m) + 1;
    end
end


function write_budget_nc(filename, data)
    % 如果文件已存在，先删除以避免维度冲突
    if exist(filename, 'file')
        delete(filename);
    end
    
    % 获取 3D 变量的基准维度
    [Nt, Nlat, Nlon] = size(data.Tmld);
    
    % --- 1. 写入坐标变量 ---
    nccreate(filename, 'time', 'Dimensions', {'time', Inf}, 'Format', '64bit');
    ncwrite(filename, 'time', data.time);
    ncwriteatt(filename, 'time', 'units', 'days since 0000-01-01 00:00:00');
    ncwriteatt(filename, 'time', 'calendar', 'noleap');
    
    % 为 12xNxN 的气候态变量创建一个专门的 month 维度
    nccreate(filename, 'month', 'Dimensions', {'month', 12});
    ncwrite(filename, 'month', (1:12)');
    
    nccreate(filename, 'TLAT', 'Dimensions', {'lon', Nlon, 'lat', Nlat});
    ncwrite(filename, 'TLAT', data.tlat'); 
    ncwriteatt(filename, 'TLAT', 'units', 'degrees_north');
    
    nccreate(filename, 'TLONG', 'Dimensions', {'lon', Nlon, 'lat', Nlat});
    ncwrite(filename, 'TLONG', data.tlon');
    ncwriteatt(filename, 'TLONG', 'units', 'degrees_east');
    
    % --- 2. 批量写入物理量 ---
    vars = fieldnames(data);
    for i = 1:length(vars)
        vname = vars{i};
        
        if ~ismember(vname, {'time', 'tlat', 'tlon'})
            var_data = data.(vname);
            var_size = size(var_data);
            
            % 情况A：完整的 3D 变量 (time, lat, lon)
            if isequal(var_size, [Nt, Nlat, Nlon])
                var_data_permuted = permute(var_data, [3, 2, 1]);
                nccreate(filename, vname, 'Dimensions', {'lon', Nlon, 'lat', Nlat, 'time', Inf}, 'Datatype', 'double');
                ncwrite(filename, vname, var_data_permuted);
                ncwriteatt(filename, vname, '_FillValue', NaN);
                
            % 情况B：12个月的气候态 3D 变量 (12, lat, lon)，专门处理 mnwpdTpdz
            elseif length(var_size) == 3 && var_size(1) == 12 && var_size(2) == Nlat && var_size(3) == Nlon
                var_data_permuted = permute(var_data, [3, 2, 1]);
                nccreate(filename, vname, 'Dimensions', {'lon', Nlon, 'lat', Nlat, 'month', 12}, 'Datatype', 'double');
                ncwrite(filename, vname, var_data_permuted);
                ncwriteatt(filename, vname, '_FillValue', NaN);

            % 情况C：2D 变量 (lat, lon)
            elseif isequal(var_size, [Nlat, Nlon])
                var_data_permuted = permute(var_data, [2, 1]);
                nccreate(filename, vname, 'Dimensions', {'lon', Nlon, 'lat', Nlat}, 'Datatype', 'double');
                ncwrite(filename, vname, var_data_permuted);
                ncwriteatt(filename, vname, '_FillValue', NaN);
                
            else
                fprintf('警告: 变量 %s 的维度 %s 无法识别，已被跳过。\n', vname, mat2str(var_size));
            end
        end
    end
end
