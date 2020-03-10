# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4
import datetime
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

#--------------------

if (len(sys.argv) < 3):
    print ('Usage: ' + sys.argv[0] + ' mip_id model_name start_year end_year')
    sys.exit()

mip = sys.argv[1]
model_filt = sys.argv[2]
styr = int(sys.argv[3])
edyr = int(sys.argv[4])

#--------------------

if (mip == 'omip1'):
    start_yr = 1948
elif (mip == 'omip2'):
    start_yr = 1958
else:
    print(' Invalid mip id :', mip)
    sys.exit()

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

metainfo = json.load(open("./json/zos_" + mip + ".json"))
model_list = metainfo.keys()

#----------------------------------------------

path_out   = 'OMIP_out'

for model in metainfo.keys():

    if (model != model_filt):
        continue

    sshfile = metainfo[model]['path'] + '/' + metainfo[model]['fname']
    undef_value = float(metainfo[model]['undef'])
    undef_nan = metainfo[model]['undefnan']
    gtorlt = metainfo[model]['gtorlt']
    sshvname = metainfo[model]['name']
    lonname = metainfo[model]['lonname']
    latname = metainfo[model]['latname']
    londim = metainfo[model]['londim']
    latdim = metainfo[model]['latdim']

    ncssh = netCDF4.Dataset(sshfile,'r')
    ssh_vars = ncssh.variables[sshvname].ncattrs()

    miss_val_ssh = np.NaN
    if ('_FillValue' in ssh_vars):
        miss_val_ssh = ncssh.variables[sshvname]._FillValue
    elif ('missing_value' in ssh_vars):
        miss_val_ssh = ncssh.variables[sshvname].missing_value

    nx = len(ncssh.dimensions[londim])
    ny = len(ncssh.dimensions[latdim])

    print (model, nx, ny, gtorlt, undef_value, miss_val_ssh)

    lon_ = ncssh.variables[lonname][:]
    lat_ = ncssh.variables[latname][:]

    #############################################
    # Output to netCDF4

    fmon_out = path_out + '/' + 'zos_filter_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'

    lon_bnds_ = np.array(np.empty((nx,2)))
    lat_bnds_ = np.array(np.empty((ny,2)))

    lon_bnds_[:,0] = lon_[:] - 0.5
    lon_bnds_[:,1] = lon_[:] + 0.5
    lat_bnds_[:,0] = lat_[:] - 0.5
    lat_bnds_[:,1] = lat_[:] + 0.5

    #-----

    ncmon = netCDF4.Dataset(fmon_out, 'w', format='NETCDF4')

    ncmon.createDimension('lon', nx)
    ncmon.createDimension('lat', ny)
    ncmon.createDimension('bnds', 2)
    ncmon.createDimension('time', None)

    lat = ncmon.createVariable('lat', np.dtype('float').char, ('lat'))
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'
    lat.standard_name = 'latitude'
    lat_bnds = ncmon.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

    lon = ncmon.createVariable('lon', np.dtype('float').char, ('lon'))
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon.standard_name = 'latitude'
    lon_bnds = ncmon.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

    stdate = str(styr) + '-01-01'
    eddate = str(edyr) + '-12-01'
    dtime = pd.date_range(stdate,eddate,freq='MS')
    dtime_start = datetime.date(styr, 1, 1)
    td=pd.to_datetime(dtime[:]).date - dtime_start
    time = ncmon.createVariable('time', np.dtype('int32').char, ('time',))
    time.units = 'days since ' + str(styr) + '-01-01 00:00:00'
    time.axis = 'T'

    zosmon = ncmon.createVariable('zos', np.dtype(np.float64).char, ('time', 'lat', 'lon'), zlib=True)
    zosmon.long_name = 'sea surface height'
    zosmon.units = 'm'
    zosmon.missing_value = -9.99e33
    
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    time_vars = np.array(np.zeros((len(td))))
    for i in range(len(td)):
        time_vars[i] = td[i].days

    time[:]=time_vars

    ncmon.description="Monthly climatology " + str(styr) + " through " + str(edyr)

    #----------------------------------------------
    # read data and store in np.array

    ssh_tmp = np.array(np.zeros((ny,nx)),dtype=np.float64)
    mask_model = np.array(np.ones((ny,nx)),dtype=np.int64)

    for yr in range(styr,edyr+1):

        rec_base = (yr-start_yr)*12

        for mn in range(1,13):

            recn = rec_base + mn - 1
            recd = (yr-styr)*12 + mn - 1

            if (yr == styr or yr == edyr):
                if (mn == 12):
                    print (yr,mn,recn,recd)

            ssh = ncssh.variables['zos'][recn,:,:]

            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flags = (ssh > undef_value)
                else:
                    undef_flags = (ssh < undef_value)
            else:
                undef_flags = np.isnan(ssh)

            ssh[undef_flags] = np.NaN

            mask_model = np.where(np.isnan(ssh), 0, mask_model)

            for j in range(1,ny-1):
                for i in range(1,nx-1):
                    if (np.isnan(ssh[j,i])):
                        continue
                    else:
                        wgt = 4 * mask_model[j,i] + mask_model[j,i-1] \
                            + mask_model[j,i+1] + mask_model[j-1,i] + mask_model[j+1,i]
                        ssh_tmp[j,i] = (4.0 * np.float(mask_model[j,i]) * ssh[j,i] \
                                        + np.float(mask_model[j,i-1]) * ssh[j,i-1] \
                                        + np.float(mask_model[j,i+1]) * ssh[j,i+1] \
                                        + np.float(mask_model[j-1,i]) * ssh[j-1,i] \
                                        + np.float(mask_model[j+1,i]) * ssh[j+1,i]) \
                                        / np.float(wgt)
               
            j = 0
            for i in range(1,nx-1):
                if (np.isnan(ssh[j,i])):
                    continue
                else:
                    wgt = 4.0 * mask_model[j,i] + mask_model[j,i-1] \
                        + mask_model[j,i+1] + mask_model[j+1,i]
                    ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                    + mask_model[j,i-1] * ssh[j,i-1] \
                                    + mask_model[j,i+1] * ssh[j,i+1] \
                                    + mask_model[j+1,i] * ssh[j+1,i]) / wgt
            j = ny-1
            for i in range(1,nx-1):
                if (np.isnan(ssh[j,i])):
                    continue
                else:
                    wgt = 4.0 * mask_model[j,i] + mask_model[j,i-1] \
                        + mask_model[j,i+1] + mask_model[j-1,i]
                    ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                    + mask_model[j,i-1] * ssh[j,i-1] \
                                    + mask_model[j,i+1] * ssh[j,i+1] \
                                    + mask_model[j-1,i] * ssh[j-1,i]) / wgt

            for j in range(1,ny-1):
                i = 0
                if (np.isnan(ssh[j,i])):
                    continue
                else:
                    wgt = 4.0 * mask_model[j,i] + mask_model[j,nx-1] \
                        + mask_model[j,i+1] + mask_model[j-1,i] + mask_model[j+1,i]
                    ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                    + mask_model[j,nx-1] * ssh[j,nx-1] \
                                    + mask_model[j,i+1] * ssh[j,i+1] \
                                    + mask_model[j-1,i] * ssh[j-1,i] \
                                    + mask_model[j+1,i] * ssh[j+1,i]) / wgt
                i = nx - 1
                if (np.isnan(ssh[j,i])):
                    continue
                else:
                    wgt = 4.0 * mask_model[j,i] + mask_model[j,i-1] \
                        + mask_model[j,0] + mask_model[j-1,i] + mask_model[j+1,i]
                    ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                    + mask_model[j,i-1] * ssh[j,i-1] \
                                    + mask_model[j,0] * ssh[j,0] \
                                    + mask_model[j-1,i] * ssh[j-1,i] \
                                    + mask_model[j+1,i] * ssh[j+1,i]) / wgt
                    
            j = 0
            i = 0
            if (~np.isnan(ssh[j,i])):
                wgt = 4.0 * mask_model[j,i] + mask_model[j,nx-1] \
                    + mask_model[j,i+1] + mask_model[j+1,i]
                ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                + mask_model[j,nx-1] * ssh[j,nx-1] \
                                + mask_model[j,i+1] * ssh[j,i+1] \
                                + mask_model[j+1,i] * ssh[j+1,i]) / wgt
            i = nx - 1
            if (~np.isnan(ssh[j,i])):
                wgt = 4.0 * mask_model[j,i] + mask_model[j,i-1] \
                    + mask_model[j,0] + mask_model[j+1,i]
                ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                + mask_model[j,i-1] * ssh[j,i-1] \
                                + mask_model[j,0] * ssh[j,0] \
                                + mask_model[j+1,i] * ssh[j+1,i]) / wgt

            j = ny-1
            i = 0
            if (~np.isnan(ssh[j,i])):
                wgt = 4.0 * mask_model[j,i] + mask_model[j,nx-1] \
                    + mask_model[j,i+1] + mask_model[j-1,i]
                ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                + mask_model[j,nx-1] * ssh[j,nx-1] \
                                + mask_model[j,i+1] * ssh[j,i+1] \
                                + mask_model[j-1,i] * ssh[j-1,i]) / wgt
            i = nx - 1
            if (~np.isnan(ssh[j,i])):
                wgt = 4.0 * mask_model[j,i] + mask_model[j,i-1] \
                    + mask_model[j,0] + mask_model[j-1,i]
                ssh_tmp[j,i] = (4.0 * mask_model[j,i] * ssh[j,i] \
                                + mask_model[j,i-1] * ssh[j,i-1] \
                                + mask_model[j,0] * ssh[j,0] \
                                + mask_model[j-1,i] * ssh[j-1,i]) / wgt
                
            ssh = ssh_tmp
            ssh[undef_flags] = np.NaN
            print(recd)
            zosmon[recd,:,:]=np.where(np.isnan(ssh), -9.99e33, ssh)
            
    del ssh_tmp
    print(mask_model)

    ncssh.close()
    ncmon.close()

    del time_vars
    del lon_bnds_
    del lat_bnds_

    del mask_model
