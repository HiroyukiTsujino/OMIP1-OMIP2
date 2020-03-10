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

if (len(sys.argv) < 5):
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year end_year [modelname or all]')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])
modelname = sys.argv[4]

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

path_out = '../analysis/SSH/MODEL'

for model in metainfo.keys():

    if ( modelname != 'all' ):
        if ( model != modelname ):
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

    print (model, nx, ny, gtorlt, miss_val_ssh)

    lon_ = ncssh.variables[lonname][:]
    lat_ = ncssh.variables[latname][:]

    if ( model == 'NorESM-BLOM' ):
        lon_rot = np.roll(lon_,180)
        lon_ = lon_rot
        lon_ = np.where(lon_ < 0.0, lon_+360, lon_)
        print(' NorESM-BLOM longitude rolled ')
        print(lon_)
    
    if ( model == 'MIROC-COCO4.9' ):
        lat_flip = np.flip(lat_)
        lat_ = lat_flip
        print(' MIROC-COCO4.9 latitude flipped ')
        print(lat_)
    
    #############################################
    # Output to netCDF4

    fann_out = path_out + '/' + 'zos_annual_' + model + '_' + mip + '_' + str(styr) + '-' + str(edyr) + '.nc'

    lon_bnds_ = np.array(np.empty((nx,2)))
    lat_bnds_ = np.array(np.empty((ny,2)))

    lon_bnds_[:,0] = lon_[:] - 0.5
    lon_bnds_[:,1] = lon_[:] + 0.5
    lat_bnds_[:,0] = lat_[:] - 0.5
    lat_bnds_[:,1] = lat_[:] + 0.5

    #-----

    ncann = netCDF4.Dataset(fann_out, 'w', format='NETCDF4')
    ncann.description="Annual mean " + str(styr) + " through " + str(edyr)

    ncann.createDimension('lon', nx)
    ncann.createDimension('lat', ny)
    ncann.createDimension('bnds', 2)
    ncann.createDimension('time', None)

    lat = ncann.createVariable('lat', np.dtype('float').char, ('lat'))
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'
    lat.standard_name = 'latitude'
    lat_bnds = ncann.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

    lon = ncann.createVariable('lon', np.dtype('float').char, ('lon'))
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon.standard_name = 'latitude'
    lon_bnds = ncann.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

    stdate=str(styr) + '-01-01'
    eddate=str(edyr) + '-12-31'

    dtime = pd.date_range(stdate,eddate,freq='AS-JAN')
    dtime_start = datetime.date(1850, 1, 1)
    td=pd.to_datetime(dtime[:]).date - dtime_start
    time = ncann.createVariable('time', np.dtype('int32').char, ('time',))
    time.units = 'days since 1850-01-01 00:00:00'
    time.axis = 'T'

    zosann = ncann.createVariable('zos', np.dtype(np.float64).char, ('time', 'lat', 'lon'), zlib=True)
    zosann.long_name = 'sea surface height'
    zosann.units = 'm'
    zosann.missing_value = -9.99e33
    
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    time_vars = np.array(np.zeros((len(td))))
    for i in range(len(td)):
        time_vars[i] = td[i].days

    time[:]=time_vars

    #----------------------------------------------
    # read data and store in np.array

    ssh_ann = np.array(np.zeros((ny,nx)),dtype=np.float64)
    day_ann = np.array(np.zeros((ny,nx)),dtype=np.int64)
    mask_model = np.array(np.ones((ny,nx)),dtype=np.int64)

    nout = 0
    for yr in range(styr,edyr+1):

        rec_base = (yr-start_yr)*12
        ssh_ann = 0.0
        day_ann = 0
        mask_model = 1

        for mn in range(1,13):

            recn = rec_base + mn - 1
            if (yr == styr or yr == edyr):
                if (mn == 12):
                    print (yr,mn,recn)

            ssh = ncssh.variables['zos'][recn,:,:]

            if ( model == 'NorESM-BLOM' ):
                ssh_rot = np.roll(ssh, 180, axis=1)
                ssh = ssh_rot

            if ( model == 'MIROC-COCO4.9' ):
                ssh_flip = np.flip(ssh, axis=0)
                ssh = ssh_flip

            if ( model == 'Kiel-NEMO' ):
                ssh = np.where(ssh==0.0, np.NaN, ssh)

            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flags = (ssh > undef_value)
                else:
                    undef_flags = (ssh < undef_value)
            else:
                undef_flags = np.isnan(ssh)

            ssh[undef_flags] = np.NaN

            mask_model = np.where(np.isnan(ssh), 0, mask_model)

            ssh_ann = ssh_ann + mask_model * ssh * mon_days[mn-1]
            day_ann = day_ann + mask_model * mon_days[mn-1]

        #print(mask_model)
        ssh_ann = mask_model * ssh_ann / (1 - mask_model + day_ann)
        zosann[nout,:,:]=np.where(mask_model == 0, -9.99e33, ssh_ann)
        nout = nout + 1

    ncann.close()
    ncssh.close()

    del time_vars
    del lon_bnds_
    del lat_bnds_
    del ssh_ann
    del day_ann
    del mask_model
