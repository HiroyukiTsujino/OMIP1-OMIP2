# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset, num2date
import datetime


def potential_temp( s, t, p ):
    #- Gill (1982) Eq.(A3.13)
    p10  = -3.6504e-4
    p11  = -8.3198e-5
    p12  =  5.4065e-7
    p13  = -4.0274e-9
    sp10 = -1.7439e-5
    sp11 =  2.9778e-7
    p20  = -8.9309e-7
    p21  =  3.1628e-8
    p22  = -2.1987e-10
    sp20 =  4.1057e-9
    p30  =  1.6056e-10
    p31  = -5.0484e-12

    pottemp = t \
        + p * (p10 + t * (p11 + t * (p12  + t * p13)) \
               + (s - 35.0) * (sp10 + t * sp11) \
                + p * (p20 + t * (p21 + t * p22) \
                        + (s - 35.0) * sp20 \
                        + p * (p30 + p31 * t)))

    return pottemp

v_potential_temp = np.vectorize(potential_temp, excluded=['p'])

#------

if (len(sys.argv) < 3) :
    print ('Usage: '+ sys.argv[0] + ' start_year end_year')
    sys.exit()

styr = int(sys.argv[1])
edyr = int(sys.argv[2])

#------

path_input_temp = 'Ishii_v7_2/v7.2/temp/netcdf'
path_input_sal = 'Ishii_v7_2/v7.2/sal/netcdf'

tmpfile = path_input_temp + '/' + 'temp.1955.nc'

nctmp = netCDF4.Dataset(tmpfile,'r')
nt = len(nctmp.dimensions['time'])
nz = len(nctmp.dimensions['depth'])
ny = len(nctmp.dimensions['latitude'])
nx = len(nctmp.dimensions['longitude'])
lon_ = nctmp.variables['longitude'][:]
lat_ = nctmp.variables['latitude'][:]
dep_ = nctmp.variables['depth'][:]
ptemp = np.array(np.zeros((nt,nz,ny,nx)),dtype=np.float64)

time_vars = np.array(np.empty(nt))
lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))
lon_bnds_[:,0] = lon_[:] - 0.5
lon_bnds_[:,1] = lon_[:] + 0.5
lat_bnds_[:,0] = lat_[:] - 0.5
lat_bnds_[:,1] = lat_[:] + 0.5

for yr in range(styr,edyr+1):

    tmpfile = path_input_temp + '/' + 'temp.' + str(yr) + '.nc'
    salfile = path_input_sal + '/' + 'sal.' + str(yr) + '.nc'

    nctmp = netCDF4.Dataset(tmpfile,'r')
    miss_val_tmp = nctmp.variables['var10_4_192']._FillValue

    ncsal = netCDF4.Dataset(salfile,'r')
    miss_val_sal = ncsal.variables['var10_4_193']._FillValue

    for mn in range(1,13):

        print(yr,mn)
        temp = nctmp.variables['var10_4_192'][mn-1,:,:,:]
        sal  = ncsal.variables['var10_4_193'][mn-1,:,:,:]

        undef_flags = (temp > miss_val_tmp * 0.9)
        temp[undef_flags] = np.NaN
        undef_flags = (sal > miss_val_sal * 0.9)
        sal [undef_flags] = np.NaN

        for k in range (nz):
            prsbar = dep_[k] * 0.1
            ptemp[mn-1,k,:,:]= v_potential_temp(sal[k,:,:],temp[k,:,:],prsbar)

    ptemp = np.where(np.isnan(ptemp),-9.99e33,ptemp)

    nctmp.close()
    ncsal.close()

    ptmfile = path_input_temp + '/' + 'ptmp.' + str(yr) + '.nc'
    ncptmp = netCDF4.Dataset(ptmfile, 'w', format='NETCDF4')
    ncptmp.createDimension('lon', nx)
    ncptmp.createDimension('lat', ny)
    ncptmp.createDimension('bnds', 2)
    ncptmp.createDimension('depth', nz)
    ncptmp.createDimension('time', None)

    strmn = str(yr) + '-01-01'
    endmn = str(yr) + '-12-01'
    dtime = pd.date_range(strmn,endmn,freq='MS')
    dtime_start = datetime.date(1955, 1, 1)
    time = ncptmp.createVariable('time', np.dtype('int64').char, ('time',))
    time.long_name = 'time of monthly mean potential temperature '
    time.units = 'days since 1955-01-01 00:00:00'

    depth = ncptmp.createVariable('depth', np.dtype('float').char, ('depth'))
    depth.long_name = 'depth below sea level'
    depth.units = 'm'
    depth.axis = 'Z'
    depth.standard_name = 'depth'

    lat = ncptmp.createVariable('lat', np.dtype('float').char, ('lat'))
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'
    lat.standard_name = 'latitude'
    lat_bnds = ncptmp.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

    lon = ncptmp.createVariable('lon', np.dtype('float').char, ('lon'))
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon.standard_name = 'latitude'
    lon_bnds = ncptmp.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

    thetao = ncptmp.createVariable('thetao', np.dtype('float').char, ('time','depth','lat','lon'), zlib=True)
    thetao.long_name = 'water potential temparture'
    thetao.units = 'degC'
    thetao.missing_value = -9.99e33

    td=pd.to_datetime(dtime[:]).date - dtime_start

    if (len(td) != nt):
        print(' time axis is not consistent ')
        exit
    else:
        for i in range(len(td)):
            time_vars[i] = td[i].days

    time[:]=time_vars
    lat[:]=lat_
    lon[:]=lon_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_
    depth[:]=dep_
    thetao[:,:,:,:] = ptemp

    ncptmp.close()
