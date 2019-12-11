# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

#------

if (len(sys.argv) < 3) :
    print ('Usage: '+ sys.argv[0] + ' start_year end_year')
    sys.exit()

styr = int(sys.argv[1])
edyr = int(sys.argv[2])

#------

path_input = 'Ishii_v7_2/v7.2/temp/netcdf'

tmpfile = path_input + '/' + 'ptmp.1955.nc'

nctmp = netCDF4.Dataset(tmpfile,'r')

nt = len(nctmp.dimensions['time'])
nz = len(nctmp.dimensions['depth'])
ny = len(nctmp.dimensions['lat'])
nx = len(nctmp.dimensions['lon'])

lon_ = nctmp.variables['lon'][:]
lat_ = nctmp.variables['lat'][:]
dep_ = nctmp.variables['depth'][:]

mask = np.array(np.zeros((nz,ny,nx)),dtype=np.float64)
vat700 = np.array(np.zeros((ny,nx)),dtype=np.float64)
vatdep = np.array(np.zeros((ny,nx)),dtype=np.float64)
maskv = np.array(np.zeros((ny,nx)),dtype=np.float64)

lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))
lon_bnds_[:,0] = lon_[:] - 0.5
lon_bnds_[:,1] = lon_[:] + 0.5
lat_bnds_[:,0] = lat_[:] - 0.5
lat_bnds_[:,1] = lat_[:] + 0.5

v700file = path_input + '/' + 'vat700_' + str(styr) + '01-' + str(edyr) + '12.nc'
ncv700 = netCDF4.Dataset(v700file, 'w', format='NETCDF4')
ncv700.createDimension('lon', nx)
ncv700.createDimension('lat', ny)
ncv700.createDimension('bnds', 2)
ncv700.createDimension('time', None)

strmn = str(styr) + '-01-01'
endmn = str(edyr) + '-12-01'

dtime = pd.date_range(strmn,endmn,freq='MS')
dtime_start = datetime.date(1955, 1, 1)
time = ncv700.createVariable('time', np.dtype('int64').char, ('time',))
time.long_name = 'time of monthly mean vertically averaged temperature '
time.units = 'days since 1955-01-01 00:00:00'

lat = ncv700.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncv700.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

lon = ncv700.createVariable('lon', np.dtype('float').char, ('lon'))
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'
lon.standard_name = 'latitude'
lon_bnds = ncv700.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

thetao = ncv700.createVariable('thetao', np.dtype('float').char, ('time','lat','lon'), zlib=True)
thetao.long_name = 'water potential temparture'
thetao.units = 'degC'
thetao.missing_value = -9.99e33

td=pd.to_datetime(dtime[:]).date - dtime_start

time_vars = np.array(np.empty(len(td)))
for i in range(len(td)):
    time_vars[i] = td[i].days
      
time[:]=time_vars
lat[:]=lat_
lon[:]=lon_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

nrec=0
for yr in range(styr,edyr+1):

    tmpfile = path_input + '/' + 'ptmp.' + str(yr) + '.nc'

    nctmp = netCDF4.Dataset(tmpfile,'r')
    miss_val_tmp = nctmp.variables['thetao'].missing_value
    
    vat700[:,:] = 0.0

    for mn in range(1,13):

        print(mn,nrec)
        temp = nctmp.variables['thetao'][mn-1,:,:,:]
        mask[:,:,:] = np.where((temp[:,:,:] < miss_val_tmp*0.9),0.0,1.0)
        vatdep[:,:] = 0.0
        k = 0
        while (dep_[k+1] <= 700.0):
            vat700[:,:] = vat700[:,:] + mask[k,:,:] * 0.5 * (temp[k,:,:] + temp[k+1,:,:]) * (dep_[k+1]-dep_[k])
            vatdep[:,:] = vatdep[:,:] + mask[k,:,:] * (dep_[k+1] - dep_[k])
            k = k + 1

        maskv[:,:] = np.where(vatdep[:,:]==0.0, 0.0, 1.0)
        vat700[:,:] = maskv[:,:] * vat700[:,:] / (1.0 - maskv[:,:] + vatdep[:,:])
        vat700[:,:] = np.where(maskv[:,:]==0.0, miss_val_tmp, vat700[:,:])
        thetao[nrec,:,:] = vat700[:,:]
        nrec = nrec + 1

    nctmp.close()


ncv700.close()
