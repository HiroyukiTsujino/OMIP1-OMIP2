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

if len(sys.argv) < 3:
    print ('Usage: ' + sys.argv[0] + ' start_year end_year')
    sys.exit()

styr = int(sys.argv[1])
edyr = int(sys.argv[2])

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

path_input = 'Ishii_v7_2/v7.2/temp/netcdf'

v700mon = path_input + '/' + 'vat700_' + str(styr) + '01-' + str(edyr) + '12.nc'

ncmon = netCDF4.Dataset(v700mon,'r')

nt = len(ncmon.dimensions['time'])
ny = len(ncmon.dimensions['lat'])
nx = len(ncmon.dimensions['lon'])

lon_ = ncmon.variables['lon'][:]
lat_ = ncmon.variables['lat'][:]
miss_val_vat = ncmon.variables['thetao'].missing_value

mask = np.array(np.zeros((ny,nx)),dtype=np.float64)
maskv = np.array(np.zeros((ny,nx)),dtype=np.float64)
vat_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)

lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))
lon_bnds_[:,0] = lon_[:] - 0.5
lon_bnds_[:,1] = lon_[:] + 0.5
lat_bnds_[:,0] = lat_[:] - 0.5
lat_bnds_[:,1] = lat_[:] + 0.5

#--------------------------------------------

v700file = path_input + '/' + 'vat700_' + str(styr) + '-' + str(edyr) + '.nc'
ncv700 = netCDF4.Dataset(v700file, 'w', format='NETCDF4')
ncv700.createDimension('lon', nx)
ncv700.createDimension('lat', ny)
ncv700.createDimension('bnds', 2)
ncv700.createDimension('time', None)

strmn = str(styr) + '-01-01'
endmn = str(edyr) + '-01-01'

dtime = pd.date_range(strmn,endmn,freq='AS-JAN')
dtime_start = datetime.date(1955, 1, 1)
time = ncv700.createVariable('time', np.dtype('int64').char, ('time',))
time.long_name = 'time of annual mean vertically averaged temperature '
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

#-----------------------------------------------------

nrec=0
start_yr = 1955
for yr in range(styr,edyr+1):

    rec_base = (yr-start_yr)*12
    vat_annclim[:,:] = 0.0
    day_annclim[:,:] = 0.0
    
    for mn in range(1,13):

        recn = rec_base + mn - 1
        print (yr,mn,recn)
        vat_tmp = ncmon.variables['thetao'][recn,:,:]
        vat = vat_tmp.astype(np.float64)

        mask = np.where((vat < miss_val_vat*0.9), 0.0, 1.0)

        vat_annclim = vat_annclim + mask * vat * mon_days[mn-1]
        day_annclim = day_annclim + mask * mon_days[mn-1]

    maskv = np.where((day_annclim == 0.0), 0.0, 1.0)
    vat_annclim = maskv * vat_annclim / (1.0 - maskv + day_annclim)
    thetao[nrec,:,:] = vat_annclim
    nrec = nrec + 1
    
ncv700.close()
