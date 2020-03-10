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

if len(sys.argv) == 1:
    print ('Usage: ' + sys.argv[0] + ' start_year end_year')
    sys.exit()

styr = int(sys.argv[1])
edyr = int(sys.argv[2])

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

print( "Loading AMIP data" )

path_amip = '../refdata/PCMDI-SST'
path_out = '../analysis/SST/PCMDI-SST'

sstfile = path_amip + '/' + 'tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
mskfile = path_amip + '/' + 'sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncsst = netCDF4.Dataset(sstfile,'r')
miss_val_sst = ncsst.variables['tos'].missing_value
nx = len(ncsst.dimensions['lon'])
ny = len(ncsst.dimensions['lat'])

print (nx,ny)

lon_ = ncsst.variables['lon'][:]
lat_ = ncsst.variables['lat'][:]

ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]

ncmsk = netCDF4.Dataset(mskfile,'r')
mask = ncmsk.variables['sftof'][:,:]

# Ad hoc modification of the Kaspian Sea
for j in range(ny):
    for i in range(nx):
        if (45 < lon_[i]) & (lon_[i] < 60):
             if (34 < lat_[j]) & (lat_[j] < 50):
                  mask[j,i] = 0

        if mask[j,i] < 100:
            mask[j,i] = 0

mask = mask / 100

#----------------------------------------------
# Output to netCDF4

fann_out = path_out + '/AMIP-' + 'tos_annual_gn_' + str(styr) + '-' + str(edyr) + '.nc'
ncann = netCDF4.Dataset(fann_out, 'w', format='NETCDF4')

lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))

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

print(dtime)

td=pd.to_datetime(dtime[:]).date - dtime_start
time = ncann.createVariable('time', np.dtype('int32').char, ('time',))
time.units = 'days since 1850-01-01 00:00:00'
time.axis = 'T'

tosann = ncann.createVariable('tos', np.dtype(np.float64).char, ('time', 'lat', 'lon'), zlib=True)
tosann.long_name = 'sea surface temperature'
tosann.units = 'degC'
tosann.missing_value = -9.99e33

lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

time_vars = np.array(np.zeros((len(td))))
for i in range(len(td)):
  time_vars[i] = td[i].days

time[:]=time_vars

ncann.description="Time series of annual mean " + str(styr) + " through " + str(edyr)

#--------------------------------------------------------------------------------------
# read data and store in np.array

sst_ann = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_ann = np.array(np.zeros((ny,nx)),dtype=np.int64)
#mask_amip = np.array(np.ones((ny,nx)),dtype=np.int64)

start_yr = 1870

nout = 0
for yr in range(styr,edyr+1):

    rec_base = (yr-start_yr)*12
    sst_ann = 0.0
    day_ann = 0.0

    for mn in range(1,13):

        recn = rec_base + mn - 1
        print (yr,mn,recn)
        sst_tmp = ncsst.variables['tos'][recn,:,:]
        sst = sst_tmp.astype(np.float64)

        undef_flags = (sst == miss_val_sst)
        sst[undef_flags] = np.NaN
        #mask_amip = np.where(np.isnan(sst), 0, mask_amip)

        sst_ann = sst_ann + mask * sst * mon_days[mn-1]
        day_ann = day_ann + mask * mon_days[mn-1]

    sst_ann = np.where(day_ann == 0, np.NaN, sst_ann / day_ann)
    tosann[nout,:,:]=np.where(np.isnan(sst_ann), -9.99e33, sst_ann)
    nout = nout + 1

ncann.close()
