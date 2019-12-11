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
    print ('Usage: ' + sys.argv[0] + ' start_year end_year filtered(yes or no)')
    sys.exit()

styr = int(sys.argv[1])
edyr = int(sys.argv[2])
filtered = sys.argv[3]

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

print( "Loading CMEMS data" )

path_cmems = '../refdata/CMEMS'
path_amip = '../refdata/PCMDI-SST'

if (filtered == 'yes'): 
    sshfile = path_cmems + '/' + 'zos_adt_filter_CMEMS_1x1_monthly_199301-201812.nc'
else:
    sshfile = path_cmems + '/' + 'zos_adt_CMEMS_1x1_monthly_199301-201812.nc'

arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncssh = netCDF4.Dataset(sshfile,'r')
#miss_val_ssh = ncssh.variables['zos'].missing_value
miss_val_ssh = -9.00e33
nx = len(ncssh.dimensions['lon'])
ny = len(ncssh.dimensions['lat'])

print (nx,ny)

lon_ = ncssh.variables['lon'][:]
lat_ = ncssh.variables['lat'][:]

ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]

#############################################
# Output to netCDF4

path_cmems_out = '../analysis/SSH/CMEMS'

if (filtered == 'yes'): 
    fann_out = path_cmems_out + '/' + 'zos_filter_annual_gn_' + str(styr) + '-' + str(edyr) + '.nc'
else:
    fann_out = path_cmems_out + '/' + 'zos_annual_gn_' + str(styr) + '-' + str(edyr) + '.nc'

lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))

lon_bnds_[:,0] = lon_[:] - 0.5
lon_bnds_[:,1] = lon_[:] + 0.5
lat_bnds_[:,0] = lat_[:] - 0.5
lat_bnds_[:,1] = lat_[:] + 0.5

ncann = netCDF4.Dataset(fann_out, 'w', format='NETCDF4')

ncann.createDimension('lon', nx)
ncann.createDimension('lat', ny)
ncann.createDimension('bnds', 2)
ncann.createDimension('time', None)

lat = ncann.createVariable('lat', np.dtype('float').char, ('lat'))
lat.latg_name = 'latitude'
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

ncann.description="Annual mean " + str(styr) + " through " + str(edyr)

#----------------------------------------------
# read data and store in np.array

ssh_ann = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_ann = np.array(np.zeros((ny,nx)),dtype=np.int64)
mask_cmems = np.array(np.ones((ny,nx)),dtype=np.int64)

start_yr = 1993

nout = 0
for yr in range(styr,edyr+1):

    rec_base = (yr-start_yr)*12
    ssh_ann = 0.0
    day_ann = 0
    mask_cmems = 1.0

    for mn in range(1,13):

        recn = rec_base + mn - 1
        print (yr,mn,recn)
        ssh_tmp = ncssh.variables['zos'][recn,:,:]
        ssh = ssh_tmp.astype(np.float64)

        undef_flags = (ssh < miss_val_ssh)
        ssh[undef_flags] = np.NaN
        mask_cmems = np.where(np.isnan(ssh), 0, mask_cmems)

        ssh_ann = ssh_ann + mask_cmems * ssh * mon_days[mn-1]
        day_ann = day_ann + mask_cmems * mon_days[mn-1]

    ssh_ann = mask_cmems * ssh_ann / (1 - mask_cmems + day_ann)
    zosann[nout,:,:]=np.where(mask_cmems == 0.0, -9.99e33, ssh_ann)
    nout = nout + 1

ncann.close()

