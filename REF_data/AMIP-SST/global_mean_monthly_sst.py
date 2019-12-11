# -*- coding: utf-8 -*-
import fix_proj_lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

nx = 360
ny = 180

path_amip = 'AMIP_orig'
path_out = 'AMIP'

sstfile = path_amip + '/' + 'tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
mskfile = path_amip + '/' + 'sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncsst = netCDF4.Dataset(sstfile,'r')
miss_val_sst = ncsst.variables['tos'].missing_value

ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]

ncmsk = netCDF4.Dataset(mskfile,'r')
mask = ncmsk.variables['sftof'][:,:]
lon_ = ncmsk.variables['lon'][:]
lat_ = ncmsk.variables['lat'][:]

# Ad hoc modification of the Kaspian Sea (below)
for j in range(ny):
    for i in range (nx):
        if (45 < lon_[i]) & (lon_[i] < 60):
             if (34 < lat_[j]) & (lat_[j] < 50):
                  mask[j,i] = 0

        if mask[j,i] < 100:
            mask[j,i] = 0

mask = mask / 100

# Ad hoc modification of the Kaspian Sea (above)

sstglbmf = path_out + '/netCDF/' + 'AMIPSST_glbm_monthly_194801-201812.nc'

ncsstm = netCDF4.Dataset(sstglbmf, 'w', format='NETCDF4')
ncsstm.createDimension('ntime', None)

dtime = pd.date_range('1948-01-01','2018-12-01',freq='MS')
dtime_start = datetime.date(1948, 1, 1)
time = ncsstm.createVariable('time', np.dtype('int32').char, ('ntime',))
time.long_name = 'time of monthly global mean SST '
time.units = 'days since 1948-01-01 00:00:00'

tosga = ncsstm.createVariable('tosga', np.dtype('double').char, ('ntime'))
tosga.long_name = 'Global average sea surface temperature'
tosga.units = 'degrees_Celsius'

td=pd.to_datetime(dtime[:]).date - dtime_start

time_vars = np.array(np.zeros((len(td))))
sstm_vars = np.array(np.zeros((len(td))))

for i in range(len(td)):
  time_vars[i] = td[i].days


ic = 0

start_yr = 1870

for yr in range(1948,2018):


    rec_base = (yr-1870)*12

    for mn in range(1,13):

        aresum = 0.0
        sstsum = 0.0

        recn = rec_base + mn - 1
        print recn
        sst = ncsst.variables['tos'][recn,:,:]

        undef_flags = (sst == miss_val_sst)
        sst[undef_flags] = np.NaN
        print sst
        
        for i in range (nx):
            for j in range(ny):
                if (not np.isnan(sst[j,i])):
                    aresum = aresum + area[j,i] * mask[j,i]
                    sstsum = sstsum + area[j,i] * mask[j,i] * sst[j,i]
                    
        sstsum = sstsum / aresum
        sstm_vars[ic] = sstsum
        print yr, mn, ic, '(mean)', sstm_vars[ic]
        ic = ic + 1
        

for i in range(len(td)):
    print time_vars[i], sstm_vars[i]

time[:]=time_vars
tosga[:]=sstm_vars

ncsstm.close()
