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
aa = 9.0e20

path_cobe = 'DATA/grads'
path_amip = 'AMIP'

arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
mskfile = path_amip + '/' + 'sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

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

ncsstm = netCDF4.Dataset('COBESST_glbm_monthly_194801-201812.nc', 'w', format='NETCDF4')
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

for yr in range(1948,2019):

    infile = path_cobe + '/' + 'sst-glb.' + str(yr)

    print infile

    f1 = open(infile,'rb')

    for mn in range(1,13):

        aresum = 0.0
        sstsum = 0.0

        sst = np.array(np.empty((ny,nx)))

        for j in range(ny):
            sst[j,:] = np.fromfile(f1, dtype = '>f', count = nx)

        undef_flags = sst > aa
        sst[undef_flags] = np.NaN

        for i in range (nx):
            for j in range(ny):
                if (not np.isnan(sst[j,i])):
                    aresum = aresum + area[j,i] * mask[j,i]
                    sstsum = sstsum + area[j,i] * mask[j,i] * sst[j,i]

        sstsum = sstsum / aresum
        sstm_vars[ic] = sstsum
        print yr, mn, ic, '(mean)', sstm_vars[ic]
        ic = ic + 1
        
    f1.close()


for i in range(len(td)):
    print time_vars[i], sstm_vars[i]

time[:]=time_vars
tosga[:]=sstm_vars

ncsstm.close()
