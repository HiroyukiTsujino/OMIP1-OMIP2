# -*- coding: utf-8 -*-
import fix_proj_lib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

#--------------------

if len(sys.argv) < 3:
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year end_year')
    sys.exit()

mip_id = sys.argv[1]
start = int(sys.argv[2])-1
end = int(sys.argv[3])

#st_mon = start * 12
#ed_mon = end * 12

#--------------------

if ( mip_id == 'omip1'):
    file = 'OMIP/AWI-FESOM/20190830/OMIP1/loop6/uo_Omon_FESOM1.4_historical_loop6_gr_194801-200912.nc'
else:
    file = 'OMIP/AWI-FESOM/20190830/OMIP2/loop6/uo_Omon_FESOM1.4_historical_loop6_gr_194801-200912.nc'

#--------------------

nc = netCDF4.Dataset(file,'r')
lat_ = nc.variables['lat'][:]
depth_ = nc.variables['depth_coord'][:]
dtime = pd.date_range('1948-01-01','2009-01-01',freq='AS-JAN')
jmut = len(lat_)
km   = len(depth_)

#latS = -20.5
#latN = 20.5
latitude_ = lat_

print (jmut, km)
print (latitude_)
print (depth_)

xx, yy = np.meshgrid(latitude_,depth_)

miss_val = nc.variables['uo']._FillValue

#-------------------

tmp = nc.variables['uo'][:,:,:]
uo_raw = tmp.transpose()

np.where(uo_raw == miss_val, np.NaN, uo_raw)

uo_raw_tzy = uo_raw[start:end,:,:]

print (np.shape(uo_raw_tzy))

uo = np.average(uo_raw_tzy,axis=0)

print (np.shape(uo))
print (np.shape(xx))
print (np.shape(yy))

ct = np.arange(-0.8,1.2,0.1)

fig = plt.figure(figsize = (15,9))

axes = fig.add_subplot(2,2,1)
axes.invert_yaxis()
axes.set_title('zonal velocity at 140W') 
c = axes.contourf(xx,yy,uo,ct)
fig.colorbar(c,ax=axes)

plt.show()

nc.close()
