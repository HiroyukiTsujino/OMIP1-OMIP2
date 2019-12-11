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

#--------------------

if ( mip_id == 'omip1'):
    file = 'OMIP/MIROC-COCO4-9/20190731/CORE/uo.nc'
else:
    file = 'OMIP/MIROC-COCO4-9/20190731/JRA/uo.nc'

#--------------------

nc = netCDF4.Dataset(file,'r')
lat_ = nc.variables['lat'][:]
depth_ = nc.variables['depth'][:]
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

miss_val = nc.variables['uo'].missing_value

#-------------------

uo_raw = nc.variables['uo'][:,:,:,:]

np.where(uo_raw == miss_val, np.NaN, uo_raw)

uo_raw_tzy = uo_raw[start:end,:,:,0]

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
