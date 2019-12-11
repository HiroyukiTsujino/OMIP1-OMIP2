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

if len(sys.argv) < 1:
    print ('Usage: ' + sys.argv[0] + ' start_year end_year')
    sys.exit()

start = int(sys.argv[1])-1
end = int(sys.argv[2])

#--------------------

file = 'OMIP/NCAR-POP/20190731/omip1/uo_Oyr_CESM2_omip1_r2i1p1f1_gn_140W_20S-20N_0311-0372.nc'
#file = 'OMIP/NCAR-POP/20190731/omip2/uo_Oyr_CESM2_omip2_r1i1p1f1_gn_140W_20S-20N_0306-0366.nc'

#--------------------

nc = netCDF4.Dataset(file,'r')
lat_ = nc.variables['lat'][:,:]
depth_ = nc.variables['lev'][:]
dtime = pd.date_range('1948-01-01','2009-01-01',freq='AS-JAN')
jmut = len(lat_)
km   = len(depth_)

#latS = -20.5
#latN = 20.5
latitude_ = lat_[:,0]

print (jmut, km)
print (lat_)
print (latitude_)

xx, yy = np.meshgrid(latitude_,depth_*1.e-2)

miss_val = nc.variables['uo'].missing_value

#-------------------

uo_raw = nc.variables['uo'][:,:,:,:]

uo_raw_tzy = uo_raw[start:end,:,:,0]

print (np.shape(uo_raw_tzy))

np.where(uo_raw_tzy == miss_val, np.NaN, uo_raw_tzy)

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
