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

if len(sys.argv) == 1:
    print ('Usage: read_amip_sst.py start end')
    sys.exit()

start = int(sys.argv[1])-1
end = int(sys.argv[2])

#--------------------

file = 'OMIP/FSU-COAPS/20190731/ts_annual_yzt_6thcycle_core2_4basins_fsu.nc'

#--------------------

nc = netCDF4.Dataset(file,'r')
lat_ = nc.variables['latitude'][:]
depth_ = nc.variables['depth'][:]
dtime = pd.date_range('1948-01-01','2009-01-01',freq='AS-JAN')
jmut = len(lat_)
km   = len(depth_)

latS = -89.5; latN = 89.5
latitude_ = np.linspace(latS,latN,jmut)

print (jmut, km)

xx, yy = np.meshgrid(latitude_,depth_)

miss_val = nc.variables['thetao'].missing_value

#-------------------

theta2d = nc.variables['thetao'][1,start,:,:]

np.where(theta2d == miss_val, np.NaN, theta2d)

ct = np.arange(0,30,5)

fig = plt.figure(figsize = (15,9))

axes = fig.add_subplot(2,2,1)
axes.invert_yaxis()
axes.set_title('temperature') 
axes.contourf(xx,yy,theta2d)

plt.show()

nc.close()
