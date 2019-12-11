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
    print ('Usage: ' + sys.argv[0] + ' start end')
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

xx, yy = np.meshgrid(latitude_,depth_)

miss_val = nc.variables['thetao'].missing_value

#-------------------

theta = nc.variables['thetao'][:,:,:,:]

np.where(theta == miss_val, np.NaN, theta)

ct = np.arange(-2,30,1)
basins = ('Global', 'Altantic' , 'Indian', 'Pacific')

fig = plt.figure(figsize = (15,9))

for i in range(4):

    print (i)

    theta_glb = theta[start:end,i,:,:]

    print (theta_glb.shape)
    theta_glb_mean = np.average(theta_glb,axis=0)
    print (theta_glb_mean.shape)

    axes = fig.add_subplot(2,2,i+1)
    axes.invert_yaxis()
    axes.set_title('temperature '+basins[i]) 
    c = axes.contourf(xx,yy,theta_glb_mean,ct)
    fig.colorbar(c,ax=axes)

plt.show()
nc.close()
