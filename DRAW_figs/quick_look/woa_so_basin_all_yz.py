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

#if len(sys.argv) == 1:
#    print 'Usage: read_amip_sst.py start end'
#    sys.exit()

#--------------------

file = 'WOA13v2/annual/woa13_decav_s_basin.1000'

#--------------------

nc = netCDF4.Dataset(file,'r')
lat_ = nc.variables['lat'][:]
depth_ = nc.variables['depth'][:]
basin_ = nc.variables['basin'][:][:]
basin_name = netCDF4.chartostring(basin_)
jmut = len(lat_)
km   = len(depth_)
num_basin = len(basin_name)

xx, yy = np.meshgrid(lat_,depth_)

miss_val = nc.variables['so'].missing_value

#-------------------

so = nc.variables['so'][:,:,:,:]

np.where(so == miss_val, np.NaN, so)

ct = np.arange(32,38,0.2)

fig = plt.figure(figsize = (15,9))

for i in range(num_basin):

    so_yz = so[0,i,:,:]

    print (so_yz.shape)

    axes = fig.add_subplot(2,2,i+1)
    axes.invert_yaxis()
    axes.set_title('salinity '+basin_name[i].strip()) 
    c = axes.contourf(xx,yy,so_yz,ct)
    fig.colorbar(c,ax=axes)

plt.show()
nc.close()
