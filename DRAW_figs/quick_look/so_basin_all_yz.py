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

xx, yy = np.meshgrid(latitude_,depth_)

miss_val = nc.variables['so'].missing_value

#-------------------

so = nc.variables['so'][:,:,:,:]
item = 'salinity'

np.where(so == miss_val, np.NaN, so)

ct = np.arange(32,38,0.2)
basins = ('Global', 'Altantic' , 'Indian', 'Pacific')

fig = plt.figure(figsize = (15,9))

for i in range(4):

    print (i)

    so_glb = so[start:end,i,:,:]

    print (so_glb.shape)
    so_glb_mean = np.average(so_glb,axis=0)
    print (so_glb_mean.shape)

    axes = fig.add_subplot(2,2,i+1)
    axes.invert_yaxis()
    axes.set_title(item+' '+basins[i]) 
    c = axes.contourf(xx,yy,so_glb_mean,ct)
    fig.colorbar(c,ax=axes)

plt.show()
nc.close()
