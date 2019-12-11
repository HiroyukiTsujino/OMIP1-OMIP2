# -*- coding: utf-8 -*-
import fix_proj_lib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

#--------------------

if len(sys.argv) == 1:
    print 'Usage: msftmyz_basin_all_yz.py start end'
    sys.exit()

start = int(sys.argv[1])-1
end = int(sys.argv[2])

#--------------------

#file = 'OMIP/FSU-COAPS/20190724/msftmyz_annual_yzt_allcycle_core2_fsu.nc'
file = 'OMIP/FSU-COAPS/20190724/msftmyz_annual_yzt_allcycle_jra55_fsu.nc'

#--------------------

nc = netCDF4.Dataset(file,'r')
lat_ = nc.variables['latitude'][:]
depth_ = nc.variables['depth'][:]
#dtime = pd.date_range('1658-01-01','2009-01-01',freq='AS-JAN')
jmut = len(lat_)
km   = len(depth_)

latS = -89.5; latN = 89.5
latitude_ = np.linspace(latS,latN,jmut)

xx, yy = np.meshgrid(latitude_,depth_)

miss_val = nc.variables['msftmyz'].missing_value

#-------------------

moc = nc.variables['msftmyz'][:,:,:,:]
print moc.shape

np.where(moc == miss_val, np.NaN, moc)

basins = ('Global', 'Altantic', 'Indo-Pacific')

fig = plt.figure(figsize = (15,9))

for i in range(3):

    print i

    moc_glb = moc[start:end,i,:,:]

    print moc_glb.shape
    moc_glb_mean = np.average(moc_glb,axis=0)
    print moc_glb_mean.shape

    axes = fig.add_subplot(2,2,i+1)
    axes.invert_yaxis()
    axes.set_title('MOC '+basins[i]) 
    ct = np.arange(-50,50,1)
    c = axes.contourf(xx,yy,moc_glb_mean,ct)
    fig.colorbar(c,ax=axes)

    ct = np.arange(-30,30,5)
    cm = axes.contour(xx,yy,moc_glb_mean,ct,colors='black')
    axes.clabel(cm, fmt='%2i', fontsize=10)

plt.show()
nc.close()
