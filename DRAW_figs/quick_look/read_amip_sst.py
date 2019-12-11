# -*- coding: utf-8 -*-
import fix_proj_lib
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

#--------------------

if len(sys.argv) == 1:
    print 'Usage: read_amip_sst.py start end'
    sys.exit()

start = int(sys.argv[1])-1
end = int(sys.argv[2])

#--------------------

file = 'AMIP/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
file_mask = 'AMIP/sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

#--------------------

nc = netCDF4.Dataset(file,'r')
ncmsk = netCDF4.Dataset(file_mask,'r')
lon_ = nc.variables['lon'][:]
lat_ = nc.variables['lat'][:]
time_var = nc.variables['time'][:]
time_units = nc.variables['time'].units
dtime = num2date(time_var[:],time_units)
imut = len(lon_)
jmut = len(lat_)
numdata = len(time_var)

print imut, jmut, numdata

mask = ncmsk.variables['sftof'][:,:]
undef_flags = mask <= 0.0

print undef_flags[60,40]

lon, lat = np.meshgrid(lon_,lat_)

fig = plt.figure(figsize = (15,9))

def init_sst():

    plt.cla()

def draw_sst(i, fig_title):

    if i != 0:
        plt.cla()

    ct = np.arange(-2,32,1)
    tos = nc.variables['tos'][i,:,:]
    tos[undef_flags] = np.NaN
    print i, dtime[i]
    m = Basemap(projection='cyl',llcrnrlon=0,urcrnrlon=360,llcrnrlat=-90,urcrnrlat=90)
    m.drawmeridians(np.arange(0, 360, 30), labels=[True, False, False, True])
    m.drawparallels(np.arange(-90, 90, 10), labels=[True, False, False, False])
    x, y = m(lon,lat)
    m.contourf(x,y,tos,ct,cmap='rainbow')
    m.colorbar()
    ct = np.arange(0,30,5)
    cs = m.contour(x,y,tos,ct,colors=['black'],linewidths=0.5)
    plt.clabel(cs, fmt='%2i', fontsize=12)
    m.fillcontinents(color='gray',lake_color='gray')
    m.drawcoastlines()
    plt.title(fig_title + str(dtime[i])) 

ani = animation.FuncAnimation(fig, draw_sst, fargs = ('AMIP SST ',), interval = 100, frames = np.arange(start,end,1), repeat = True, init_func = init_sst)

plt.show()

nc.close()
