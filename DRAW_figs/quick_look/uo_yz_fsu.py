# -*- coding: utf-8 -*-
#
import sys
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset

#--------------------

if len(sys.argv) < 3:
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year end_year')
    sys.exit()

mip_id = sys.argv[1]
start = int(sys.argv[2])-1
end = int(sys.argv[3])

#--------------------

if ( mip_id == 'omip1'):
    file = 'OMIP/FSU-COAPS/20190823/uo_annual_yzt_6thcycle_core2_fsu.nc'
else:
    file = 'OMIP/FSU-COAPS/20190823/uo_annual_yzt_6thcycle_jra55_fsu.nc'

outfile = 'uo_140W_FSU-' + mip_id + '.png'

#--------------------

nc = netCDF4.Dataset(file,'r')
lat_ = nc.variables['latitude'][:]
depth_ = nc.variables['depth'][:]
jmut = len(lat_)
km = len(depth_)

print (jmut, km)
print (lat_, depth_)

xx, yy = np.meshgrid(lat_,depth_)

miss_val = nc.variables['uo'].missing_value

uo_raw = nc.variables['uo'][:,:,:]

nc.close()

np.where(uo_raw == miss_val, np.NaN, uo_raw)

uo_raw_30 = uo_raw[start:end,:,:]
# mean
uo = np.average(uo_raw_30,axis=0) 
# snapshot
#uo = uo_raw[0,:,:]

ct = np.arange(-1.2,1.2,0.1)

fig = plt.figure(figsize = (15,9))

axes = fig.add_subplot(1,1,1)
axes.invert_yaxis()
axes.set_title('zonal velocity at 140W') 
c = axes.contourf(xx,yy,uo,ct)
fig.colorbar(c,ax=axes)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()

