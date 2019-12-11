# -*- coding: utf-8 -*-

import fix_proj_lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

nx = 360
ny = 180
aa = 9.0e20

lonW =  0.5; lonE = 359.5
latS = -89.5; latN = 89.5
lon = np.linspace(lonW,lonE,nx)
lat = np.linspace(latS,latN,ny)

dtime = pd.date_range('1948-01-01','2018-12-01',freq='MS')

path = 'DATA/grads'

mask = np.array(np.zeros((ny,nx)))
mean = np.array(np.zeros((ny,nx)))

ic = 0

for yr in range(1948,1949):

    infile = path + '/' + 'sst-glb.' + str(yr)

    print infile

    f1 = open(infile,'rb')

    for mn in range(1,13):

        ic = ic + 1
        sst = np.array(np.empty((ny,nx)))

        for j in range(ny):
            sst[j,:] = np.fromfile(f1, dtype = '>f', count = nx)

        undef_flags = (sst > aa)
        sst[undef_flags] = np.NaN

        print yr, mn, sst[122,137]
        for i in range (nx):
            for j in range(ny):
                if (not np.isnan(sst[j,i])):
                    mask[j,i] = mask[j,i] + 1.0
                    mean[j,i] = mean[j,i] + sst[j,i]

for i in range (nx):
    for j in range(ny):
          if mask[j,i] > 0.0:
             mean[j,i] = mean[j,i] / mask[j,i]
          else:
             mean[j,i] = np.NaN

mask = mask / ic

print 'mean', ic, mean[122,90], mask[122,90]


#---------------------------------


lon2, lat2 = np.meshgrid(lon,lat)


fig = plt.figure(figsize = (12,15))

ax = fig.add_subplot(212)
ax.set_title("COBE-SST mask")
m = Basemap(projection='cyl',llcrnrlon=lonW,urcrnrlon=lonE,llcrnrlat=latS,urcrnrlat=latN)
m.drawmeridians(np.arange(0, 360, 30), labels=[True, False, False, True])
m.drawparallels(np.arange(-90, 90.1, 10), labels=[True, False, False, False])
x, y = m(lon2,lat2)
ct = np.arange(-0.1,1.1,0.1)
m.drawcoastlines()
m.contourf(x,y,mask,ct)
m.colorbar()

ax = fig.add_subplot(211)
ax.set_title("COBE-SST mean")
m = Basemap(projection='cyl',llcrnrlon=lonW,urcrnrlon=lonE,llcrnrlat=latS,urcrnrlat=latN)
m.drawmeridians(np.arange(0, 360, 30), labels=[True, False, False, True])
m.drawparallels(np.arange(-90, 90.1, 10), labels=[True, False, False, False])
x, y = m(lon2,lat2)
ct = np.arange(-2,31,1)
m.drawcoastlines()
m.contourf(x,y,mean,ct)
m.colorbar()

plt.show()
