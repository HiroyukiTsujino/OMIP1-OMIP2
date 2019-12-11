# -*- coding: utf-8 -*-
import fix_proj_lib
import numpy as np
import numpy as np

#from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

nx = 2049
ny = 784
aa = -9.98e33
f1 = open('hs_ssh.19930105','rb')
ssh = np.array(np.empty((ny,nx)))
ssh = np.empty((ny,nx))
lonW =  98.81818; lonE = 285.000
latS = -15.20000; latN = 63.1000
lon = np.linspace(lonW,lonE,nx)
lat = np.linspace(latS,latN,ny)

for j in range(ny):
   ssh[j,:] = np.fromfile(f1, dtype = '>f', count = nx)

undef_flags = ssh < aa
ssh[undef_flags] = np.NaN

#CS = plt.contour(lon,lat,ssh,15)

lon2, lat2 = np.meshgrid(lon,lat)
m = Basemap(projection='cyl',llcrnrlon=lonW,urcrnrlon=lonE,llcrnrlat=latS,urcrnrlat=latN)

m.drawmeridians(np.arange(0, 360, 30), labels=[True, False, False, True])
m.drawparallels(np.arange(-90, 90.1, 10), labels=[True, False, False, False])
x, y = m(lon2,lat2)

ct = np.arange(-60,180,15)

#m.fillcontinents(color='gray',lake_color='gray')
m.drawcoastlines()

m.contourf(x,y,ssh,ct)
m.colorbar()
plt.show()
