# -*- coding: utf-8 -*-
#import fix_proj_lib
import json
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

ny = 720
nt = 12

path_grads = './grads'
path_netcdf = './netcdf'

lat_bnds_ = np.array(np.empty((ny,2)))
latS = -89.875; latN = 89.875
lat_ = np.linspace(latS,latN,ny)

lat_bnds_[:,0] = lat_[:] - 0.125
lat_bnds_[:,1] = lat_[:] + 0.125

dtime = pd.date_range('2013-01-01','2013-12-01',freq='MS')
dtime_start = datetime.date(2013, 1, 1)
td=pd.to_datetime(dtime[:]).date - dtime_start
time_vars = np.array(np.zeros((len(td))))
for i in range(len(td)):
   time_vars[i] = td[i].days

#############
# Atlantic

ftx_atl_in_base = path_grads + '/' + 'taux_atl_zm.2013'
ftx_atl_out = path_netcdf + '/' + 'taux_atl_zm.nc'
taux_atl = np.array(np.empty((nt,ny)),dtype=np.float32)

for n in range(nt):
   ftx_atl_in=ftx_atl_in_base + '{mon:02}'.format(mon=n+1)
   print('Reading from '+ ftx_atl_in)
   f1 = open(ftx_atl_in,'rb')
   taux_atl[n,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()

#print(taux_atl[0,:])

nctxa = netCDF4.Dataset(ftx_atl_out, 'w', format='NETCDF4')
nctxa.createDimension('lat', ny)
nctxa.createDimension('bnds', 2)
nctxa.createDimension('time', None)

txatl = nctxa.createVariable('tauuo_atl', np.dtype(np.float32).char, ('time', 'lat'), zlib=True)
txatl.long_name = 'basin-wide averaged zonal wind stress of the Atlantic Ocean'
txatl.units = 'N m-2'
txatl.missing_value = -9.99e33
txatl[:,:]=taux_atl

#print (txatl.missing_value.dtype)

lat = nctxa.createVariable('lat', np.dtype('float').char, ('lat'))
lat.latg_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = nctxa.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_

time = nctxa.createVariable('time', np.dtype('int32').char, ('time',))
time.units = 'days since 2013-01-01 00:00:00'
time.axis = 'T'
time[:]=time_vars

nctxa.description="Monthly climatology Nov 1999 through Oct 2009"

nctxa.close()

#############
# Pacific

ftx_pac_in_base = path_grads + '/' + 'taux_pac_zm.2013'
ftx_pac_out = path_netcdf + '/' + 'taux_pac_zm.nc'
taux_pac = np.array(np.empty((nt,ny)),dtype=np.float32)

for n in range(nt):
   ftx_pac_in=ftx_pac_in_base + '{mon:02}'.format(mon=n+1)
   print('Reading from '+ ftx_pac_in)
   f1 = open(ftx_pac_in,'rb')
   taux_pac[n,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()

#print(taux_pac[0,:])

nctxp = netCDF4.Dataset(ftx_pac_out, 'w', format='NETCDF4')
nctxp.createDimension('lat', ny)
nctxp.createDimension('bnds', 2)
nctxp.createDimension('time', None)

txpac = nctxp.createVariable('tauuo_pac', np.dtype(np.float32).char, ('time', 'lat'), zlib=True)
txpac.long_name = 'basin-wide averaged zonal wind stress of the Pacific Ocean'
txpac.units = 'N m-2'
txpac.missing_value = -9.99e33
txpac[:,:]=taux_pac

#print (txpac.missing_value.dtype)

lat = nctxp.createVariable('lat', np.dtype('float').char, ('lat'))
lat.latg_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = nctxp.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_

time = nctxp.createVariable('time', np.dtype('int32').char, ('time',))
time.units = 'days since 2013-01-01 00:00:00'
time.axis = 'T'
time[:]=time_vars

nctxp.description="Monthly climatology Nov 1999 through Oct 2009"

nctxp.close()

#############
# Global

ftx_glb_in_base = path_grads + '/' + 'taux_glb_zm.2013'
ftx_glb_out = path_netcdf + '/' + 'taux_glb_zm.nc'
taux_glb = np.array(np.empty((nt,ny)),dtype=np.float32)

for n in range(nt):
   ftx_glb_in=ftx_glb_in_base + '{mon:02}'.format(mon=n+1)
   print('Reading from '+ ftx_glb_in)
   f1 = open(ftx_glb_in,'rb')
   taux_glb[n,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()

#print(taux_glb[0,:])

nctxg = netCDF4.Dataset(ftx_glb_out, 'w', format='NETCDF4')
nctxg.createDimension('lat', ny)
nctxg.createDimension('bnds', 2)
nctxg.createDimension('time', None)

txglb = nctxg.createVariable('tauuo_glb', np.dtype(np.float32).char, ('time', 'lat'), zlib=True)
txglb.long_name = 'basin-wide averaged zonal wind stress of the Global Ocean'
txglb.units = 'N m-2'
txglb.missing_value = -9.99e33
txglb[:,:]=taux_glb

#print (txglb.missing_value.dtype)

lat = nctxg.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = nctxg.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_

time = nctxg.createVariable('time', np.dtype('int32').char, ('time',))
time.units = 'days since 2013-01-01 00:00:00'
time.axis = 'T'
time[:]=time_vars

nctxg.description="Monthly climatology Nov 1999 through Oct 2009"

nctxg.close()
