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

ny = 180

path_base = './Heat_transport'

lat_bnds_ = np.array(np.empty((ny,2)))
latS = -89.5; latN = 89.5
lat_ = np.linspace(latS,latN,ny)

lat_bnds_[:,0] = lat_[:] - 0.5
lat_bnds_[:,1] = lat_[:] + 0.5

#############
# CORE-run for adjustment

core2g_base = path_base + '/' + 'CORE2g-run-20130808/'

nt=60
core2g_evap  = np.array(np.empty((nt,3,ny)),dtype=np.float32)
core2g_prec  = np.array(np.empty((nt,3,ny)),dtype=np.float32)
core2g_river = np.array(np.empty((nt,3,ny)),dtype=np.float32)
core2g_sice  = np.array(np.empty((nt,3,ny)),dtype=np.float32)

n=0
for nyr in range(1948,2008):
   fcore2g_evap_in=core2g_base + 'merid_ht_evap.{year:04}'.format(year=nyr)
   print('Reading from '+ fcore2g_evap_in)
   f1 = open(fcore2g_evap_in,'rb')
   for nb in range(3):
      core2g_evap[n,nb,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()

   fcore2g_prec_in=core2g_base + 'merid_ht_prec.{year:04}'.format(year=nyr)
   print('Reading from '+ fcore2g_prec_in)
   f1 = open(fcore2g_prec_in,'rb')
   for nb in range(3):
      core2g_prec[n,nb,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()

   fcore2g_river_in=core2g_base + 'merid_ht_river.{year:04}'.format(year=nyr)
   print('Reading from '+ fcore2g_river_in)
   f1 = open(fcore2g_river_in,'rb')
   for nb in range(3):
      core2g_river[n,nb,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()

   fcore2g_sice_in=core2g_base + 'merid_ht_sice.{year:04}'.format(year=nyr)
   print('Reading from '+ fcore2g_sice_in)
   f1 = open(fcore2g_sice_in,'rb')
   for nb in range(3):
      core2g_sice[n,nb,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()

   n += 1
   
core2g_evap_mean = core2g_evap.mean(axis=0)
core2g_prec_mean = core2g_prec.mean(axis=0)
core2g_river_mean = core2g_river.mean(axis=0)
core2g_sice_mean = core2g_sice.mean(axis=0)

#############
# CORE

core_base = path_base + '/' + 'core_cobesst_annual_aug2017/'

nt=62
core_nht  = np.array(np.empty((nt,3,ny)),dtype=np.float32)

n=0
for nyr in range(1948,2010):
   fcore_in=core_base + 'mhtran_ly2009.{year:04}'.format(year=nyr)
   print('Reading from '+ fcore_in)
   f1 = open(fcore_in,'rb')
   for nb in range(3):
      core_nht[n,nb,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()
   n += 1
   
core_nht = np.where(core_nht < -9.0e33, np.NaN, core_nht)
print(core_nht[0,0,:])

#############
# JRA55-do-v1.3

jra_base = path_base + '/' + 'jra55fcst_v1_3_annual_1x1/'

nt=59
jra_nht = np.array(np.empty((nt,3,ny)),dtype=np.float32)
jra_nht_adj = np.array(np.empty((nt,3,ny)),dtype=np.float32)

n=0
for nyr in range(1958,2017):
   fjra_in=jra_base + 'mhtran_noadj.{year:04}'.format(year=nyr)
   print('Reading from '+ fjra_in)
   f1 = open(fjra_in,'rb')
   for nb in range(3):
      jra_nht[n,nb,:] = np.fromfile(f1, dtype = '>f', count = ny)
   f1.close()
   n += 1
   
jra_nht = np.where(jra_nht < -9.0e33, np.NaN, jra_nht)
for n in range(59):
   jra_nht_adj[n,:,:] =jra_nht[n,:,:] - core2g_evap_mean[:,:] + core2g_prec_mean[:,:] + core2g_river_mean[:,:] + core2g_sice_mean[:,:]
print(jra_nht_adj[0,0,:])

#############
# CORE

nht_core_out = core_base + '/' + 'nht_core_ly2009.nc'

nccore = netCDF4.Dataset(nht_core_out, 'w', format='NETCDF4')
nccore.createDimension('lat', ny)
nccore.createDimension('basin', 3)
nccore.createDimension('bnds', 2)
nccore.createDimension('time', None)

nhtcore = nccore.createVariable('nht', np.dtype(np.float32).char, ('time', 'basin', 'lat'), zlib=True)
nhtcore.long_name = 'basin-wide integrated meridional heat transport'
nhtcore.units = 'PW'
nhtcore.missing_value = np.NaN
nhtcore[:,:,:]=core_nht*1.0e-15

#print (txpac.missing_value.dtype)

lat = nccore.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = nccore.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_

basin = nccore.createVariable('basin', np.dtype('float').char, ('basin'))
basin.long_name = 'basin number'
basin.units = '1'
basin.standard_name = 'basin'
basin[:]=[1,2,3]

stdate=str(1948) + '-01-01'
eddate=str(2009) + '-12-31'

dtime = pd.date_range(stdate,eddate,freq='AS-JAN')
dtime_start = datetime.date(1850, 1, 1)
td=pd.to_datetime(dtime[:]).date - dtime_start
time = nccore.createVariable('time', np.dtype('int32').char, ('time',))
time.units = 'days since 1850-01-01 00:00:00'
time.axis = 'T'
time_vars = np.array(np.zeros((len(td))))
for i in range(len(td)):
   time_vars[i] = td[i].days

time[:]=time_vars

nccore.description="Northward heat transport of CORE dataset based on COBESST and LY2009"

nccore.close()

#############
# JRA55-do-v1.3

nht_jra_out = jra_base + '/' + 'nht_jra55do_v1_3.nc'

ncjra = netCDF4.Dataset(nht_jra_out, 'w', format='NETCDF4')
ncjra.createDimension('lat', ny)
ncjra.createDimension('basin', 3)
ncjra.createDimension('bnds', 2)
ncjra.createDimension('time', None)

nhtjra = ncjra.createVariable('nht', np.dtype(np.float32).char, ('time', 'basin', 'lat'), zlib=True)
nhtjra.long_name = 'basin-wide integrated meridional heat transport'
nhtjra.units = 'PW'
nhtjra.missing_value = np.NaN
nhtjra[:,:,:]=jra_nht_adj*1.0e-15

#print (txpac.missing_value.dtype)

lat = ncjra.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncjra.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_

basin = ncjra.createVariable('basin', np.dtype('float').char, ('basin'))
basin.long_name = 'basin number'
basin.units = '1'
basin.standard_name = 'basin'
basin[:]=[1,2,3]

stdate=str(1958) + '-01-01'
eddate=str(2016) + '-12-31'

dtime = pd.date_range(stdate,eddate,freq='AS-JAN')
dtime_start = datetime.date(1850, 1, 1)
td=pd.to_datetime(dtime[:]).date - dtime_start
time = ncjra.createVariable('time', np.dtype('int32').char, ('time',))
time.units = 'days since 1850-01-01 00:00:00'
time.axis = 'T'
time_vars = np.array(np.zeros((len(td))))
for i in range(len(td)):
   time_vars[i] = td[i].days

time[:]=time_vars

ncjra.description="Northward heat transport of JRA55-do-v1.3 dataset based on COBESST"

ncjra.close()
