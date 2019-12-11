# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

nx = 360
ny = 180
miss = -9999.0

path_siindx = 'NSIDC_SII/20190801'

######

ncsiidx = netCDF4.Dataset(path_siindx + '/nsidc_si_index_monthly_NH_SH.nc', 'w', format='NETCDF4')
ncsiidx.createDimension('ntime', None)

dtime = pd.date_range('1948-01-01','2019-12-01',freq='MS')
dtime_start = datetime.date(1948, 1, 1)
time = ncsiidx.createVariable('time', np.dtype('int32').char, ('ntime',))
time.long_name = 'time of monthly sea ice extent and area '
time.units = 'days since 1948-01-01 00:00:00'

siextentn = ncsiidx.createVariable('siextentn', np.dtype('float').char, ('ntime'))
siextentn.long_name = 'Northern hemisphere sea ice extent'
siextentn.units = 'km^2'

siextents = ncsiidx.createVariable('siextents', np.dtype('float').char, ('ntime'))
siextents.long_name = 'Southern hemisphere sea ice extent'
siextents.units = 'km^2'

siarean = ncsiidx.createVariable('siarean', np.dtype('float').char, ('ntime'))
siarean.long_name = 'Northern hemisphere sea ice area'
siarean.units = 'km^2'

siareas = ncsiidx.createVariable('siareas', np.dtype('float').char, ('ntime'))
siareas.long_name = 'Southern hemisphere sea ice area'
siareas.units = 'km^2'

for mn in range(1,13):

    infile = path_siindx + '/' + 'N_{:0=2}'.format(mn) + '_extent_v3.0.csv'

    print infile

    siextn_tmp = pd.read_csv(infile,skipinitialspace=True,parse_dates=[['year','mo']],index_col=0)

    if (mn == 1):
        siextn_all = siextn_tmp
    else:
        siextn_all = pd.concat([siextn_all,siextn_tmp])


    infile = path_siindx + '/' + 'S_{:0=2}'.format(mn) + '_extent_v3.0.csv'

    print infile

    siexts_tmp = pd.read_csv(infile,skipinitialspace=True,parse_dates=[['year','mo']],index_col=0)

    if (mn == 1):
        siexts_all = siexts_tmp
    else:
        siexts_all = pd.concat([siexts_all,siexts_tmp])


siextn_sorted = siextn_all.sort_index()
siexts_sorted = siexts_all.sort_index()

print siextn_sorted.index
print siexts_sorted

#dtime = siextn_sorted.index

td=pd.to_datetime(siextn_sorted.index).date - dtime_start
time_vars = np.array(np.zeros((len(td))))
siextn_vars = np.array(np.zeros((len(td))))
siexts_vars = np.array(np.zeros((len(td))))
siaren_vars = np.array(np.zeros((len(td))))
siares_vars = np.array(np.zeros((len(td))))

siextn_vars = siextn_sorted['extent'].values
siexts_vars = siexts_sorted['extent'].values
siaren_vars = siextn_sorted['area'].values
siares_vars = siexts_sorted['area'].values

siextn_vars = np.where(siextn_vars==miss,np.NaN,siextn_vars)
siexts_vars = np.where(siexts_vars==miss,np.NaN,siexts_vars)
siaren_vars = np.where(siaren_vars==miss,np.NaN,siaren_vars)
siares_vars = np.where(siares_vars==miss,np.NaN,siares_vars)

for i in range(len(td)):
    time_vars[i] = td[i].days
    print i, time_vars[i], siextn_vars[i], siexts_vars[i], siaren_vars[i], siares_vars[i]
  
time[:]=time_vars
siextentn[:]=siextn_vars
siextents[:]=siexts_vars
siarean[:]=siaren_vars
siareas[:]=siares_vars

ncsiidx.close()
