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

path_siindx = 'Global_SL_Budget/original'

######

ncsiidx = netCDF4.Dataset(path_siindx + '/GLOBAL_SL_Budget.nc', 'w', format='NETCDF4')
ncsiidx.createDimension('ntime', None)

dtime = pd.date_range('1948-01-01','2019-01-01',freq='AS-JAN')
dtime_start = datetime.date(1948, 1, 1)
time = ncsiidx.createVariable('time', np.dtype('int32').char, ('ntime',))
time.long_name = 'time of annual mean sea level '
time.units = 'days since 1948-01-01 00:00:00'

gmsl = ncsiidx.createVariable('gmsl', np.dtype('float').char, ('ntime'))
gmsl.long_name = 'Global mean sea level'
gmsl.units = 'mm'

steric = ncsiidx.createVariable('steric', np.dtype('float').char, ('ntime'))
steric.long_name = 'Global mean ocean thermal expansion'
steric.units = 'mm'

glacier = ncsiidx.createVariable('glacier', np.dtype('float').char, ('ntime'))
glacier.long_name = 'Glacier contribution'
glacier.units = 'mm'

greenland = ncsiidx.createVariable('greenland', np.dtype('float').char, ('ntime'))
greenland.long_name = 'Greenland Icesheet contribution'
greenland.units = 'mm'

antarctica = ncsiidx.createVariable('antarctica', np.dtype('float').char, ('ntime'))
antarctica.long_name = 'Antarctica Icesheet contribution'
antarctica.units = 'mm'

grace = ncsiidx.createVariable('grace', np.dtype('float').char, ('ntime'))
grace.long_name = 'Grace based ocean mass'
grace.units = 'mm'

####################3

infile = path_siindx + '/GMSL.txt'
print infile
df_tmp = pd.read_csv(infile,delim_whitespace=True,parse_dates=['Time'],index_col=0)
df_tmp.drop(['(mm)'],axis='columns',inplace=True)
#df_tmp = pd.read_csv(infile,sep=' ')
print df_tmp
sealevel_all = df_tmp

infile = path_siindx + '/Steric.txt'
print infile
df_tmp = pd.read_csv(infile,delim_whitespace=True,parse_dates=['Time'],index_col=0)
df_tmp.drop(['thermal', 'expansion', '(mm)'],axis='columns',inplace=True)
df_new = df_tmp.rename(columns={'Mean': 'Steric'})
sealevel_all = pd.concat([sealevel_all,df_new],axis=1)

infile = path_siindx + '/Glaciers.txt'
print infile
df_tmp = pd.read_csv(infile,delim_whitespace=True,parse_dates=['Time'],index_col=0)
df_tmp.drop(['(mm)'],axis='columns',inplace=True)
print df_tmp
sealevel_all = pd.concat([sealevel_all,df_tmp],axis=1)

infile = path_siindx + '/GreenlandIcesheet.txt'
print infile
df_tmp = pd.read_csv(infile,delim_whitespace=True,parse_dates=['Time'],index_col=0)
df_tmp.drop(['(mm)'],axis='columns',inplace=True)
print df_tmp
sealevel_all = pd.concat([sealevel_all,df_tmp],axis=1)

infile = path_siindx + '/AntarcticIceSheet.txt'
print infile
df_tmp = pd.read_csv(infile,delim_whitespace=True,parse_dates=['Time'],index_col=0)
df_tmp.drop(['(mm)'],axis='columns',inplace=True)
print df_tmp
sealevel_all = pd.concat([sealevel_all,df_tmp],axis=1)

infile = path_siindx + '/GRACE_oceanmass.txt'
print infile
df_tmp = pd.read_csv(infile,delim_whitespace=True,parse_dates=['Time'],index_col=0)
print df_tmp
df_tmp.drop(['GRACE', 'based', 'ocean', 'mass', '(mm)'],axis='columns',inplace=True)
df_new = df_tmp.rename(columns={'Mean': 'Grace'})
print df_new
sealevel_all = pd.concat([sealevel_all,df_new],axis=1)

print sealevel_all

td=pd.to_datetime(sealevel_all.index).date - dtime_start
time_vars = np.array(np.zeros((len(td))))
gmsl_vars = np.array(np.zeros((len(td))))
steric_vars = np.array(np.zeros((len(td))))
glacier_vars = np.array(np.zeros((len(td))))
greenland_vars = np.array(np.zeros((len(td))))
antarctica_vars = np.array(np.zeros((len(td))))
grace_vars = np.array(np.zeros((len(td))))

gmsl_vars = sealevel_all['GMSL'].values
steric_vars = sealevel_all['Steric'].values
glacier_vars = sealevel_all['Antarctica'].values
greenland_vars = sealevel_all['Greenland'].values
antarctica_vars = sealevel_all['Antarctica'].values
grace_vars = sealevel_all['Grace'].values

for i in range(len(td)):
    time_vars[i] = td[i].days
    print i, time_vars[i], gmsl_vars[i], steric_vars[i], glacier_vars[i], greenland_vars[i], antarctica_vars[i], grace_vars[i]
  
time[:]=time_vars

gmsl[:]=gmsl_vars
steric[:]=steric_vars
glacier[:]=glacier_vars
greenland[:]=greenland_vars
antarctica[:]=antarctica_vars
grace[:]=grace_vars

ncsiidx.close()
