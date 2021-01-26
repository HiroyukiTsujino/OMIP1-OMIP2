# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4
import datetime
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

#-------------------------------------------

if (len(sys.argv) < 5) :
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year end_year filtered(yes or no)')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])
filtered = sys.argv[4]

#-------------------------------------------

metainfo = json.load(open('./json/zos_' + mip + '.json'))
model_list = metainfo.keys()

#----------------------------------------------

path_amip = '../refdata/PCMDI-SST'
path_cmems = '../analysis/SSH/CMEMS'

if (filtered == 'yes'):
    cmemsclim = path_cmems + '/' + 'zos_filter_annclim_gn_199301-200912.nc'
    cmemsmskf = path_cmems + '/' + 'zos_filter_mask_gn_199301-200912.nc'
else:
    cmemsclim = path_cmems + '/' + 'zos_annclim_gn_199301-200912.nc'
    cmemsmskf = path_cmems + '/' + 'zos_mask_gn_199301-200912.nc'

nccmems = netCDF4.Dataset(cmemsclim,'r')
miss_val_cmems = nccmems.variables['zos'].missing_value
nx = len(nccmems.dimensions['lon'])
ny = len(nccmems.dimensions['lat'])
sshcmems = nccmems.variables['zos'][:,:]
lon_ = nccmems.variables['lon'][:]
lat_ = nccmems.variables['lat'][:]

print (nx,ny, miss_val_cmems)

ncmskcmems = netCDF4.Dataset(cmemsmskf,'r')
maskcmems = ncmskcmems.variables['zosmask'][:,:]
ncmskcmems.close()

################################################
# Ad hoc modification for Mediterranean (mask out entirely)
maskcmems[120:140,0:40] = 0
maskcmems[120:130,355:360] = 0
################################################

arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]
ncare.close()

#----------------------------------------------

path_omip_cl= '../analysis/SSH/MODEL'

for model in metainfo.keys():

    print (' ')
    print ('Processing ', model)

    omipclim = path_omip_cl + '/' + 'zos_annclim_' + model + '_' + mip + '_199301-200912.nc'
    omipmskf = path_omip_cl + '/' + 'zos_mask_' + model + '_' + mip + '_199301-200912.nc'

    ncomip = netCDF4.Dataset(omipclim,'r')
    miss_val_omip = ncomip.variables['zos'].missing_value
    nx = len(ncomip.dimensions['lon'])
    ny = len(ncomip.dimensions['lat'])
    sshomip = ncomip.variables['zos'][:,:]

    print (nx,ny, miss_val_omip)

    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip_tmp = ncmskomip.variables['zosmask'][:,:]
    ncmskomip.close()

    maskomip = maskomip_tmp.astype(np.float64)

    #----------------------------------------------
    maskboth = maskomip * maskcmems
    wgtboth = maskboth * area

    wgtboth_sum = (wgtboth).sum()

    cmemsssh_sum = (sshcmems*wgtboth).sum()
    cmemsssh_mean = cmemsssh_sum/wgtboth_sum

    omipssh_sum = (sshomip*wgtboth).sum()
    omipssh_mean = omipssh_sum/wgtboth_sum
    print('Mean ', cmemsssh_mean, omipssh_mean)

    varcmems_sum = ((sshcmems-cmemsssh_mean)**2 * wgtboth).sum()
    varomip_sum = ((sshomip-omipssh_mean)**2 * wgtboth).sum()

    varcmems = varcmems_sum/wgtboth_sum
    varomip = varomip_sum/wgtboth_sum

    stdcmems = np.sqrt(varcmems)
    stdomip = np.sqrt(varomip)
    print('Standard deviation ', stdcmems, stdomip)

    corr_sum = ((sshcmems-cmemsssh_mean)*(sshomip-omipssh_mean) * wgtboth).sum()
    corr = corr_sum / wgtboth_sum / stdcmems / stdomip
    print('Correlation ', corr)

    dist_sum = (((sshcmems-cmemsssh_mean) - (sshomip-omipssh_mean))**2 * wgtboth).sum()
    dist = dist_sum / wgtboth_sum
    print('Distance (raw)          ', dist)

    dist_tmp = varcmems + varomip - 2.0 * stdcmems * stdomip * corr
    print('Distance (confirmation) ', dist_tmp)
