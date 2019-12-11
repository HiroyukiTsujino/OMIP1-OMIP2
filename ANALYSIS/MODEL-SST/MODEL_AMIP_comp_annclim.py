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

#--------------------

if (len(sys.argv) < 4) :
    print ('Usage: MODEL_AMIP_comp_annclim.py mip_id start_year end_year')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])

#--------------------

metainfo = json.load(open('./json/tos_' + mip + '.json'))
model_list = metainfo.keys()

#----------------------------------------------
# Load AMIP data

path_amip = '../refdata/PCMDI-SST'
path_amip_cl = '../analysis/SST/PCMDI-SST'

amipclim = path_amip_cl + '/' + 'tos_annclim_gn_198001-200912.nc'
amipmskf= path_amip + '/' + 'sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncamip = netCDF4.Dataset(amipclim,'r')
miss_val_amip = ncamip.variables['tos'].missing_value
nx = len(ncamip.dimensions['lon'])
ny = len(ncamip.dimensions['lat'])
sstamip = ncamip.variables['tos'][:,:]
lon_ = ncamip.variables['lon'][:]
lat_ = ncamip.variables['lat'][:]

print (nx,ny, miss_val_amip)

ncmskamip = netCDF4.Dataset(amipmskf,'r')
maskamip = ncmskamip.variables['sftof'][:,:]
ncmskamip.close()

# Ad hoc modification of the Kaspian Sea
for j in range(ny):
    for i in range(nx):
        if (45 < lon_[i]) & (lon_[i] < 60):
             if (34 < lat_[j]) & (lat_[j] < 50):
                  maskamip[j,i] = 0

        if maskamip[j,i] < 100:
            maskamip[j,i] = 0

maskamip = maskamip / 100


arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]
ncare.close()

#----------------------------------------------

path_omip_cl= '../analysis/SST/MODEL'

for model in metainfo.keys():

    print (' ')
    print ('Processing ', model)

    omipclim = path_omip_cl + '/' + 'tos_annclim_' + model + '_' + mip + '_198001-200912.nc'
    omipmskf = path_omip_cl + '/' + 'tos_mask_' + model + '_' + mip + '_198001-200912.nc'

    ncomip = netCDF4.Dataset(omipclim,'r')
    miss_val_omip = ncomip.variables['tos'].missing_value
    nx = len(ncomip.dimensions['lon'])
    ny = len(ncomip.dimensions['lat'])
    sstomip = ncomip.variables['tos'][:,:]

    print (nx,ny, miss_val_omip)

    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip_tmp = ncmskomip.variables['tosmask'][:,:]
    ncmskomip.close()

    maskomip = maskomip_tmp.astype(np.float64)

    #----------------------------------------------

    maskboth = maskomip * maskamip
    wgtboth = maskboth * area

    wgtboth_sum = (wgtboth).sum()

    amipsst_sum = (sstamip*wgtboth).sum()
    amipsst_mean=amipsst_sum/wgtboth_sum

    omipsst_sum = (sstomip*wgtboth).sum()
    omipsst_mean=omipsst_sum/wgtboth_sum
    print('Mean ', amipsst_mean, omipsst_mean)

    varamip_sum = ((sstamip-amipsst_mean)**2 * wgtboth).sum()
    varomip_sum = ((sstomip-omipsst_mean)**2 * wgtboth).sum()

    varamip = varamip_sum/wgtboth_sum
    varomip = varomip_sum/wgtboth_sum

    stdamip = np.sqrt(varamip)
    stdomip = np.sqrt(varomip)
    print('Standard deviation ', stdamip, stdomip)

    corr_sum = ((sstamip-amipsst_mean)*(sstomip-omipsst_mean) * wgtboth).sum()
    corr = corr_sum / wgtboth_sum / stdamip / stdomip
    print('Correlation ', corr)

    dist_sum = (((sstamip-amipsst_mean) - (sstomip-omipsst_mean))**2 * wgtboth).sum()
    dist = dist_sum / wgtboth_sum
    print('Distance (raw)          ', dist)

    dist_tmp = varamip + varomip - 2.0 * stdamip * stdomip * corr
    print('Distance (confirmation) ', dist_tmp)
