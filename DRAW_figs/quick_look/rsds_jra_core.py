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

if (len(sys.argv) < 4):
    print ('Usage: ' + sys.argv[0] + ' year month day ')
    sys.exit()

year = int(sys.argv[1])
month = int(sys.argv[2])
day = int(sys.argv[3])

print(str(year),str(month),str(day))
#=================================

def check_leap_year(year):
    if year % 400 == 0:
        return True
    elif year % 4 == 0 and year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

def day_of_year(year,mon,day):
    doy = 0
    mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
    if check_leap_year(year):
        mon_days[1] = mon_day[1]+1
    for m in range(mon-1):
        doy = doy + mon_days[m]
    doy = doy + day
    return doy

def day_of_year_noleap(year,mon,day):
    doy = 0
    mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
    for m in range(mon-1):
        doy = doy + mon_days[m]
    doy = doy + day
    return doy

#=================================

#metainfo = json.load(open("./json/tos_" + mip + ".json"))
#model_list = metainfo.keys()

#----------------------------------------------
# CORE

core_path='/denkei-shared/og/ocpublic/refdata/CORE/COREv2/ciaf/orgdata'
corerfile = core_path + '/' + 'ncar_rad.2007.05APR2010.nc'

nccorer = netCDF4.Dataset(corerfile,'r')

nxc = len(nccorer.dimensions['LON'])
nyc = len(nccorer.dimensions['LAT'])

lonc = nccorer.variables['LON'][:]
latc = nccorer.variables['LAT'][:]

dayofyr = day_of_year_noleap(year,month,day)

recn = dayofyr - 1

swcore = nccorer.variables['SWDN_MOD'][recn,:,:]

#---------
# JRA

jra_rpath='/denkei-shared/og1/htsujino/SURF_FLUX/forcing/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/atmos/3hr/rsds/gr/v20190308'

jrarfile = jra_rpath + '/' + 'rsds_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_200701010130-200712312230.nc'

ncjrar = netCDF4.Dataset(jrarfile,'r')

nxj = len(ncjrar.dimensions['lon'])
nyj = len(ncjrar.dimensions['lat'])

lonj = ncjrar.variables['lon'][:]
latj = ncjrar.variables['lat'][:]

dayofyr = day_of_year(year,month,day)

recn = (dayofyr - 1) * 8

swjra = ncjrar.variables['rsds'][recn,:,:]

#--------------------
# Draw Figures
    
suptitle = 'Shortwave radiation ' + str(year) + '-' + str(month) + '-' + str(day) + ' 01:30Z'
title = [ 'OMIP-1 (CORE)' , 'OMIP-2 (JRA55-do)']
outfile = 'fig/SW-' + str(year) + '-' + str(month) + '-' + str(day) + '.png'

ct = np.arange(0,1000,20)

fig = plt.figure(figsize=(15,15))
fig.suptitle( suptitle, fontsize=20, y=0.95 )

#proj = ccrs.PlateCarree(central_longitude=-140.)
proj = ccrs.Robinson(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(2,1,1,projection=proj),
    plt.subplot(2,1,2,projection=proj),
]

#ax = [ plt.axes([0.1,0.55,0.8,0.4]),
#       plt.axes([0.1,0.05,0.8,0.4]),]

# [left, bottom, width, height]

#ax_cbar = plt.axes([0.15,0.05,0.7,0.1])

print(ct)

for panel in range(2):
    if (panel == 0):
        tmp=swcore
        lon_tmp=np.array(lonc)
        lat_tmp=np.array(latc)
    else:
        tmp=swjra
        lon_tmp=np.array(lonj)
        lat_tmp=np.array(latj)

    tmp, lon_tmp = add_cyclic_point(tmp, coord=lon_tmp)
    ca = ax[panel].contourf(lon_tmp, lat_tmp, tmp, ct, cmap='RdYlBu_r', extend='max',transform=ccrs.PlateCarree())
    ax[panel].coastlines()
    ax[panel].gridlines(linestyle='-',color='gray')
    #ax[panel].set_xlim([100,280])
    #ax[panel].set_ylim([-10,70])
    #ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    #ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    #ax[panel].xaxis.set_major_formatter(lon_formatter)
    #ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_title(title[panel],fontsize=18)
    fig.colorbar(ca,ax=ax[panel],orientation='vertical',shrink=0.7,label='[$\mathrm{W}\,\mathrm{m}^{-2}$]')

#fig.colorbar(ca,ax=ax_cbar,orientation='horizontal',shrink=0.7,label='[m]')

plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
#fig.tight_layout()
plt.show()
