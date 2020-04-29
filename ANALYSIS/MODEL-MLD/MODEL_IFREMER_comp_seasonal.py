# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import netCDF4
import datetime
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

#--------------------

if (len(sys.argv) < 4):
    print ('Usage: '+ sys.argv[0] + ' mip start_year end_year season (winter or summer)')
    sys.exit()

mip = sys.argv[1]
stclyr = int(sys.argv[2])
edclyr = int(sys.argv[3])
season = sys.argv[4]
#exlab = int(sys.argv[5])

if (season == 'winter'):
    seaabbr='win'
elif (season == 'summer'):
    seaabbr='sum'

if (mip == 'omip1'):
    period='1948-2009'
    styr=1948
    edyr=2009
    mid = 1
elif (mip == 'omip2'):
    period='1958-2018'
    styr=1958
    edyr=2018
    mid = 2

nyr = edyr - styr + 1
nyrcl = edclyr - stclyr + 1

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]

#----------------------------------------------

#metainfo = json.load(open("./json/mlotst_" + mip + "_dummy.json"))
metainfo = json.load(open("./json/mlotst_" + mip + ".json"))
model_list = metainfo.keys()

#----------------------------------------------

path_obs = '../analysis/MLD/MLD_deBoyer_Montegut'
fobsann = path_obs + '/' + 'mld_DR003_'+str(seaabbr)+'clim.nc'

ncobsann = netCDF4.Dataset(fobsann,'r')
nx = len(ncobsann.dimensions['lon'])
ny = len(ncobsann.dimensions['lat'])
mldobs_ann = ncobsann.variables['mlotst'][:,:]
miss_val_obsann = ncobsann.variables['mlotst'].missing_value
lon_ = ncobsann.variables['lon'][:]
lat_ = ncobsann.variables['lat'][:]

print (nx, ny, miss_val_obsann)

# mask

fobsmskf= path_obs + '/' + 'mld_DR003_'+str(seaabbr)+'mask.nc'
ncmskobs = netCDF4.Dataset(fobsmskf,'r')
maskobs = ncmskobs.variables['mldmask'][:,:]
ncmskobs.close()

# ad hoc exclusion of the Weddell Sea

#for j in range(0,ny):
#    for i in range (0,nx):
####        if (lat_[j] < -60.0 and lon_[i] > 300.0):
#        if (lat_[j] < -60.0):
#            maskobs[j,i] = 0.0
#
#if (exlab == 1):
#    for j in range(0,ny):
#        for i in range (0,nx):
#            if (lat_[j] > 45.0 and lat_[j] < 80.0):
#                if (lon_[i] > 280.0 or lon_[i] < 30.0):
#                    maskobs[j,i] = 0.0
#    
#----------------------------------------------
# cell area

path_amip = '../refdata/PCMDI-SST'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nxm = len(ncare.dimensions['lon'])
nym = len(ncare.dimensions['lat'])
aream = ncare.variables['areacello'][:,:]
ncare.close()
        
#----------------------------------------------

path_omip_cl= '../analysis/MLD/MODEL'
path_out= '../analysis/MLD/MODEL'

nmodel=0
d = np.empty( (len(model_list),ny,nx) )

for model in metainfo.keys():

    print (' ')
    print ('Processing ', model)

    fomipann = path_omip_cl + '/' + 'mlotst_' + str(season) + '_' + model + '_' + mip + '_' + str(period) + '.nc'
    print(fomipann)
    ncomipann = netCDF4.Dataset(fomipann,'r')
    mldomip_ann = ncomipann.variables['mlotst'][:,:,:]
    miss_val_omipann = ncomipann.variables['mlotst'].missing_value
    mldomip_ann = np.where(np.isnan(mldomip_ann),0.0,mldomip_ann)

    omipmskf = path_omip_cl + '/' + 'mlotst_mask_season_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'
    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip = ncmskomip.variables['mldmask'][:,:]
    ncmskomip.close()

    areamask_tmp = maskomip * aream
    areamask = areamask_tmp.astype(np.float64)

    amsk_all = np.array(np.empty((ny,nx)),dtype=np.float64)
    mask_all = np.array(np.empty((ny,nx)),dtype=np.float64)
    mldannm= np.array(np.empty((nyrcl,ny,nx)),dtype=np.float64)

    amsk_all[ny-1,0] = areamask[2*ny-1,0] + areamask[2*ny-1,nxm-1]
    for j in range(0,ny-1):
        amsk_all[j,0] = areamask[2*j+1,0] + areamask[2*j+1,nxm-1] \
                      + areamask[2*j+2,0] + areamask[2*j+2,nxm-1]
        for i in range(1,nx):
            amsk_all[j,i] = areamask[2*j+1,2*i-1] + areamask[2*j+2,2*i-1] \
                          + areamask[2*j+1,2*i] + areamask[2*j+2,2*i]
    for i in range(1,nx):
        amsk_all[ny-1,i] = areamask[2*ny-1,2*i-1] + areamask[2*ny-1,2*i]

    mask_all = np.where(amsk_all > 0.0, 1.0, 0.0)

    nn = 0
    nm = 0
    for nyear in range(styr,edyr+1):
        if (nyear >= stclyr and nyear <= edclyr):
            mldannm[nm,ny-1,0] = areamask[2*ny-1,0] * mldomip_ann[nn,2*ny-1,0] \
                               + areamask[2*ny-1,nxm-1] * mldomip_ann[nn,2*ny-1,nxm-1]

            for j in range(0,ny-1):
                mldannm[nm,j,0] = areamask[2*j+1,0]     * mldomip_ann[nn,2*j+1,0] \
                                + areamask[2*j+1,nxm-1] * mldomip_ann[nn,2*j+1,nxm-1] \
                                + areamask[2*j+2,0]     * mldomip_ann[nn,2*j+2,0] \
                                + areamask[2*j+2,nxm-1] * mldomip_ann[nn,2*j+2,nxm-1] 
                for i in range(1,nx):
                    mldannm[nm,j,i] = areamask[2*j+1,2*i-1] * mldomip_ann[nn,2*j+1,2*i-1] \
                                    + areamask[2*j+2,2*i-1] * mldomip_ann[nn,2*j+2,2*i-1] \
                                    + areamask[2*j+1,2*i] * mldomip_ann[nn,2*j+1,2*i] \
                                    + areamask[2*j+2,2*i] * mldomip_ann[nn,2*j+2,2*i] 

            for i in range(1,nx):
                mldannm[nm,ny-1,i] = areamask[2*ny-1,2*i-1] * mldomip_ann[nn,2*ny-1,2*i-1] \
                                   + areamask[2*ny-1,2*i] * mldomip_ann[nn,2*ny-1,2*i]
            nm += 1
        nn += 1

    mldannm[0:nyrcl,:,:] = mldannm[0:nyrcl,:,:] / (1.0 - mask_all[:,:] + amsk_all[:,:]) 
    mldannm[0:nyrcl,:,:] = np.where(mask_all[:,:] == 0, np.NaN, mldannm[0:nyrcl,:,:])
        
    d[nmodel] = mldannm.mean(axis=0)
    nmodel += 1
    
    del amsk_all
    del mask_all
    del mldannm

DS = xr.Dataset( {'mld': (['model','lat','lon'], d),
                  'mldbias': (['model','lat','lon'], d - mldobs_ann),
                  'obs': (['lat','lon'], mldobs_ann),},
                 coords = { 'lat': lat_, 'lon': lon_, } )

#--------------------
# Draw Figures

suptitle = str(season) + ' MLD Climatology ' + str(mip) + ' ' + str(stclyr) + ' to ' + str(edclyr)
outfile = 'fig/' + mip + '-MLD_' + str(season) + '_climatology_np.png'

fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(3,1,1,projection=proj),
    plt.subplot(3,1,2,projection=proj),
    plt.subplot(3,1,3,projection=proj),
]

title = [ 'MODEL' , 'OBS', 'BIAS' ]
cmap = [ 'RdYlBu_r','RdYlBu_r', 'RdBu_r' ]

if (season == 'winter'):
    co = np.array([0,10,20,50,100,200,300,400,500,600,1000,1500,2000,2500,3000,4000,5000])
    cd = np.arange(-200,200,20)
elif (season == 'summer'):
    co = np.arange(0,150,10)
    cd = np.arange(-100,100,5)

for panel in range(3):
    if (panel == 0):
        da = DS['mld'].mean(dim='model',skipna=True)
        ct = co
        norm = colors.BoundaryNorm(ct,256)
    elif (panel == 1):
        da = DS['obs']
        ct = co
        norm = colors.BoundaryNorm(ct,256)
    else:
        da = DS['mldbias'].mean(dim='model',skipna=True)
        ct = cd
        norm = colors.BoundaryNorm(ct,256)

    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=ct,
            extend='both',
            cbar_kwargs={'orientation': 'vertical',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'label': "",
                         'ticks': ct,},
            transform=ccrs.PlateCarree())


    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_xlabel('')
    ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top'})
    ax[panel].tick_params(labelsize=9)
    ax[panel].background_patch.set_facecolor('lightgray')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
