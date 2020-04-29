# -*- coding: utf-8 -*-
import sys
sys.path.append("../../python")
import json
import numpy as np
import xarray as xr
import netCDF4
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime
from uncertain_Wakamatsu import uncertain_2d

from numba import jit


@jit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],i4,i4,i4,i4)', nopython=True)
def onedeg2twodeg(d_two, d_one, am_one, nx, ny, nxm, nym):

    d_two[ny-1,0] = am_one[2*ny-1,0] * d_one[2*ny-1,0] \
                  + am_one[2*ny-1,nxm-1] * d_one[2*ny-1,nxm-1]
    for i in range(1,nx):
        d_two[ny-1,i] = am_one[2*ny-1,2*i-1] * d_one[2*ny-1,2*i-1] \
                      + am_one[2*ny-1,2*i] * d_one[2*ny-1,2*i]
    for j in range(0,ny-1):
        d_two[j,0] = am_one[2*j+1,0]     * d_one[2*j+1,0] \
                   + am_one[2*j+1,nxm-1] * d_one[2*j+1,nxm-1] \
                   + am_one[2*j+2,0]     * d_one[2*j+2,0] \
                   + am_one[2*j+2,nxm-1] * d_one[2*j+2,nxm-1] 
        for i in range(1,nx):
            d_two[j,i] = am_one[2*j+1,2*i-1] * d_one[2*j+1,2*i-1] \
                       + am_one[2*j+2,2*i-1] * d_one[2*j+2,2*i-1] \
                       + am_one[2*j+1,2*i] * d_one[2*j+1,2*i] \
                       + am_one[2*j+2,2*i] * d_one[2*j+2,2*i] 
    return d_two

#---------------------------------------------------------------------

if (len(sys.argv) < 2):
    print ('Usage: '+ sys.argv[0] + ' [season (summer or winter)]')
    sys.exit()


season=sys.argv[1]

stclyr = 1980
edclyr = 2009
nyrcl = edclyr - stclyr + 1

factor_5ptail = 1.64  # 5-95%
num_bootstraps = 10000

metainfo = [ json.load(open("./json/mld_season_omip1.json")), 
             json.load(open("./json/mld_season_omip2.json")) ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

suptitle = 'Multi Model Mean' + ' (MLD ave. from '+str(stclyr)+' to '+str(edclyr)+')'
outfile = './fig/MLD_'+str(season)+'_diff_bias.png'

#----------------------------------------------

print( "Loading IFREMER data" )
if (season == 'Summer'):
    reffile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_sumclim.nc'
    mskfile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_summask.nc'
else:
    reffile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_winclim.nc'
    mskfile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_windmask.nc'
    
# Obs
ncobsann = netCDF4.Dataset(reffile,'r')
nx = len(ncobsann.dimensions['lon'])
ny = len(ncobsann.dimensions['lat'])
mldobs_ann = ncobsann.variables['mlotst'][:,:]
miss_val_obsann = ncobsann.variables['mlotst'].missing_value
lon_ = ncobsann.variables['lon'][:]
lat_ = ncobsann.variables['lat'][:]

dtwoout = np.array(np.empty((ny,nx)),dtype=np.float64)

# cell area (1x1)

path_amip = '../refdata/PCMDI-SST'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nxm = len(ncare.dimensions['lon'])
nym = len(ncare.dimensions['lat'])
aream = ncare.variables['areacello'][:,:]
ncare.close()
        
# cell area (2x2)

area = np.array(np.empty((ny,nx)),dtype=np.float64)

area[ny-1,0] = aream[2*ny-1,0] + aream[2*ny-1,nxm-1]
for i in range(1,nx):
    area[ny-1,i] = aream[2*ny-1,2*i-1] + aream[2*ny-1,2*i]

for j in range(0,ny-1):
    area[j,0] = aream[2*j+1,0] + aream[2*j+1,nxm-1] \
              + aream[2*j+2,0] + aream[2*j+2,nxm-1]
    for i in range(1,nx):
        area[j,i] = aream[2*j+1,2*i-1] + aream[2*j+2,2*i-1] \
                  + aream[2*j+1,2*i]   + aream[2*j+2,2*i]

#----------------------------------------------


#J 時刻情報 (各モデルの時刻情報を上書きする)
time0 = np.empty(nyrcl,dtype='object')
for yr in range(stclyr,edclyr+1):
    time0[yr-stclyr] = datetime.datetime(yr,1,1)

#J データ読込・平均
data = []
for omip in range(2):

    if (omip == 0):
        styr=1948
        edyr=2009
    else:
        styr=1958
        edyr=2018

    nyr = edyr - styr + 1
    print( "Loading OMIP" + str(omip+1) + " data" )

    d = np.empty( (len(model_list[omip]),nyrcl,ny,nx) )

    nmodel = 0
    for model in model_list[omip]:

        path = metainfo[omip][model]['path']
        if (season == 'summer'): 
            fdname = metainfo[omip][model]['fsumname']
        else:
            fdname = metainfo[omip][model]['fwinname']

        fmname = metainfo[omip][model]['fmskname']
        datafile = path + '/' + fdname
        maskfile = path + '/' + fmname
        #DS = xr.open_dataset( datafile )

        print (' ')
        print ('Processing ', model)

        print(datafile)
        ncomipann = netCDF4.Dataset(datafile,'r')
        mldomip_ann = ncomipann.variables['mlotst'][:,:,:]
        miss_val_omipann = ncomipann.variables['mlotst'].missing_value
        mldomip_ann = np.where(np.isnan(mldomip_ann),0.0,mldomip_ann)

        ncmskomip = netCDF4.Dataset(maskfile,'r')
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
                donein = mldomip_ann[nn,:,:]
                dtwoout[:,:] = 0.0
                onedeg2twodeg(dtwoout, donein, areamask, nx, ny, nxm, nym)
                mldannm[nm,:,:] = dtwoout
                nm += 1
            nn += 1

        mldannm[0:nyrcl,:,:] = mldannm[0:nyrcl,:,:] / (1.0 - mask_all[:,:] + amsk_all[:,:]) 
        mldannm[0:nyrcl,:,:] = np.where(mask_all[:,:] == 0.0, np.NaN, mldannm[0:nyrcl,:,:])
        
        d[nmodel] = mldannm
        nmodel += 1
    
        del amsk_all
        del mask_all
        del mldannm

    data += [d]
    del d

DS = xr.Dataset( {'omip2-1': (['model','time','lat','lon'], data[1] - data[0]), },
                 coords = { 'time': time0,
                            'lat': lat_, 'lon': lon_ ,}) 

print( 'Calculating OMIP2 - OMIP1' )
dout = uncertain_2d( DS['omip2-1'].values, num_bootstraps )
DS_stats = xr.Dataset( { 'mean': (['lat','lon'], dout[0]),
                       'std':  (['lat','lon'], dout[1]),
                       'M':    (['lat','lon'], dout[2]),
                       'V':    (['lat','lon'], dout[3]),
                       'B':    (['lat','lon'], dout[4]),},
                       coords = { 'lat': lat_, 'lon': lon_, }, )

print( 'Output netCDF4' )
path_out='../analysis/STDs/'
outstatsfile= path_out + 'MLD_'+str(season)+'_omip1-omip2_stats.nc'
DS_stats.to_netcdf(path=outstatsfile,mode='w',format='NETCDF4')


#J 描画
print( 'Start drawing' )
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

if (season == 'winter'):
    bounds = np.arange(-200,200,20)
elif (season == 'summer'):
    bounds = np.arange(-100,100,5)

ticks_bounds = bounds

cmap = 'RdBu_r'
item = 'omip2-1'

ax1=plt.axes(projection=proj)

da = DS_stats["mean"]

da.plot(ax=ax1,
        cmap=cmap,
        levels=bounds,
        extend='both',
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': '[m]',
                     'ticks': ticks_bounds,},
        transform=ccrs.PlateCarree())

mpl.rcParams['hatch.color'] = 'limegreen'
mpl.rcParams['hatch.linewidth'] = 0.5

x = DS_stats["lon"].values
y = DS_stats["lat"].values
#z = np.abs(DS_stats["mean"]) - factor_5ptail * DS_stats["std"]
z = np.abs(DS_stats["mean"]) - 0.5 * DS_stats["std"]
z = np.where( z > 0, 1, np.nan )
print(DS_stats["mean"])
print(DS_stats["std"])
ax1.contourf(x,y,z,hatches=['xxxxx'],colors='none',transform=ccrs.PlateCarree())

ax1.coastlines()
ax1.set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.set_title('OMIP2 - OMIP1')
ax1.background_patch.set_facecolor('lightgray')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
