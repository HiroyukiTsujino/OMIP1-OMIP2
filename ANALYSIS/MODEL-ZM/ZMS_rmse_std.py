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

#------------------------
if (len(sys.argv) < 1):
    print ('Usage: ' + sys.argv[0] + ' mip_id' )
    sys.exit()

mip = int(sys.argv[1])
#------------------------

ystr = 1980
yend = 2009
nyr = yend - ystr + 1

suptitle = 'Multi Model Mean OMIP-' + str(mip) + ' zonal mean salinity (ave. from '+str(ystr)+' to '+str(yend)+')'
outfile = './fig/ZMS_bias_std_OMIP-'+str(mip)+'.png'

print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_s_basin.1000'
da_ref = xr.open_dataset( reffile, decode_times=False)["so"].mean(dim='time')
da_ref = da_ref.assign_coords(basin=[0,1,2,3])

#-------------------------

metainfo = [ json.load(open("./json/zms_omip1.json")), 
             json.load(open("./json/zms_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

time0 = np.empty(nyr,dtype='object')
for yr in range(ystr,yend+1):
    time0[yr-ystr] = datetime.datetime(yr,1,1)

time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

omip = mip - 1

d = np.empty( (len(model_list[omip]),nyr,4,33,180) )
print( "Loading OMIP" + str(omip+1) + " data" )

nmodel = 0
for model in model_list[omip]:

    path = metainfo[omip][model]['path']
    fname = metainfo[omip][model]['fname']
    var = metainfo[omip][model]['name']
    infile = path + '/' + fname

    if ( model == 'Kiel-NEMO' ):

        infile_glb = infile + '_global.nc'
        infile_atl = infile + '_atl.nc'
        infile_ind = infile + '_ind.nc'
        infile_pac = infile + '_pac.nc'

        ncglb = netCDF4.Dataset(infile_glb,'r')
        so_glb = ncglb.variables['so_global'][:,:,:]
        ncglb.close()
        so_glb = np.where(so_glb > 9.9e36, np.NaN, so_glb)
        
        ncatl = netCDF4.Dataset(infile_atl,'r')
        so_atl = ncatl.variables['so_atl'][:,:,:]
        ncatl.close()
        so_atl = np.where(so_atl > 9.9e36, np.NaN, so_atl)

        ncind = netCDF4.Dataset(infile_ind,'r')
        so_ind = ncind.variables['so_ind'][:,:,:]
        ncind.close()
        so_ind = np.where(so_ind > 9.9e36, np.NaN, so_ind)

        ncpac = netCDF4.Dataset(infile_pac,'r')
        so_pac = ncpac.variables['so_pac'][:,:,:]
        ncpac.close()
        so_pac = np.where(so_pac > 9.9e36, np.NaN, so_pac)

        if ( omip == 0 ):
            so_all = np.array(np.zeros((62,4,33,180)),dtype=np.float32)
            so_all[0:62,0,0:33,0:180] = so_glb[0:62,0:33,0:180]
            so_all[0:62,1,0:33,0:180] = so_atl[0:62,0:33,0:180]
            so_all[0:62,2,0:33,0:180] = so_ind[0:62,0:33,0:180]
            so_all[0:62,3,0:33,0:180] = so_pac[0:62,0:33,0:180]
            DS_read = xr.Dataset({'so': (['time','basin','depth','lat'], so_all)},
                                 coords = {'time' : time[omip], 'depth': lev33, 'lat': np.linspace(-89.5,89.5,num=180) } )

        else:
            so_all = np.array(np.zeros((61,4,33,180)),dtype=np.float32)
            so_all[0:61,0,0:33,0:180] = so_glb[0:61,0:33,0:180]
            so_all[0:61,1,0:33,0:180] = so_atl[0:61,0:33,0:180]
            so_all[0:61,2,0:33,0:180] = so_ind[0:61,0:33,0:180]
            so_all[0:61,3,0:33,0:180] = so_pac[0:61,0:33,0:180]
            DS_read = xr.Dataset({'so': (['time','basin','depth','lat'], so_all)},
                                 coords = {'time' : time[omip], 'depth': lev33, 'lat': np.linspace(-89.5,89.5,num=180) } )

    else:
        DS_read = xr.open_dataset(infile,decode_times=False)

        DS_read['time'] = time[omip]

    tmp = DS_read[var].sel(time=slice(1980,2009))

    if model == 'NorESM-BLOM':
        tmp = tmp.transpose().interp(depth=lev33)
    if model == 'AWI-FESOM':
        tmp = tmp.transpose()
    if model == "MIROC-COCO4.9":
        tmp = tmp.sel(lat=slice(None, None, -1))
    if model == 'EC-Earth3-NEMO':
        tmp = tmp.transpose("time","basin","depth","lat")
    if model == 'GFDL-MOM':
        tmp = tmp.interp(z_l=lev33)

    d[nmodel] = tmp.values.reshape(nyr,4,33,180)
    nmodel += 1

print( 'Calculating bias and model std' )

# absolute difference
mmm = np.mean(d,axis=(0,1)) # multi model mean
mmm_bias = np.abs(mmm - da_ref.values)

tmm = np.mean(d,axis=1) # time mean
tmm_std = np.std(tmm,axis=0,ddof=1) # model std of time mean

DS_stats = xr.Dataset( { 'bias': (['basin','depth','lat'], mmm_bias),
                         'uncertainty':  (['basin','depth','lat'], tmm_std*2),},
                       coords = { 'depth': lev33, 'lat': np.linspace(-89.5,89.5,num=180),}, )


print( 'Output netCDF4' )
path_out='../analysis/STDs/'
outstatsfile=path_out + 'ZMS_omip-'+str(omip+1)+'-STDs.nc'
DS_stats.to_netcdf(path=outstatsfile,mode='w',format='NETCDF4')


#J 描画
print( 'Start drawing' )
#J 描画

fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=14 )

title = [ 'Southern Ocean', 'Atlantic Ocean', 
          'Indian Ocean', 'Pacific Ocean' ]
xlim = [ [-90, -30], [-30, 90], [-30, 30], [-30, 60] ]

# [left, bottom, width, height]
axes0 = np.array( [ [0.07, 0.55, 0.14, 0.35],
                    [0.25, 0.55, 0.26, 0.35],
                    [0.55, 0.55, 0.15, 0.35],
                    [0.74, 0.55, 0.20, 0.35], ],
)
ax = [ [ plt.axes(axes0[0]),
         plt.axes(axes0[1]),
         plt.axes(axes0[2]),
         plt.axes(axes0[3]), ],
       [ plt.axes(axes0[0]+np.array([0,-0.48,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.48,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.48,0,0])),
         plt.axes(axes0[3]+np.array([0,-0.48,0,0])), ] ]

# [left, bottom, width, height]
ax_cbar = [ plt.axes([0.1, 0.50, 0.8, 0.02]),
            plt.axes([0.1, 0.02, 0.8, 0.02]) ]

bounds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
cmap = 'YlOrBr'
ticks_bounds = bounds

for m in range(4):
    DS_stats['bias'].isel(basin=m).plot(ax=ax[0][m],cmap=cmap,
                                        levels=bounds,
                                        extend='both',
                                        cbar_kwargs={'orientation': 'horizontal',
                                                     'spacing':'uniform',
                                                     'ticks': ticks_bounds,},
                                        cbar_ax=ax_cbar[0],
                                        add_labels=False,add_colorbar=True)
    ax[0][m].set_title(title[m],{'fontsize':10, 'verticalalignment':'top'})

    mpl.rcParams['hatch.color'] = 'limegreen'
    mpl.rcParams['hatch.linewidth'] = 0.5

    x = DS_stats["lat"].values
    y = DS_stats["depth"].values
    z = np.abs(DS_stats["bias"].isel(basin=m)) - DS_stats["uncertainty"].isel(basin=m)
    z = np.where( z > 0, 1, np.nan )
    ax[0][m].contourf(x,y,z,hatches=['xxxxx'],colors='none')

    ax[0][m].tick_params(labelsize=9)
    ax[0][m].invert_yaxis()
    ax[0][m].set_xlim(xlim[m][0],xlim[m][1])
    ax[0][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
    ax[0][m].set_facecolor('lightgray')

    for m in range(1,4):
        ax[0][m].tick_params(axis='y',labelleft=False)

for m in range(4):
    DS_stats['uncertainty'].isel(basin=m).plot(ax=ax[1][m],cmap=cmap,
                                        levels=bounds,
                                        extend='both',
                                        cbar_kwargs={'orientation': 'horizontal',
                                                     'spacing':'uniform',
                                                     'ticks': ticks_bounds,},
                                        cbar_ax=ax_cbar[1],
                                        add_labels=False,add_colorbar=True)
    ax[1][m].set_title(title[m],{'fontsize':10, 'verticalalignment':'top'})
    ax[1][m].tick_params(labelsize=9)
    ax[1][m].invert_yaxis()
    ax[1][m].set_xlim(xlim[m][0],xlim[m][1])
    ax[1][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
    ax[1][m].set_facecolor('lightgray')

    for m in range(1,4):
        ax[1][m].tick_params(axis='y',labelleft=False)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
