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


ystr = 1980
yend = 2009
nyr = yend - ystr + 1

factor_5ptail = 1.64  # 5-95%
num_bootstraps = 10000

metainfo = [ json.load(open("./json/zmt_omip1.json")), 
             json.load(open("./json/zmt_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

suptitle = 'Multi Model Mean' + ' zonal mean temperature (ave. from '+str(ystr)+' to '+str(yend)+')'
outfile = './fig/ZMT_diff_bias.png'

time0 = np.empty(nyr,dtype='object')
for yr in range(ystr,yend+1):
    time0[yr-ystr] = datetime.datetime(yr,1,1)

time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

data = []
for omip in range(2):
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
            thetao_glb = ncglb.variables['thetao_global'][:,:,:]
            ncglb.close()
            thetao_glb = np.where(thetao_glb > 9.9e36, np.NaN, thetao_glb)
            
            ncatl = netCDF4.Dataset(infile_atl,'r')
            thetao_atl = ncatl.variables['thetao_atl'][:,:,:]
            ncatl.close()
            thetao_atl = np.where(thetao_atl > 9.9e36, np.NaN, thetao_atl)

            ncind = netCDF4.Dataset(infile_ind,'r')
            thetao_ind = ncind.variables['thetao_ind'][:,:,:]
            ncind.close()
            thetao_ind = np.where(thetao_ind > 9.9e36, np.NaN, thetao_ind)

            ncpac = netCDF4.Dataset(infile_pac,'r')
            thetao_pac = ncpac.variables['thetao_pac'][:,:,:]
            ncpac.close()
            thetao_pac = np.where(thetao_pac > 9.9e36, np.NaN, thetao_pac)

            if ( omip == 0 ):
                thetao_all = np.array(np.zeros((62,4,33,180)),dtype=np.float32)
                thetao_all[0:62,0,0:33,0:180] = thetao_glb[0:62,0:33,0:180]
                thetao_all[0:62,1,0:33,0:180] = thetao_atl[0:62,0:33,0:180]
                thetao_all[0:62,2,0:33,0:180] = thetao_ind[0:62,0:33,0:180]
                thetao_all[0:62,3,0:33,0:180] = thetao_pac[0:62,0:33,0:180]
                #print(thetao_glb[0:62,0,90])
                DS_read = xr.Dataset({'thetao': (['time','basin','depth','lat'], thetao_all)},
                                 coords = {'time' : time[omip], 'depth': lev33, 'lat': np.linspace(-89.5,89.5,num=180) } )

            else:
                thetao_all = np.array(np.zeros((61,4,33,180)),dtype=np.float32)
                thetao_all[0:61,0,0:33,0:180] = thetao_glb[0:61,0:33,0:180]
                thetao_all[0:61,1,0:33,0:180] = thetao_atl[0:61,0:33,0:180]
                thetao_all[0:61,2,0:33,0:180] = thetao_ind[0:61,0:33,0:180]
                thetao_all[0:61,3,0:33,0:180] = thetao_pac[0:61,0:33,0:180]
                DS_read = xr.Dataset({'thetao': (['time','basin','depth','lat'], thetao_all)},
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

    data += [d]

DS = xr.Dataset( {'omip2-1': (['model','time','basin','depth','lat'], data[1] - data[0]), },
                 coords = { 'time': time0,
                            'lat': np.linspace(-89.5,89.5,num=180)}) 

print( 'Calculating OMIP2 - OMIP1' )

tmp_stats=np.array(np.zeros((5,4,33,180)),dtype=np.float32)
#tmp_series=np.array(np.zeros((nmodel,nyr,33,180)),dtype=np.float32)
for m in range(4):
    tmp_series=DS['omip2-1'].isel(basin=m).values
    dout = uncertain_2d( tmp_series, num_bootstraps )
    tmp_stats[0,m,:,:]=dout[0]
    tmp_stats[1,m,:,:]=dout[1]
    tmp_stats[2,m,:,:]=dout[2]
    tmp_stats[3,m,:,:]=dout[3]
    tmp_stats[4,m,:,:]=dout[4]

DS_stats = xr.Dataset( { 'mean': (['basin','depth','lat'], tmp_stats[0]),
                         'std':  (['basin','depth','lat'], tmp_stats[1]),
                         'M':    (['basin','depth','lat'], tmp_stats[2]),
                         'V':    (['basin','depth','lat'], tmp_stats[3]),
                         'B':    (['basin','depth','lat'], tmp_stats[4]),},
                       coords = { 'depth': lev33, 'lat': np.linspace(-89.5,89.5,num=180),}, )

print( 'Output netCDF4' )
path_out='../analysis/STDs/'
outstatsfile = path_out + 'ZMT_omip1-omip2_stats.nc'
DS_stats.to_netcdf(path=outstatsfile,mode='w',format='NETCDF4')


#J 描画
print( 'Start drawing' )
#J 描画

fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

title = [ 'Southern Ocean', 'Atlantic Ocean', 
          'Indian Ocean', 'Pacific Ocean' ]
xlim = [ [-90, -30], [-30, 90], [-30, 30], [-30, 60] ]

# [left, bottom, width, height]
axes0 = np.array( [ [0.07, 0.10, 0.12, 0.80],
                    [0.25, 0.10, 0.24, 0.80],
                    [0.55, 0.10, 0.12, 0.80],
                    [0.70, 0.10, 0.18, 0.80], ])
ax = [ plt.axes(axes0[0]),
         plt.axes(axes0[1]),
         plt.axes(axes0[2]),
         plt.axes(axes0[3]), ]

# [left, bottom, width, height]
ax_cbar = plt.axes([0.02, 0.05, 0.7, 0.02])

bounds = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
cmap = 'RdBu_r'
item = 'omip2-1'
ticks_bounds = bounds

for m in range(4):
    DS_stats['mean'].isel(basin=m).plot(ax=ax[m],cmap=cmap,
                                        levels=bounds,
                                        extend='both',
                                        cbar_kwargs={'orientation': 'horizontal',
                                                     'spacing':'uniform',
                                                     'ticks': ticks_bounds,},
                                        cbar_ax=ax_cbar,
                                        add_labels=False,add_colorbar=True)
    ax[m].set_title(title[m],{'fontsize':8, 'verticalalignment':'top'})

    mpl.rcParams['hatch.color'] = 'limegreen'
#    mpl.rcParams['hatch.color'] = 'black'
    mpl.rcParams['hatch.linewidth'] = 0.5

    x = DS_stats["lat"].values
    y = DS_stats["depth"].values
    z = np.abs(DS_stats["mean"].isel(basin=m)) - factor_5ptail * DS_stats["std"].isel(basin=m)
    z = np.where( z > 0, 1, np.nan )

    ax[m].contourf(x,y,z,hatches=['xxxxx'],colors='none')
    ax[m].tick_params(labelsize=9)
    ax[m].invert_yaxis()
    ax[m].set_xlim(xlim[m][0],xlim[m][1])
    ax[m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
    ax[m].set_facecolor('lightgray')

    for m in range(1,4):
        ax[m].tick_params(axis='y',labelleft=False)

fig.text(0.25,0.925,'OMIP2-OMIP1',fontsize=12,
         horizontalalignment='center',verticalalignment='center')
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
