# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4


xlim = [ [-90, -30], [-30, 90], [-30, 30], [-30, 60] ]


metainfo = [ json.load(open("./json/zmt_omip1.json")),
             json.load(open("./json/zmt_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ 'Southern Ocean', 'Atlantic Ocean', 
          'Indian Ocean', 'Pacific Ocean' ]
title2 = [ '(a) OMIP1 - WOA13v2', '(b) OMIP2 - WOA13v2',
           '(c) OMIP2 - OMIP1', '(d) WOA13v2' ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

if len(sys.argv) == 1:
    outfile = './fig/ZMT_bias.png'
    suptitle = 'Multi Model Mean' + ' Zonal mean temperature (ave. from 1980 to 2009)'

else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    if ( sys.argv[1] == 'Kiel-NEMO'):
        if (len(sys.argv) == 3):
            if (sys.argv[2] == 'filt'):
                outfile = './fig/ZMT_bias_' + sys.argv[1] + '_filt.png'
                filter = 'yes'
            else:
                outfile = './fig/ZMT_bias_' + sys.argv[1] + '_asis.png'
                filter = 'no'
        else:
            outfile = './fig/ZMT_bias_' + sys.argv[1] + '_asis.png'
    else:
        outfile = './fig/ZMT_bias_' + sys.argv[1] + '_asis.png'
    suptitle = sys.argv[1] + ' Zonal mean temperature (ave. from 1980 to 2009)'

print("Drawing " + suptitle)

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]


#J データ読込・平均
print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_th_basin.1000'
da_ref = xr.open_dataset( reffile, decode_times=False)["thetao"].mean(dim='time')
da_ref = da_ref.assign_coords(basin=[0,1,2,3])


data = []
for omip in range(2):
    d = np.empty( (len(model_list[omip]),4,33,180) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    if ( omip == 0 ):
        nyr = 62
    else:
        nyr = 61

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
            thetao_glb = np.where(thetao_glb > 9.0e36, np.NaN, thetao_glb)
            if (filter == 'yes'):
                thetao_glb = np.where(thetao_glb > 40, np.NaN, thetao_glb)
                thetao_glb = np.where(thetao_glb < -2, np.NaN, thetao_glb)
                for n in range(0,nyr):
                    for k in range(0,32):
                        for j in range(0,180):
                            if (np.isnan(thetao_glb[n,k+1,j]) and abs(thetao_glb[n,k,j]) < 1.0e-4):
                                thetao_glb[n,k,j] = np.NaN
                    for j in range(0,180):
                        if (abs(thetao_glb[n,32,j]) < 1.0e-4):
                            thetao_glb[n,32,j] = np.NaN
            
            ncatl = netCDF4.Dataset(infile_atl,'r')
            thetao_atl = ncatl.variables['thetao_atl'][:,:,:]
            ncatl.close()
            thetao_atl = np.where(thetao_atl > 9.0e36, np.NaN, thetao_atl)
            if (filter == 'yes'):
                thetao_atl = np.where(thetao_atl > 40, np.NaN, thetao_atl)
                thetao_atl = np.where(thetao_atl < -2, np.NaN, thetao_atl)
                for n in range(0,nyr):
                    for k in range(0,32):
                        for j in range(0,180):
                            if (np.isnan(thetao_atl[n,k+1,j]) and abs(thetao_atl[n,k,j]) < 1.0e-4):
                                thetao_atl[n,k,j] = np.NaN
                    for j in range(0,180):
                        if (abs(thetao_atl[n,32,j]) < 1.0e-4):
                            thetao_atl[n,32,j] = np.NaN

            ncind = netCDF4.Dataset(infile_ind,'r')
            thetao_ind = ncind.variables['thetao_ind'][:,:,:]
            ncind.close()
            thetao_ind = np.where(thetao_ind > 9.0e36, np.NaN, thetao_ind)
            if (filter == 'yes'):
                thetao_ind = np.where(thetao_ind > 40, np.NaN, thetao_ind)
                thetao_ind = np.where(thetao_ind < -2, np.NaN, thetao_ind)
                for n in range(0,nyr):
                    for k in range(0,32):
                        for j in range(0,180):
                            if (np.isnan(thetao_ind[n,k+1,j]) and abs(thetao_ind[n,k,j]) < 1.0e-4):
                                thetao_ind[n,k,j] = np.NaN
                    for j in range(0,180):
                        if (abs(thetao_ind[n,32,j]) < 1.0e-4):
                            thetao_ind[n,32,j] = np.NaN

            ncpac = netCDF4.Dataset(infile_pac,'r')
            thetao_pac = ncpac.variables['thetao_pac'][:,:,:]
            ncpac.close()
            thetao_pac = np.where(thetao_pac > 9.0e36, np.NaN, thetao_pac)
            if (filter == 'yes'):
                thetao_pac = np.where(thetao_pac > 40, np.NaN, thetao_pac)
                thetao_pac = np.where(thetao_pac < -2, np.NaN, thetao_pac)
                for n in range(0,nyr):
                    for k in range(0,32):
                        for j in range(0,180):
                            if (np.isnan(thetao_pac[n,k+1,j]) and abs(thetao_pac[n,k,j]) < 1.0e-4):
                                thetao_pac[n,k,j] = np.NaN
                    for j in range(0,180):
                        if (abs(thetao_pac[n,32,j]) < 1.0e-4):
                            thetao_pac[n,32,j] = np.NaN

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

        tmp = DS_read[var].sel(time=slice(1980,2009)).mean(dim='time')

        if model == 'NorESM-O-CICE':
            tmp = tmp.transpose().interp(depth=lev33)
        if model == 'AWI-FESOM':
            tmp = tmp.transpose()
        if model == "MIROC-COCO4-9":
            tmp = tmp.sel(lat=slice(None, None, -1))
        if model == 'BSC-NEMO':
            tmp = tmp.transpose("basin","depth","lat")
        if model == 'GFDL-MOM':
            #print(tmp)
            tmp = tmp.interp(z_l=lev33)

        d[nmodel] = tmp.values
        nmodel += 1

    data += [d]

bias = data - da_ref.values

DS = xr.Dataset({'omip1bias': (['model','basin','depth','lat'], bias[0]),
                 'omip2bias': (['model','basin','depth','lat'], bias[1]),
                 'omip2-1': (['model','basin','depth','lat'], data[1]-data[0]),
                 'obs': (['basin','depth','lat'], da_ref.values), },
                coords = {'depth': lev33, 'lat': np.linspace(-89.5,89.5,num=180) } )


#J 描画

fig = plt.figure(figsize=(16,12))
fig.suptitle( suptitle, fontsize=20 )

# [left, bottom, width, height]
axes0 = np.array( [ [0.07, 0.6, 0.06, 0.3],
                    [0.14, 0.6, 0.12, 0.3],
                    [0.27, 0.6, 0.06, 0.3],
                    [0.34, 0.6, 0.09, 0.3],
                    [0.07, 0.55, 0.36, 0.015], ])
ax = [ [ plt.axes(axes0[0]),
         plt.axes(axes0[1]),
         plt.axes(axes0[2]),
         plt.axes(axes0[3]), ],
       [ plt.axes(axes0[0]+np.array([0.5,0,0,0])),
         plt.axes(axes0[1]+np.array([0.5,0,0,0])),
         plt.axes(axes0[2]+np.array([0.5,0,0,0])),
         plt.axes(axes0[3]+np.array([0.5,0,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.5,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.5,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.5,0,0])),
         plt.axes(axes0[3]+np.array([0,-0.5,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.5,-0.5,0,0])),
         plt.axes(axes0[1]+np.array([0.5,-0.5,0,0])),
         plt.axes(axes0[2]+np.array([0.5,-0.5,0,0])),
         plt.axes(axes0[3]+np.array([0.5,-0.5,0,0])), ] ]
ax_cbar = [ plt.axes(axes0[4]),
            plt.axes(axes0[4]+np.array([0.5,0,0,0])),
            plt.axes(axes0[4]+np.array([0,-0.5,0,0])),
            plt.axes(axes0[4]+np.array([0.5,-0.5,0,0])), ]

bounds1 = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0]
bounds2 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
bounds3 = np.arange(-1,30.1,1)
ticks_bounds3 = [0, 5, 10, 15, 20, 25, 30] 

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]

for panel in range(4):
    if item[panel] == 'omip1bias' or item[panel] == 'omip2bias':
        bounds = bounds1
        ticks_bounds = bounds1
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
    if item[panel] == 'obs':
        da = DS[item[panel]]
    else:
        da = DS[item[panel]].mean(dim='model',skipna=False)
    for m in range(4):
        da.isel(basin=m).plot(ax=ax[panel][m],cmap=cmap[panel],
                              levels=bounds,
                              extend='both',
                              cbar_kwargs={'orientation': 'horizontal',
#                                           'spacing':'proportional',
                                           'spacing':'uniform',
                                           'ticks': ticks_bounds,},
                              cbar_ax=ax_cbar[panel],
                              add_labels=False,add_colorbar=True)
        ax[panel][m].set_title(title[m])
        ax[panel][m].invert_yaxis()
        ax[panel][m].set_xlim(xlim[m][0],xlim[m][1])
        ax[panel][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
        ax[panel][m].set_facecolor('lightgray')
    for m in range(1,4):
        ax[panel][m].tick_params(axis='y',labelleft=False)

fig.text(0.25,0.94,title2[0],fontsize=16,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.75,0.94,title2[1],fontsize=16,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.25,0.44,title2[2],fontsize=16,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.75,0.44,title2[3],fontsize=16,
         horizontalalignment='center',verticalalignment='center')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
