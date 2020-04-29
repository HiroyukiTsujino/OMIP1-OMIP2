# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4


xlim = [ [-90, -30], [-30, 90], [-30, 30], [-30, 60] ]


metainfo = [ json.load(open("./json/zms_omip1.json")),
             json.load(open("./json/zms_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ 'Southern Ocean', 'Atlantic Ocean', 
          'Indian Ocean', 'Pacific Ocean' ]
title2 = [ '(a) OMIP1 - WOA13v2', '(b) OMIP2 - WOA13v2', '(c) OMIP2 - OMIP1', '(d) WOA13v2' ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

dz33 = np.array(np.empty((33)),dtype=np.float64)
dz33[0] = 0.5 * lev33[1]
for k in range(1,32):
    dz33[k] = 0.5 * (lev33[k+1] + lev33[k]) - 0.5 * (lev33[k] + lev33[k-1])
dz33[32] = 500.0

if len(sys.argv) == 1:
    outfile = './fig/ZMS_bias.png'
    suptitle = 'Multi Model Mean' + ' Zonal mean salinity (ave. from 1980 to 2009)'
else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    if ( sys.argv[1] == 'Kiel-NEMO'):
        if (len(sys.argv) == 3):
            if (sys.argv[2] == 'filt'):
                outfile = './fig/ZMS_bias_' + sys.argv[1] + '_filt.png'
                filter = 'yes'
            else:
                outfile = './fig/ZMS_bias_' + sys.argv[1] + '_asis.png'
                filter = 'no'
        else:
            outfile = './fig/ZMS_bias_' + sys.argv[1] + '_asis.png'
    else:
        outfile = './fig/ZMS_bias_' + sys.argv[1] + '.png'
    suptitle = sys.argv[1] + ' Zonal mean salinity (ave. from 1980 to 2009)'


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]


#J データ読込・平均
print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_s_basin.1000'
da_ref = xr.open_dataset( reffile, decode_times=False)["so"].mean(dim='time')
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
            so_glb = ncglb.variables['so_global'][:,:,:]
            ncglb.close()
            so_glb = np.where(so_glb > 9.0e36, np.NaN, so_glb)
            if (filter == 'yes'):
                so_glb = np.where(so_glb > 40.0, np.NaN, so_glb)
                so_glb = np.where(so_glb < 30.0, np.NaN, so_glb)
                so_glb[0:nyr,9:33,0:180] = np.where(so_glb[0:nyr,9:33,0:180] < 33.0, np.NaN, so_glb[0:nyr,9:33,0:180])
            
            ncatl = netCDF4.Dataset(infile_atl,'r')
            so_atl = ncatl.variables['so_atl'][:,:,:]
            ncatl.close()
            so_atl = np.where(so_atl > 9.0e36, np.NaN, so_atl)
            if (filter == 'yes'):
                so_atl = np.where(so_atl > 40.0, np.NaN, so_atl)
                so_atl = np.where(so_atl < 30.0, np.NaN, so_atl)
                so_atl[0:nyr,9:33,0:180] = np.where(so_atl[0:nyr,9:33,0:180] < 33.0, np.NaN, so_atl[0:nyr,9:33,0:180])

            ncind = netCDF4.Dataset(infile_ind,'r')
            so_ind = ncind.variables['so_ind'][:,:,:]
            ncind.close()
            so_ind = np.where(so_ind > 9.0e36, np.NaN, so_ind)
            if (filter == 'yes'):
                so_ind = np.where(so_ind > 40.0, np.NaN, so_ind)
                so_ind = np.where(so_ind < 30.0, np.NaN, so_ind)
                so_ind[0:nyr,9:33,0:180] = np.where(so_ind[0:nyr,9:33,0:180] < 33.0, np.NaN, so_ind[0:nyr,9:33,0:180])

            ncpac = netCDF4.Dataset(infile_pac,'r')
            so_pac = ncpac.variables['so_pac'][:,:,:]
            ncpac.close()
            so_pac = np.where(so_pac > 9.0e36, np.NaN, so_pac)
            if (filter == 'yes'):
                so_pac = np.where(so_pac > 40.0, np.NaN, so_pac)
                so_pac = np.where(so_pac < 30.0, np.NaN, so_pac)
                so_pac[0:nyr,9:33,0:180] = np.where(so_pac[0:nyr,9:33,0:180] < 33.0, np.NaN, so_pac[0:nyr,9:33,0:180])

            if ( omip == 0 ):
                so_all = np.array(np.zeros((62,4,33,180)),dtype=np.float32)
                so_all[0:62,0,0:33,0:180] = so_glb[0:62,0:33,0:180]
                so_all[0:62,1,0:33,0:180] = so_atl[0:62,0:33,0:180]
                so_all[0:62,2,0:33,0:180] = so_ind[0:62,0:33,0:180]
                so_all[0:62,3,0:33,0:180] = so_pac[0:62,0:33,0:180]
                #print(so_glb[0:62,0,90])
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

bounds1 = [-1.0, -0.5, -0.3, -0.2, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.2, 0.3, 0.5, 1.0]
bounds2 = [-0.2, -0.15, -0.1, -0.07, -0.04, -0.02, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2 ]
bounds3 = [33.0, 34.0, 34.2, 34.4, 34.6, 34.7, 34.8, 34.9, 35.0, 35.2, 35.5, 35.8, 36.1, 36.5, 36.9 ]

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
        ticks_bounds = bounds3
    if item[panel] == 'obs':
        da = DS[item[panel]]
    else:
        da = DS[item[panel]].mean(dim='model',skipna=False)
    for m in range(4):
        tmp = da.isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1]))
        nz, ny = tmp.shape
        dztmp = np.tile(dz33,ny).reshape(ny,nz).T
        msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
        tmp1 = (datmp * datmp * msktmp * dztmp).sum()
        tmp2 = (msktmp * dztmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        title_bas = title[m] + '\n ' \
            + 'rmse = ' + '{:.2f}'.format(rmse) + ' psu'

        tmp_bias=tmp.values
        if (panel < 2 and m == 2):
            for k in range(nz):
                for j in range(ny):
                    if (abs(tmp_bias[k,j]) > 1.0):
                        print(m,k,j,dztmp[k,j],tmp_bias[k,j])
        da.isel(basin=m).plot(ax=ax[panel][m],cmap=cmap[panel],
                              levels=bounds,
                              extend='both',
                              cbar_kwargs={'orientation': 'horizontal',
#                                           'spacing':'proportional',
                                           'spacing':'uniform',
                                           'ticks': ticks_bounds,},
                              cbar_ax=ax_cbar[panel],
                              add_labels=False,add_colorbar=True)
        ax[panel][m].set_title(title_bas,{'fontsize':9, 'verticalalignment':'top', 'linespacing':0.8})
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
