# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4

if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1 - WOA13v2', 'OMIP2 - WOA13v2', 'OMIP2 - OMIP1' ]

suptitle = head_title[nv_out]  + ' Zonal mean temperature (ave. from 1980 to 2009)'

metainfo = [ json.load(open("./json/zmt_omip1.json")),
             json.load(open("./json/zmt_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ 'Southern Ocean', 'Atlantic Ocean', 
          'Indian Ocean', 'Pacific Ocean' ]

basin_name = [ 'Southern', 'Atlantic', 'Indian', 'Pacific' ]
title_bas = ['a', 'b', 'c', 'd']
index_bas = ['a', 'b', 'c', 'd']

xlim = [ [-90, -30], [-30, 90], [-30, 30], [-30, 60] ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

dz33 = np.array(np.empty((33)),dtype=np.float64)
dz33[0] = 0.5 * lev33[1]
for k in range(1,32):
    dz33[k] = 0.5 * (lev33[k+1] + lev33[k]) - 0.5 * (lev33[k] + lev33[k-1])
dz33[32] = 500.0

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

            thetao_all = np.array(np.zeros((nyr,4,33,180)),dtype=np.float32)
            thetao_all[0:nyr,0,0:33,0:180] = thetao_glb[0:nyr,0:33,0:180]
            thetao_all[0:nyr,1,0:33,0:180] = thetao_atl[0:nyr,0:33,0:180]
            thetao_all[0:nyr,2,0:33,0:180] = thetao_ind[0:nyr,0:33,0:180]
            thetao_all[0:nyr,3,0:33,0:180] = thetao_pac[0:nyr,0:33,0:180]
            DS_read = xr.Dataset({'thetao': (['time','basin','depth','lat'], thetao_all)},
                                 coords = {'time' : time[omip], 'depth': lev33, 'lat': np.linspace(-89.5,89.5,num=180) } )
        else:
            DS_read = xr.open_dataset(infile,decode_times=False)

            DS_read['time'] = time[omip]

        tmp = DS_read[var].sel(time=slice(1980,2009)).mean(dim='time')

        if model == 'NorESM-BLOM':
            tmp = tmp.transpose().interp(depth=lev33)
        if model == 'AWI-FESOM':
            tmp = tmp.transpose()
        if model == "MIROC-COCO4.9":
            tmp = tmp.sel(lat=slice(None, None, -1))
        if model == 'EC-Earth3-NEMO':
            tmp = tmp.transpose("basin","depth","lat")
        if model == 'GFDL-MOM':
            lattmp=tmp['lat'].values
            depnew=tmp['z_l'].values
            depnew[0] = 0.0
            tmp1 = tmp.values
            tmpgfdl = xr.DataArray(tmp1, dims = ('basin','depth','lat',), coords = {'depth': depnew, 'lat': lattmp } )
            tmp = tmpgfdl.interp(depth=lev33)

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

fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

# [left, bottom, width, height]
axes0 = np.array( [ [0.05, 0.75, 0.05, 0.17],
                    [0.11, 0.75, 0.08, 0.17],
                    [0.20, 0.75, 0.04, 0.17],
                    [0.25, 0.75, 0.06, 0.17], ])
ax = [ [ plt.axes(axes0[0]),
         plt.axes(axes0[1]),
         plt.axes(axes0[2]),
         plt.axes(axes0[3]), ],
       [ plt.axes(axes0[0]+np.array([0.30,0,0,0])),
         plt.axes(axes0[1]+np.array([0.30,0,0,0])),
         plt.axes(axes0[2]+np.array([0.30,0,0,0])),
         plt.axes(axes0[3]+np.array([0.30,0,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.60,0,0,0])),
         plt.axes(axes0[1]+np.array([0.60,0,0,0])),
         plt.axes(axes0[2]+np.array([0.60,0,0,0])),
         plt.axes(axes0[3]+np.array([0.60,0,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.23,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.23,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.23,0,0])),
         plt.axes(axes0[3]+np.array([0,-0.23,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.30,-0.23,0,0])),
         plt.axes(axes0[1]+np.array([0.30,-0.23,0,0])),
         plt.axes(axes0[2]+np.array([0.30,-0.23,0,0])),
         plt.axes(axes0[3]+np.array([0.30,-0.23,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.60,-0.23,0,0])),
         plt.axes(axes0[1]+np.array([0.60,-0.23,0,0])),
         plt.axes(axes0[2]+np.array([0.60,-0.23,0,0])),
         plt.axes(axes0[3]+np.array([0.60,-0.23,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.46,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.46,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.46,0,0])),
         plt.axes(axes0[3]+np.array([0,-0.46,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.30,-0.46,0,0])),
         plt.axes(axes0[1]+np.array([0.30,-0.46,0,0])),
         plt.axes(axes0[2]+np.array([0.30,-0.46,0,0])),
         plt.axes(axes0[3]+np.array([0.30,-0.46,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.60,-0.46,0,0])),
         plt.axes(axes0[1]+np.array([0.60,-0.46,0,0])),
         plt.axes(axes0[2]+np.array([0.60,-0.46,0,0])),
         plt.axes(axes0[3]+np.array([0.60,-0.46,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.69,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.69,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.69,0,0])),
         plt.axes(axes0[3]+np.array([0,-0.69,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.30,-0.69,0,0])),
         plt.axes(axes0[1]+np.array([0.30,-0.69,0,0])),
         plt.axes(axes0[2]+np.array([0.30,-0.69,0,0])),
         plt.axes(axes0[3]+np.array([0.30,-0.69,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.60,-0.69,0,0])),
         plt.axes(axes0[1]+np.array([0.60,-0.69,0,0])),
         plt.axes(axes0[2]+np.array([0.60,-0.69,0,0])),
         plt.axes(axes0[3]+np.array([0.60,-0.69,0,0])), ],]
# [left, bottom, width, height]
ax_cbar = plt.axes([0.93,0.15,0.015,0.7])

model_title_locx = [0.17,0.48,0.79]
model_title_locy = [0.94,0.71,0.48,0.25]

bounds1 = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0]
bounds2 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
bounds3 = np.arange(-1,30.1,1)
ticks_bounds3 = [0, 5, 10, 15, 20, 25, 30] 

cmap = [ 'RdBu_r', 'RdBu_r', 'bwr', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]
outfile = './fig/ZMT_bias_' + item[nv_out] + '.png'

dict_rmse={}
rmse=np.array(np.empty((4)),dtype=np.float64)

# MMM

if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
    bounds = bounds1
    ticks_bounds = bounds1
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    ticks_bounds = bounds2
else:
    bounds = bounds3
    ticks_bounds = ticks_bounds3

da = DS[item[nv_out]].mean(dim='model',skipna=False)
for m in range(4):
    tmp = da.isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1]))
    nz, ny = tmp.shape
    dztmp = np.tile(dz33,ny).reshape(ny,nz).T
    msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
    datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
    tmp1 = (datmp * datmp * msktmp * dztmp).sum()
    tmp2 = (msktmp * dztmp).sum()
    rmse[m] = np.sqrt(tmp1/tmp2)
    title_bas[m] = basin_name[m][0:3] + ' ' + '{:.2f}'.format(rmse[m]) + '$^{\circ}$C'
    print(title_bas[m])
    if (m == 0):
        da.isel(basin=m).plot(ax=ax[11][m],cmap=cmap[nv_out],
                              levels=bounds,
                              extend='both',
                              cbar_kwargs={'orientation': 'vertical',
                                           'spacing':'uniform',
                                           'ticks': ticks_bounds,},
                              cbar_ax=ax_cbar,
                              add_labels=False,add_colorbar=True)
    else:
        da.isel(basin=m).plot(ax=ax[11][m],cmap=cmap[nv_out],
                              levels=bounds,
                              extend='both',
                              add_labels=False,add_colorbar=False)

    ax[11][m].set_title(title_bas[m],{'fontsize':6,'verticalalignment':'top'})
    ax[11][m].tick_params(labelsize=7)
    ax[11][m].invert_yaxis()
    ax[11][m].set_xlim(xlim[m][0],xlim[m][1])
    ax[11][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
    ax[11][m].set_facecolor('lightgray')
for m in range(1,4):
    ax[11][m].tick_params(axis='y',labelleft=False)

dict_rmse['MMM']=rmse.copy()

nmodel = 0
for model in model_list[0]:
    
    if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
        bounds = bounds1
        ticks_bounds = bounds1
    elif item[nv_out] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
    if item[nv_out] == 'obs':
        da = DS[item[nv_out]]
    else:
        da = DS[item[nv_out]].isel(model=nmodel)
    for m in range(4):
        tmp = da.isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1]))
        nz, ny = tmp.shape
        dztmp = np.tile(dz33,ny).reshape(ny,nz).T
        msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
        tmp1 = (datmp * datmp * msktmp * dztmp).sum()
        tmp2 = (msktmp * dztmp).sum()
        rmse[m] = np.sqrt(tmp1/tmp2)
        title_bas[m] = basin_name[m][0:3] + ' ' + '{:.2f}'.format(rmse[m]) + '$^{\circ}$C'
        print(title_bas[m])
        da.isel(basin=m).plot(ax=ax[nmodel][m],cmap=cmap[nv_out],
                              levels=bounds,
                              extend='both',
                              add_labels=False,add_colorbar=False)

        ax[nmodel][m].set_title(title_bas[m],{'fontsize':6,'verticalalignment':'top'})
        ax[nmodel][m].tick_params(labelsize=7)
        ax[nmodel][m].invert_yaxis()
        ax[nmodel][m].set_xlim(xlim[m][0],xlim[m][1])
        ax[nmodel][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
        ax[nmodel][m].set_facecolor('lightgray')
    for m in range(1,4):
        ax[nmodel][m].tick_params(axis='y',labelleft=False)

    dict_rmse[model]=rmse.copy()
    nmodel += 1
        
nmodel = 0
for model in model_list[0]:
    ny = int(nmodel / 3)
    nx = nmodel - ny * 3
    print(nx,ny)
    fig.text(model_title_locx[nx],model_title_locy[ny],model,fontsize=10,horizontalalignment='center',verticalalignment='center')
    nmodel += 1
    
fig.text(model_title_locx[2],model_title_locy[3],'MMM',fontsize=10,horizontalalignment='center',verticalalignment='center')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

for m in range(4):
  index_bas[m]='OMIP'+str(omip_out)+'_rmse_'+ str(basin_name[m])

summary=pd.DataFrame(dict_rmse,index=index_bas)
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/ZMT_bias_OMIP' + str(omip_out) + '.csv')

if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
