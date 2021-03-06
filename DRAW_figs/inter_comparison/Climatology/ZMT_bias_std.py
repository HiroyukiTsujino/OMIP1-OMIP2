# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()

factor_5ptail = 1.64  # 5-95%

metainfo = [ json.load(open("./json/zmt_omip1.json")),
             json.load(open("./json/zmt_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

#basin_name = [ 'Southern Ocean', 'Atlantic Ocean', 
#          'Indian Ocean', 'Pacific Ocean' ]
basin_name = [ 'Southern', 'Atlantic', 'Indian', 'Pacific' ]
title_bas = ['a', 'b', 'c', 'd']

title2 = [ '(a) Ensemble bias (OMIP1 - WOA13v2)', '(b) Ensemble bias (OMIP2 - WOA13v2)',
           '(c) Ensemble std (OMIP2 - WOA13v2)', '(d) Ensemble std (OMIP2 - WOA13v2)',
           '(e) OMIP2 - OMIP1', '(f) WOA13v2' ]

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

if (sys.argv[1] == 'MMM'):
    outfile = './fig/ZMT_bias_MMM'
    suptitle = 'Multi Model Mean' + ' Zonal mean temperature (ave. from 1980 to 2009)'

else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/ZMT_bias_' + sys.argv[1]
    suptitle = sys.argv[1] + ' Zonal mean temperature (ave. from 1980 to 2009)'

print("Drawing " + suptitle)

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]


#J データ読込・平均
print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_th_basin.1000'
da_ref = xr.open_dataset( reffile, decode_times=False)["thetao"].mean(dim='time')
da_ref = da_ref.assign_coords(basin=[0,1,2,3])

# uncertainty of difference between omip-1 and omip-2

stdfile = '../analysis/STDs/ZMT_omip1-omip2_stats.nc'
DS_stats = xr.open_dataset( stdfile )

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
            lattmp = tmp['lat'].values
            depnew = tmp['z_l'].values
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
axes0 = np.array( [ [0.07, 0.66, 0.06, 0.24],
                    [0.14, 0.66, 0.12, 0.24],
                    [0.27, 0.66, 0.06, 0.24],
                    [0.34, 0.66, 0.09, 0.24], ])
ax = [ [ plt.axes(axes0[0]),
         plt.axes(axes0[1]),
         plt.axes(axes0[2]),
         plt.axes(axes0[3]), ],
       [ plt.axes(axes0[0]+np.array([0.48,0,0,0])),
         plt.axes(axes0[1]+np.array([0.48,0,0,0])),
         plt.axes(axes0[2]+np.array([0.48,0,0,0])),
         plt.axes(axes0[3]+np.array([0.48,0,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.31,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.31,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.31,0,0])),
         plt.axes(axes0[3]+np.array([0,-0.31,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.48,-0.31,0,0])),
         plt.axes(axes0[1]+np.array([0.48,-0.31,0,0])),
         plt.axes(axes0[2]+np.array([0.48,-0.31,0,0])),
         plt.axes(axes0[3]+np.array([0.48,-0.31,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.62,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.62,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.62,0,0])),
         plt.axes(axes0[3]+np.array([0,-0.62,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.48,-0.62,0,0])),
         plt.axes(axes0[1]+np.array([0.48,-0.62,0,0])),
         plt.axes(axes0[2]+np.array([0.48,-0.62,0,0])),
         plt.axes(axes0[3]+np.array([0.48,-0.62,0,0])), ] ]

# [left, bottom, width, height]
ax_cbar = [ plt.axes([0.45, 0.66, 0.012, 0.25]),
            plt.axes([0.93, 0.66, 0.012, 0.25]),
            plt.axes([0.45, 0.35, 0.012, 0.25]),
            plt.axes([0.93, 0.35, 0.012, 0.25]),
            plt.axes([0.45, 0.04, 0.012, 0.25]),
            plt.axes([0.93, 0.04, 0.012, 0.25]) ]

bounds1 = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0]
bounds2 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
bounds3 = np.arange(-1,30.1,1)
ticks_bounds3 = [0, 5, 10, 15, 20, 25, 30] 
bounds4 = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
ticks_bounds4 = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

#cmap = [ 'RdBu_r', 'RdBu_r', 'viridis', 'viridis', 'RdBu_r', 'RdYlBu_r' ]
cmap = [ 'RdBu_r', 'RdBu_r', 'terrain', 'terrain', 'bwr', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip1std', 'omip2std', 'omip2-1', 'obs' ]

ddof_dic={'ddof' : 0}

for panel in range(6):
    if (item[panel] == 'omip1bias' or item[panel] == 'omip2bias'):
        bounds = bounds1
        ticks_bounds = bounds1
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
            title_bas[m] = basin_name[m][0:3] + ' rmse=' + '{:.2f}'.format(rmse) + r'$^{\circ}$C'
            print(title_bas[m])
    elif (item[panel] == 'omip1std'):
        bounds = bounds4
        ticks_bounds = bounds4
        da = DS['omip1bias'].std(dim='model', skipna=False, **ddof_dic)
        davar = DS['omip1bias'].var(dim='model', skipna=False, **ddof_dic)
        for m in range(4):
            tmp = davar.isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1]))
            nz, ny = tmp.shape
            dztmp = np.tile(dz33,ny).reshape(ny,nz).T
            msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
            datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
            tmp1 = (datmp * msktmp * dztmp).sum()
            tmp2 = (msktmp * dztmp).sum()
            rmse = np.sqrt(tmp1/tmp2)
            z = np.abs(DS['omip1bias'].isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1])).mean(dim='model',skipna=False)) \
               - 2.0 * DS['omip1bias'].isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1])).std(dim='model', skipna=False, **ddof_dic)
            z = np.where( z > 0, 1, 0 )
            tmp3 = (msktmp * dztmp * z).sum()
            failcapt=tmp3/tmp2*100
            title_bas[m] = basin_name[m][0:3] + r' 2$\bar{\sigma}$=' + '{:.2f}'.format(2*rmse) + r'$^{\circ}$C, ' + '{:.1f}'.format(failcapt) + '%'
            print(title_bas[m])
    elif (item[panel] == 'omip2std'):
        bounds = bounds4
        ticks_bounds = bounds4
        da = DS['omip2bias'].std(dim='model',skipna=False, **ddof_dic)
        davar = DS['omip2bias'].var(dim='model', skipna=False, **ddof_dic)
        for m in range(4):
            tmp = davar.isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1]))
            nz, ny = tmp.shape
            dztmp = np.tile(dz33,ny).reshape(ny,nz).T
            msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
            datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
            tmp1 = (datmp * msktmp * dztmp).sum()
            tmp2 = (msktmp * dztmp).sum()
            rmse = np.sqrt(tmp1/tmp2)
            z = np.abs(DS['omip2bias'].isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1])).mean(dim='model',skipna=False)) \
               - 2.0 * DS['omip2bias'].isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1])).std(dim='model', skipna=False, **ddof_dic)
            z = np.where( z > 0, 1, 0 )
            tmp3 = (msktmp * dztmp * z).sum()
            failcapt=tmp3/tmp2*100
            title_bas[m] = basin_name[m][0:3] + r' 2$\bar{\sigma}$=' + '{:.2f}'.format(2*rmse) + r'$^{\circ}$C, ' + '{:.1f}'.format(failcapt) + '%'
            print(title_bas[m])
    elif (item[panel] == 'omip2-1'):
        bounds = bounds2
        ticks_bounds = bounds2
        da = DS[item[panel]].mean(dim='model',skipna=False)
        for m in range(4):
            tmp = da.isel(basin=m).sel(lat=slice(xlim[m][0],xlim[m][1]))
            nz, ny = tmp.shape
            dztmp = np.tile(dz33,ny).reshape(ny,nz).T
            msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
            datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
            tmp1 = (datmp * datmp * msktmp * dztmp).sum()
            tmp2 = (msktmp * dztmp).sum()
            rmsd = np.sqrt(tmp1/tmp2)
            title_bas[m] = basin_name[m][0:3] + ' rmsd=' + '{:.2f}'.format(rmsd) + r'$^{\circ}$C'
            print(title_bas[m])
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
        da = DS[item[panel]]
        title_bas[0:4]= basin_name[0:4]

    for m in range(4):
        da.isel(basin=m).plot(ax=ax[panel][m],cmap=cmap[panel],
                              levels=bounds,
                              extend='both',
                              cbar_kwargs={'orientation': 'vertical',
#                                           'spacing':'proportional',
                                           'spacing':'uniform',
                                           'ticks': ticks_bounds,},
                              cbar_ax=ax_cbar[panel],
                              add_labels=False,add_colorbar=True)

        if (panel == 2):
            x = da["lat"].values
            y = da["depth"].values
            z = np.abs(DS['omip1bias'].isel(basin=m).mean(dim='model',skipna=False)) \
               - 2.0 * DS['omip1bias'].isel(basin=m).std(dim='model', skipna=False, **ddof_dic)
            z = np.where( z > 0, 1, np.nan )
            ax[panel][m].contourf(x,y,z,hatches=['xxxxxxx'],colors='none')
            mpl.rcParams['hatch.color'] = 'red'
            mpl.rcParams['hatch.linewidth'] = 0.5

        if (panel == 3):
            x = da["lat"].values
            y = da["depth"].values
            z = np.abs(DS['omip2bias'].isel(basin=m).mean(dim='model',skipna=False)) \
               - 2.0 * DS['omip2bias'].isel(basin=m).std(dim='model', skipna=False, **ddof_dic)
            z = np.where( z > 0, 1, np.nan )
            ax[panel][m].contourf(x,y,z,hatches=['xxxxxxx'],colors='none')
            mpl.rcParams['hatch.color'] = 'red'
            mpl.rcParams['hatch.linewidth'] = 0.5

        if (panel == 4):
            x = DS_stats["lat"].values
            y = DS_stats["depth"].values
            z = np.abs(DS_stats["mean"].isel(basin=m)) - factor_5ptail * DS_stats["std"].isel(basin=m)
            z = np.where( z > 0, 1, np.nan )
            ax[panel][m].contourf(x,y,z,hatches=['xxxxxxx'],colors='none')
            mpl.rcParams['hatch.color'] = 'lime'
            mpl.rcParams['hatch.linewidth'] = 0.5


        ax[panel][m].set_title(title_bas[m],{'fontsize':6, 'verticalalignment':'top'})
        ax[panel][m].tick_params(labelsize=9)
        ax[panel][m].invert_yaxis()
        ax[panel][m].set_xlim(xlim[m][0],xlim[m][1])
        ax[panel][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
        ax[panel][m].set_facecolor('lightgray')


    for m in range(1,4):
        ax[panel][m].tick_params(axis='y',labelleft=False)


fig.text(0.25,0.925,title2[0],fontsize=12,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.75,0.925,title2[1],fontsize=12,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.25,0.615,title2[2],fontsize=12,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.75,0.615,title2[3],fontsize=12,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.25,0.305,title2[4],fontsize=12,
         horizontalalignment='center',verticalalignment='center')
fig.text(0.75,0.305,title2[5],fontsize=12,
         horizontalalignment='center',verticalalignment='center')

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
