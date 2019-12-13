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

title = [ 'Southern Ocean', 'Atlantic Ocean', 
          'Indian Ocean', 'Pacific Ocean' ]
title2 = [ '(a) Ensemble bias (OMIP1 - WOA13v2)', '(b) Ensemble bias (OMIP2 - WOA13v2)',
           '(c) Ensemble std (OMIP2 - WOA13v2)', '(d) Ensemble std (OMIP2 - WOA13v2)',
           '(e) OMIP2 - OMIP1', '(f) WOA13v2' ]

xlim = [ [-90, -30], [-30, 90], [-30, 30], [-30, 60] ]
lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

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
bounds4 = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]
ticks_bounds4 = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]

cmap = [ 'RdBu_r', 'RdBu_r', 'viridis', 'viridis', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip1std', 'omip2std', 'omip2-1', 'obs' ]

mpl.rcParams['hatch.color'] = 'limegreen'
mpl.rcParams['hatch.linewidth'] = 0.5

for panel in range(6):
    if (item[panel] == 'omip1bias' or item[panel] == 'omip2bias'):
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS[item[panel]].mean(dim='model',skipna=False)
    elif (item[panel] == 'omip1std'):
        bounds = bounds4
        ticks_bounds = bounds4
        da = DS['omip1bias'].std(dim='model',skipna=False)
    elif (item[panel] == 'omip2std'):
        bounds = bounds4
        ticks_bounds = bounds4
        da = DS['omip2bias'].std(dim='model',skipna=False)
    elif (item[panel] == 'omip2-1'):
        bounds = bounds2
        ticks_bounds = bounds2
        da = DS[item[panel]].mean(dim='model',skipna=False)
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
        da = DS[item[panel]]

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
        if (panel == 4):
            x = DS_stats["lat"].values
            y = DS_stats["depth"].values
            z = np.abs(DS_stats["mean"].isel(basin=m)) - factor_5ptail * DS_stats["std"].isel(basin=m)
            z = np.where( z > 0, 1, np.nan )
            ax[panel][m].contourf(x,y,z,hatches=['xxxxxxx'],colors='none')

        ax[panel][m].set_title(title[m],{'fontsize':8, 'verticalalignment':'top'})
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

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
