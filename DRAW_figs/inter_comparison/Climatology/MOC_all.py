# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4


title = [ 'Southern Ocean', 'Atlantic Ocean', 'Indo-Pacific Ocean' ]
xlim  = [ [-90, -28], [-30, 90], [-30, 70] ]

metainfo = [ json.load(open("./json/moc_omip1.json")),
             json.load(open("./json/moc_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title2 = [ '(a) OMIP1', '(b) OMIP2', '(c) OMIP2 - OMIP1', '(d) OMIP2 - OMIP1 (0 - 500 m)' ]

lev33 = np.array([ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ])

latwoa = np.linspace(-89.5,89.5,num=180)


if len(sys.argv) == 1:
    outfile = './fig/MOC_MMM.png'
    suptitle = 'Multi Model Mean' + ' Meridional Overturning Circulation (ave. from 1980 to 2009)'

else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/MOC_' + sys.argv[1] + '.png'
    suptitle = sys.argv[1] + ' Meridional Overturning Circulation (ave. from 1980 to 2009)'

print("Drawing " + suptitle)

data = []
modnam = []
nummodel = []

for omip in range(2):
    d = np.empty( (len(model_list[omip]),3,33,180) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    namtmp = []
    nmodel = 0
    for model in model_list[omip]:

        path = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        var = metainfo[omip][model]['name']
        factor = float(metainfo[omip][model]['factor'])
        infile = path + '/' + fname

        namtmp +=[model]

        #J 時刻情報 (各モデルの時刻情報を上書きする)
        if ( model == 'FSU-HYCOM'):
            time = [ np.linspace(1638,2009,372), np.linspace(1668,2018,351) ]
        elif ( model == 'GFDL-MOM'):
            time = [ np.linspace(1648,2009,362), np.linspace(1656,2018,363) ]
        elif ( model == 'Kiel-NEMO' or model == 'AWI-FESOM' ):
            time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]
        elif ( model == 'MIROC-COCO4-9'):
            time = [ np.linspace(1700,2009,310), np.linspace(1653,2018,366) ]
        else:
            time = [ np.linspace(1638,2009,372), np.linspace(1653,2018,366) ]
            
        if ( model == 'Kiel-NEMO' ):

            infile_glb = infile + '_global.nc'
            infile_atl = infile + '_atl.nc'
            infile_indpac = infile + '_indpac.nc'

            ncglb = netCDF4.Dataset(infile_glb,'r')
            msftmyz_glb = ncglb.variables['msftmyz_global'][:,:,:]
            ncglb.close()
            msftmyz_glb = np.where(msftmyz_glb > 9.9e36, np.NaN, msftmyz_glb)
            
            ncatl = netCDF4.Dataset(infile_atl,'r')
            msftmyz_atl = ncatl.variables['msftmyz_atl'][:,:,:]
            ncatl.close()
            msftmyz_atl = np.where(msftmyz_atl > 9.9e36, np.NaN, msftmyz_atl)

            ncind = netCDF4.Dataset(infile_indpac,'r')
            msftmyz_indpac = ncind.variables['msftmyz_indpac'][:,:,:]
            ncind.close()
            msftmyz_indpac = np.where(msftmyz_indpac > 9.9e36, np.NaN, msftmyz_indpac)

            if ( omip == 0 ):
                msftmyz_all = np.array(np.zeros((62,3,33,180)),dtype=np.float32)
                msftmyz_all[0:62,0,0:33,0:180] = msftmyz_glb[0:62,0:33,0:180]
                msftmyz_all[0:62,1,0:33,0:180] = msftmyz_atl[0:62,0:33,0:180]
                msftmyz_all[0:62,2,0:33,0:180] = msftmyz_indpac[0:62,0:33,0:180]
                #print(msftmyz_glb[0:62,0,90])
                DS_read = xr.Dataset({'msftmyz': (['time','basin','depth','lat'], msftmyz_all)},
                                 coords = {'time' : time[omip], 'depth': lev33, 'lat': latwoa } )

            else:
                msftmyz_all = np.array(np.zeros((61,3,33,180)),dtype=np.float32)
                msftmyz_all[0:61,0,0:33,0:180] = msftmyz_glb[0:61,0:33,0:180]
                msftmyz_all[0:61,1,0:33,0:180] = msftmyz_atl[0:61,0:33,0:180]
                msftmyz_all[0:61,2,0:33,0:180] = msftmyz_indpac[0:61,0:33,0:180]
                DS_read = xr.Dataset({'msftmyz': (['time','basin','depth','lat'], msftmyz_all)},
                                 coords = {'time' : time[omip], 'depth': lev33, 'lat': latwoa } )

        else:
            DS_read = xr.open_dataset(infile,decode_times=False)
            DS_read['time'] = time[omip]

        tmp = DS_read[var].sel(time=slice(1980,2009)).mean(dim='time')

        if model == 'GFDL-MOM':
            tmp = tmp.interp(z=lev33)
            tmp = tmp.interp(y=latwoa)

        if model == 'NorESM-O-CICE':
            tmp = tmp.interp(depth=lev33)
            tmp = tmp.interp(lat=latwoa)

        if model == "CAS-LICOM3":
            tmp = tmp.sel(lat=slice(None, None, -1))

        if model == 'NCAR-POP':
            lev33cm = np.empty( 33 )
            lev33cm[:]= lev33[:] * 1.e2
            tmp = tmp.interp(lev=lev33cm)
            tmp = tmp.interp(lat=latwoa)

        if model == 'AWI-FESOM':
            tmp = tmp.transpose("basin","depth_coord","lat")

        if (model == 'NorESM-O-CICE'):
            d[nmodel,0,:,:] = tmp.values[3,:,:] * factor
            d[nmodel,1,:,:] = tmp.values[1,:,:] * factor
            d[nmodel,2,:,:] = tmp.values[2,:,:] * factor
        elif (model == 'NCAR-POP'):
            d[nmodel,0,:,:] = tmp.values[2,:,:] * factor
            d[nmodel,1,:,:] = tmp.values[0,:,:] * factor
            d[nmodel,2,:,:] = tmp.values[1,:,:] * factor
        else:  
            d[nmodel] = tmp.values * factor

        nmodel += 1

    data += [d]
    modnam += [namtmp]
    nummodel += [nmodel]

DS = xr.Dataset({'omip1': (['model','basin','depth','lat'], data[0]),
                 'omip2': (['model','basin','depth','lat'], data[1]),
                 'omip2-1': (['model','basin','depth','lat'], data[1]-data[0]),},
                coords = {'depth': lev33, 'lat': latwoa } )

for omip in range(2):
    for nm in range(nummodel[omip]):
        print(modnam[0][nm]+'-omip'+str(omip+1),DS['omip'+str(omip+1)].sel(model=nm,basin=0,depth=slice(2000,6500)).interp(lat=-30).min(dim="depth").values)


#J 描画

fig = plt.figure(figsize=(16,12))
fig.suptitle( suptitle, fontsize=20 )

axes0 = np.array( [ [0.07, 0.6, 0.07, 0.3],
                    [0.15, 0.6, 0.17, 0.3],
                    [0.33, 0.6, 0.15, 0.3],
                    [0.07, 0.55, 0.40, 0.015], ])
ax = [ [ plt.axes(axes0[0]),
         plt.axes(axes0[1]),
         plt.axes(axes0[2]), ],
       [ plt.axes(axes0[0]+np.array([0.5,0,0,0])),
         plt.axes(axes0[1]+np.array([0.5,0,0,0])),
         plt.axes(axes0[2]+np.array([0.5,0,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.5,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.5,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.5,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.5,-0.5,0,0])),
         plt.axes(axes0[1]+np.array([0.5,-0.5,0,0])),
         plt.axes(axes0[2]+np.array([0.5,-0.5,0,0])), ] ]
ax_cbar = [ plt.axes(axes0[3]),
            plt.axes(axes0[3]+np.array([0.5,0,0,0])),
            plt.axes(axes0[3]+np.array([0,-0.5,0,0])),
            plt.axes(axes0[3]+np.array([0.5,-0.5,0,0])), ]

bounds1 = np.linspace(-30,30,31)
bounds2 = np.linspace(-8,8,17)

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r' ]

item = [ 'omip1', 'omip2', 'omip2-1', 'omip2-1' ]

for panel in range(4):
    if item[panel] == 'omip1' or item[panel] == 'omip2':
        bounds = bounds1
        ticks_bounds = bounds1
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2

    if (panel == 3):
        da = DS[item[panel]].sel(depth=slice(0,500)).mean(dim='model',skipna=False)
    else:
        da = DS[item[panel]].mean(dim='model',skipna=False)

    for m in range(3):
        da.isel(basin=m).plot(ax=ax[panel][m],cmap=cmap[panel],
                              levels=bounds,
                              extend='both',
                              cbar_kwargs={'orientation': 'horizontal',
                                           'spacing':'uniform',
                                           'ticks': ticks_bounds,},
                              cbar_ax=ax_cbar[panel],
                              add_labels=False,add_colorbar=True)
        da.isel(basin=m).plot.contour(ax=ax[panel][m],
                                      colors='black',
                                      levels=bounds,
                                      add_labels=False,
                                      linewidths=1.0)
        ax[panel][m].set_title(title[m])
        ax[panel][m].invert_yaxis()
        ax[panel][m].set_xlim(xlim[m][0],xlim[m][1])
        ax[panel][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
        ax[panel][m].set_facecolor('lightgray')
    for m in range(1,3):
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
