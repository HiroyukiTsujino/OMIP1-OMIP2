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

title = [ 'Southern Ocean', 'Atlantic Ocean', 'Indo-Pacific Ocean' ]
xlim  = [ [-90, -28], [-30, 90], [-30, 70] ]

factor_5ptail = 1.64  # 5-95%

metainfo = [ json.load(open("./json/moc_omip1.json")),
             json.load(open("./json/moc_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title2 = [ '(a) Ensemble mean OMIP1', '(b) Ensemble mean OMIP2',
           '(c) Ensemble std OMIP1', '(d) Ensemble std OMIP2',
           '(e) OMIP2 - OMIP1', '(f) OMIP2 - OMIP1 (0 - 500 m)' ]

lev33 = np.array([ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ])

latwoa = np.linspace(-89.5,89.5,num=180)


if (sys.argv[1] == 'MMM'):
    outfile = './fig/MOC_MMM'
    suptitle = 'Multi Model Mean' + ' Meridional Overturning Circulation (ave. from 1980 to 2009)'

else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/MOC_' + sys.argv[1]
    suptitle = sys.argv[1] + ' Meridional Overturning Circulation (ave. from 1980 to 2009)'

# uncertainty of difference between omip-1 and omip-2

stdfile = '../analysis/STDs/ZMS_omip1-omip2_stats.nc'
DS_stats = xr.open_dataset( stdfile )

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
        elif ( model == 'Kiel-NEMO' or model == 'AWI-FESOM' or model == 'EC-Earth3-NEMO' ):
            time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]
        elif ( model == 'MIROC-COCO4.9'):
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
            if model == 'CMCC-NEMO':
                lattmp = DS_read['lat'].values
                ny = len(lattmp)
                lattmp[ny-1] = lattmp[ny-2] + (lattmp[ny-2] - lattmp[ny-3])
                print(ny,lattmp)
                DS_read['lat'] = lattmp

        tmp = DS_read[var].sel(time=slice(1980,2009)).mean(dim='time')

        if model == 'GFDL-MOM':
            tmp = tmp.interp(z=lev33)
            tmp = tmp.interp(y=latwoa)

        if model == 'NorESM-BLOM':
            tmp = tmp.interp(depth=lev33)
            tmp = tmp.interp(lat=latwoa)

        if model == 'CMCC-NEMO':
            tmp = tmp.interp(depth=lev33)
            tmp = tmp.interp(lat=latwoa)

        if model == "CAS-LICOM3":
            tmp = tmp.sel(lat=slice(None, None, -1))

        if model == 'CESM-POP':
            lev33cm = np.empty( 33 )
            lev33cm[:]= lev33[:] * 1.e2
            tmp = tmp.interp(lev=lev33cm)
            tmp = tmp.interp(lat=latwoa)

        if model == 'AWI-FESOM':
            tmp = tmp.transpose("basin","depth_coord","lat")

        if (model == 'NorESM-BLOM'):
            d[nmodel,0,:,:] = tmp.values[3,:,:] * factor
            d[nmodel,1,:,:] = tmp.values[1,:,:] * factor
            d[nmodel,2,:,:] = tmp.values[2,:,:] * factor
        elif (model == 'CESM-POP'):
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
print("Drawing " + suptitle)

fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

# [left, bottom, width, height]
axes0 = np.array( [ [0.07, 0.66, 0.07, 0.24],
                    [0.15, 0.66, 0.15, 0.24],
                    [0.31, 0.66, 0.13, 0.24], ])

ax = [ [ plt.axes(axes0[0]),
         plt.axes(axes0[1]),
         plt.axes(axes0[2]), ],
       [ plt.axes(axes0[0]+np.array([0.48,0,0,0])),
         plt.axes(axes0[1]+np.array([0.48,0,0,0])),
         plt.axes(axes0[2]+np.array([0.48,0,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.31,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.31,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.31,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.48,-0.31,0,0])),
         plt.axes(axes0[1]+np.array([0.48,-0.31,0,0])),
         plt.axes(axes0[2]+np.array([0.48,-0.31,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0,-0.62,0,0])),
         plt.axes(axes0[1]+np.array([0,-0.62,0,0])),
         plt.axes(axes0[2]+np.array([0,-0.62,0,0])), ],
       [ plt.axes(axes0[0]+np.array([0.48,-0.62,0,0])),
         plt.axes(axes0[1]+np.array([0.48,-0.62,0,0])),
         plt.axes(axes0[2]+np.array([0.48,-0.62,0,0])), ] ]

ax_cbar = [ plt.axes([0.45, 0.66, 0.012, 0.25]),
            plt.axes([0.93, 0.66, 0.012, 0.25]),
            plt.axes([0.45, 0.35, 0.012, 0.25]),
            plt.axes([0.93, 0.35, 0.012, 0.25]),
            plt.axes([0.45, 0.04, 0.012, 0.25]),
            plt.axes([0.93, 0.04, 0.012, 0.25]) ]

bounds1 = np.linspace(-30,30,31)
tick_bounds1 = np.linspace(-30,30,16)
bounds2 = np.array([-10,-8,-6,-5,-4,-3,-2,-1,-0.5, 0, 0.5, 1,2,3,4,5,6,8,10])
tick_bounds2 = np.array([-10,-8,-6,-5,-4,-3,-2,-1,-0.5, 0, 0.5, 1,2,3,4,5,6,8,10])
bounds3 = np.linspace(0,8,9)

cmap = [ 'RdBu_r', 'RdBu_r', 'viridis', 'viridis', 'coolwarm', 'coolwarm' ]

item = [ 'omip1', 'omip2', 'omip1std', 'omip2std', 'omip2-1', 'omip2-1' ]

mpl.rcParams['hatch.color'] = 'darkgreen'
mpl.rcParams['hatch.linewidth'] = 0.5

for panel in range(6):
    if item[panel] == 'omip1' or item[panel] == 'omip2':
        bounds = bounds1
        ticks_bounds = tick_bounds1
        da = DS[item[panel]].mean(dim='model',skipna=False)
    elif item[panel] == 'omip1std':
        bounds = bounds3
        ticks_bounds = bounds3
        da = DS['omip1'].std(dim='model',skipna=False)
    elif item[panel] == 'omip2std':
        bounds = bounds3
        ticks_bounds = bounds3
        da = DS['omip2'].std(dim='model',skipna=False)
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = tick_bounds2
        if (panel == 5):
            da = DS[item[panel]].sel(depth=slice(0,500)).mean(dim='model',skipna=False)
        else:
            da = DS[item[panel]].mean(dim='model',skipna=False)

    for m in range(3):
        da.isel(basin=m).plot(ax=ax[panel][m],cmap=cmap[panel],
                              levels=bounds,
                              extend='both',
                              cbar_kwargs={'orientation': 'vertical',
                                           'spacing':'uniform',
                                           'ticks': ticks_bounds,},
                              cbar_ax=ax_cbar[panel],
                              add_labels=False,add_colorbar=True)
        if ((item[panel] != 'omip1std') and (item[panel] != 'omip2std')):
            da.isel(basin=m).plot.contour(ax=ax[panel][m],
                                          colors='black',
                                          levels=bounds,
                                          add_labels=False,
                                          linewidths=1.0)
        #if (panel == 4):
        #    x = DS_stats["lat"].values
        #    y = DS_stats["depth"].values
        #    z = np.abs(DS_stats["mean"].isel(basin=m)) - factor_5ptail * DS_stats["std"].isel(basin=m)
        #    z = np.where( z > 0, 1, np.nan )
        #    ax[panel][m].contourf(x,y,z,hatches=['xxxxxxx'],colors='none')

        ax[panel][m].set_title(title[m],{'fontsize':8, 'verticalalignment':'top'})
        ax[panel][m].tick_params(labelsize=9)
        ax[panel][m].invert_yaxis()
        ax[panel][m].set_xlim(xlim[m][0],xlim[m][1])
        ax[panel][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
        ax[panel][m].set_facecolor('lightgray')
    for m in range(1,3):
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

print("figure is saved to " + outpng + " and " + outpdf)

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
