# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4
from netCDF4 import Dataset, num2date


metainfo = [ json.load(open("./json/uo140w_omip1.json")),
             json.load(open("./json/uo140w_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ '(a) OMIP1', '(b) OMIP2', '(c) OMIP2 - OMIP1' ]

lattrop = np.linspace(-19.5,19.5,40)
lev14 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500. ]

if len(sys.argv) == 1:
    outfile = './fig/U140W.png'
    suptitle = 'Multi Model Mean' + ' Zonal velocity at eastern Tropical Pacific (ave. from 1980 to 2009)'
else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/U140W_' + sys.argv[1] + '.png'
    suptitle = sys.argv[1] + ' Zonal velocity at eastern Tropical Pacific (ave. from 1980 to 2009)'

print("Drawing "+suptitle)

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]


# There are no digital reference data at present.
#
#print( "Loading WOA13v2 data" )
#reffile = '../WOA/annual/woa13_decav_th_basin.1000'
#da_ref = xr.open_dataset( reffile, decode_times=False)["thetao"].mean(dim='time')
#da_ref = da_ref.assign_coords(basin=[0,1,2,3])
#

data = []
for omip in range(2):
    d = np.empty( (len(model_list[omip]),14,40) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        var = metainfo[omip][model]['name']
        infile = path + '/' + fname

        print (model)
        DS_read = xr.open_dataset(infile,decode_times=False)
        DS_read['time'] = time[omip]

        tmpm = DS_read[var].sel(time=slice(1980,2009)).mean(dim='time')

        if (model == 'MRI.COM'):
            tmp = tmpm
        if (model == 'FSU-HYCOM'):
            tmp = tmpm[:,1:41]
        if (model == 'CAS-LICOM3'):
            tmp1 = tmpm.sel(lat=slice(None, None, -1))
            tmp = tmp1[:,1:41]
        if model == 'AWI-FESOM':
            tmp = tmpm.transpose()
        if model == "MIROC-COCO4-9":
            tmp1 = tmpm[:,:,0]
            tmp = tmp1.sel(lat=slice(None, None, -1))
        if model == 'NCAR-POP':
            lattmp=tmpm['lat'].values
            latnew=lattmp[:,0]
            depnew=tmpm['lev'].values
            depnew=depnew*1.e-2
            depnew[0] = 0.0
            depnew[32] = 500.0
            tmp1 = tmpm[:,:,0].values
            tmpncar = xr.DataArray(tmp1, dims=('depth','lat'),
                                 coords={'depth':depnew, 'lat':latnew},
                                 name='zonal current at 140W')
            tmp2 = tmpncar.interp(depth=lev14)
            tmp = tmp2.interp(lat=lattrop)
        if model == 'NorESM-O-CICE':
            nc = netCDF4.Dataset(infile,'r')
            lat_ = nc.variables['lat'][:]
            depth_ = nc.variables['depth'][:]
            jm = len(lat_)
            km = len(depth_)
            nc.close()
            tmpnor = xr.DataArray(tmpm, dims=('depth','lat'),
                                 coords={'depth':depth_, 'lat':lat_},
                                 name='zonal current at 140W')
            tmp1 = tmpnor.interp(depth=lev14)
            tmp = tmp1.interp(lat=lattrop)
        if (model == 'BSC-NEMO'):
            tmp = tmpm[0:14,6:46,0]
        if (model == 'Kiel-NEMO'):
            tmp = tmpm[:,0:40,0]
        if (model == 'GFDL-MOM'):
            nc = netCDF4.Dataset(infile,'r')
            lat_ = nc.variables['yh'][:]
            depth_ = nc.variables['z_l'][:]
            jm = len(lat_)
            km = len(depth_)
            nc.close()
            tmpgfdl = xr.DataArray(tmpm, dims=('depth','lat'),
                                 coords={'depth':depth_, 'lat':lat_},
                                 name='zonal current at 140W')
            tmp1 = tmpgfdl.interp(depth=lev14)
            tmp = tmp1.interp(lat=lattrop)

        #print(tmp)
        d[nmodel] = tmp.values
        nmodel += 1

    data += [d]

DS = xr.Dataset({'omip1': (['model','depth','lat'], data[0]),
                 'omip2': (['model','depth','lat'], data[1]),
                 'omip2-1': (['model','depth','lat'], data[1]-data[0]), },
                coords = {'depth': lev14, 'lat': lattrop } )

#J 描画

fig = plt.figure(figsize=(16,12))
fig.suptitle( suptitle, fontsize=20 )

ax = [
    plt.subplot(2,2,1),
    plt.subplot(2,2,2),
    plt.subplot(2,2,3)
]

bounds1 = np.arange(-1.2,1.25,0.05)
tick_bounds1 = np.arange(-1.2,1.25,0.2)
bounds1_c = np.arange(-1.2,1.25,0.2)
#bounds1_cb = np.arange(-1.5,1.6,0.5)
bounds2 = np.arange(-0.5,0.55,0.05)
tick_bounds2 = np.arange(-0.5,0.6,0.1)
bounds2_c = np.arange(-0.5,0.55,0.05)

cmap = [ 'RdYlBu_r', 'RdYlBu_r', 'RdBu_r' ]

item = [ 'omip1', 'omip2', 'omip2-1' ]

for panel in range(3):
    if item[panel] == 'omip1' or item[panel] == 'omip2':
        bounds = bounds1
        boundsc = bounds1_c
        #boundscb = bounds1_cb
        ticks_bounds = tick_bounds1
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        boundsc = bounds2_c
        ticks_bounds = tick_bounds2
    else:
        bounds = bounds1
        boundsc = bounds1_c
        ticks_bounds = bounds1
    if item[panel] == 'obs':
        da = DS[item[panel]]
    else:
        da = DS[item[panel]].mean(dim='model',skipna=False)

    da.plot.contourf(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs={'orientation': 'horizontal',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'ticks': ticks_bounds,},
            add_labels=False,add_colorbar=True)
    da.plot.contour(ax=ax[panel],colors=['black'],levels=boundsc,linewidths=1)
#    if item[panel] == 'omip1' or item[panel] == 'omip2':
#        ct = da.plot.contour(ax=ax[panel],colors=['black'],levels=boundscb,linewidths=2)
#        ax[panel].clabel(ct,fontsize=10,fmt="%3.1f")

    ax[panel].set_title(title[panel])
    ax[panel].invert_yaxis()
    ax[panel].set_facecolor('lightgray')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
