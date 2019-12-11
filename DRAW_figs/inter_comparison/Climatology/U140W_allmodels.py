# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4
from netCDF4 import Dataset, num2date


if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1' ]

suptitle = head_title[nv_out]  + ' U@140$^\circ$W (average from 1980 to 2009)'

metainfo = [ json.load(open("./json/uo140w_omip1.json")),
             json.load(open("./json/uo140w_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


lattrop = np.linspace(-19.5,19.5,40)
lev14 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500. ]

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
        if model == "MIROC-COCO4.9":
            tmp1 = tmpm[:,:,0]
            tmp = tmp1.sel(lat=slice(None, None, -1))
        if model == 'CESM-POP':
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
        if model == 'NorESM-BLOM':
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
        if (model == 'EC-Earth3-NEMO'):
            tmp = tmpm[0:14,6:46,0]
        if (model == 'Kiel-NEMO'):
            tmp = tmpm[:,0:40,0]
        if (model == 'CMCC-NEMO'):
            tmp = tmpm[:,1:41]
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

fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

ax = [
    plt.subplot(4,3,1),
    plt.subplot(4,3,2),
    plt.subplot(4,3,3),
    plt.subplot(4,3,4),
    plt.subplot(4,3,5),
    plt.subplot(4,3,6),
    plt.subplot(4,3,7),
    plt.subplot(4,3,8),
    plt.subplot(4,3,9),
    plt.subplot(4,3,10),
    plt.subplot(4,3,11),
    plt.subplot(4,3,12),
]

# [left, bottom, width, height]
ax_cbar = plt.axes([0.15,0.06,0.7,0.02])

bounds1 = np.array([-1.2, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
tick_bounds1 = np.array([-1.2, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
bounds1_c =  np.array([-1.2, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
#bounds1_cb = np.arange(-1.5,1.6,0.5)
#
bounds2 = np.array([-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15, 0.2])
tick_bounds2 =  np.array([-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15, 0.2])
bounds2_c =  np.array([-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15, 0.2])

cmap = [ 'RdYlBu_r', 'RdYlBu_r', 'RdBu_r' ]

item = [ 'omip1', 'omip2', 'omip2-1' ]
outfile = './fig/U140W_allmodels_'+item[nv_out]+'.png'

# MMM

nax=11

if item[nv_out] == 'omip1' or item[nv_out] == 'omip2':
    bounds = bounds1
    boundsc = bounds1_c
    ticks_bounds = tick_bounds1
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    boundsc = bounds2_c
    ticks_bounds = tick_bounds2
else:
    bounds = bounds1
    boundsc = bounds1_c
    ticks_bounds = bounds1

da = DS[item[nv_out]].mean(dim='model',skipna=False)
da.plot.contourf(ax=ax[nax],cmap=cmap[nv_out],
                 levels=bounds,
                 extend='both',
                 cbar_kwargs={'orientation': 'horizontal',
                              'spacing':'uniform',
                              'label': '[m/s]',
                              'ticks': ticks_bounds,},
                 cbar_ax = ax_cbar,
                 add_labels=False,add_colorbar=True)

da.plot.contour(ax=ax[nax],colors=['black'],levels=boundsc,linewidths=1)
ax[nax].set_title('MMM',{'fontsize':9,'verticalalignment':'top'})
ax[nax].tick_params(labelsize=8)
ax[nax].invert_yaxis()
ax[nax].set_xlabel('latitude')
ax[nax].set_ylabel('')
ax[nax].set_facecolor('lightgray')

nmodel = 0
for model in model_list[0]:
    if item[nv_out] == 'omip1' or item[nv_out] == 'omip2':
        bounds = bounds1
        boundsc = bounds1_c
        #boundscb = bounds1_cb
        ticks_bounds = tick_bounds1
    elif item[nv_out] == 'omip2-1':
        bounds = bounds2
        boundsc = bounds2_c
        ticks_bounds = tick_bounds2
    else:
        bounds = bounds1
        boundsc = bounds1_c
        ticks_bounds = bounds1
    if item[nv_out] == 'obs':
        da = DS[item[nv_out]]
    else:
        da = DS[item[nv_out]].isel(model=nmodel)
        da.plot.contourf(ax=ax[nmodel],cmap=cmap[nv_out],
                         levels=bounds,
                         extend='both',
                         add_labels=False,add_colorbar=False)

    da.plot.contour(ax=ax[nmodel],colors=['black'],levels=boundsc,linewidths=1)
#    if item[nmodel] == 'omip1' or item[nmodel] == 'omip2':
#        ct = da.plot.contour(ax=ax[nmodel],colors=['black'],levels=boundscb,linewidths=2)
#        ax[nmodel].clabel(ct,fontsize=10,fmt="%3.1f")

    ax[nmodel].set_title(model,{'fontsize':9,'verticalalignment':'top'})
    ax[nmodel].tick_params(labelsize=8)
    ax[nmodel].invert_yaxis()
    if (nmodel > 8):
        ax[nmodel].set_xlabel('latitude')
    else:
        ax[nmodel].set_xlabel('')

    q, mod = divmod(nmodel,3)
    print(q,mod)
    if (mod == 0):        
        ax[nmodel].set_ylabel('depth')
    else:
        ax[nmodel].set_ylabel('')
        
    ax[nmodel].set_facecolor('lightgray')
    nmodel += 1

plt.subplots_adjust(left=0.08,right=0.98,bottom=0.13,top=0.93,hspace=0.26,wspace=0.15)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
