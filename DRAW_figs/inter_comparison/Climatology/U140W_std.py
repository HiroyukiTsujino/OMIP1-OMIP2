# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4
from netCDF4 import Dataset, num2date

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()

metainfo = [ json.load(open("./json/uo140w_omip1.json")),
             json.load(open("./json/uo140w_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ '(a) Ensemble mean OMIP1', '(b) Ensemble mean OMIP2',
          '(c) Ensemble std OMIP1' , '(d) Ensemble std OMIP2',
          '(e) OMIP2 - OMIP1'      , '(f) Johnson et al. (2002)' ]

lattrop = np.linspace(-19.5,19.5,40)
lev14 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500. ]

if (sys.argv[1] == 'MMM'):
    outfile = './fig/U140W_MMM'
    suptitle = 'Multi Model Mean' + ' Zonal velocity at eastern Tropical Pacific (ave. from 1980 to 2009)'
else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/U140W_' + sys.argv[1]
    suptitle = sys.argv[1] + ' Zonal velocity at eastern Tropical Pacific (ave. from 1980 to 2009)'

print("Drawing "+suptitle)

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]


# Reference data (Johnson et al. 2002)

print( "Loading Johnson et al. 2002 data" )
reffile = '../refdata/Johnson_et_al_2002/meanfit_m.cdf'
da_ref = xr.open_dataset( reffile )["UM"]

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
    plt.subplot(3,2,1),
    plt.subplot(3,2,2),
    plt.subplot(3,2,3),
    plt.subplot(3,2,4),
    plt.subplot(3,2,5),
    plt.subplot(3,2,6)
]

bounds1 = np.array([-1.2, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
tick_bounds1 = np.array([-1.2, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
bounds1_c =  np.array([-1.2, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2])
#bounds1_cb = np.arange(-1.5,1.6,0.5)
#
bounds2 = np.array([-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15, 0.2])
tick_bounds2 =  np.array([-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15, 0.2])
bounds2_c =  np.array([-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15, 0.2])
#
bounds3 = np.arange(0.0,0.20,0.02)
tick_bounds3 = np.arange(0.0,0.20,0.02)
bounds3_c = np.arange(0.0,0.20,0.02)

cmap = [ 'RdYlBu_r', 'RdYlBu_r', 'terrain', 'terrain', 'RdBu_r', 'RdYlBu_r' ]
extflg = [ 'both', 'both', 'max', 'max', 'both', 'both' ]

item = [ 'omip1', 'omip2', 'omip1std', 'omip2std', 'omip2-1', 'UM' ]


bounds = bounds1
boundsc = bounds1_c
ticks_bounds = tick_bounds1

da_ref.sel(XLON=220).plot.contourf(ax=ax[5],cmap=cmap[5],
            levels=bounds,
            extend=extflg[5],
            cbar_kwargs={'orientation': 'vertical',
                         'spacing':'uniform',
                         'ticks': ticks_bounds,},
            add_labels=False,add_colorbar=True)
da_ref.sel(XLON=220).plot.contour(ax=ax[5],colors=['black'],levels=boundsc,linewidths=1)

ax[5].set_title(title[5],{'fontsize':10, 'verticalalignment':'top'})
ax[5].tick_params(labelsize=9)
ax[5].set_xlim([-12,17])
ax[5].set_xlabel('latitude')
ax[5].set_ylim([500,0])
ax[5].set_ylabel('depth')

ddof_dic={'ddof' : 0}

for panel in range(5):
    if item[panel] == 'omip1' or item[panel] == 'omip2':
        bounds = bounds1
        boundsc = bounds1_c
        #boundscb = bounds1_cb
        ticks_bounds = tick_bounds1
        da = DS[item[panel]].mean(dim='model',skipna=False)
    elif item[panel] == 'omip1std':
        bounds = bounds3
        boundsc = bounds3_c
        ticks_bounds = tick_bounds3
        da = DS['omip1'].std(dim='model',skipna=False, **ddof_dic)
    elif item[panel] == 'omip2std':
        bounds = bounds3
        boundsc = bounds3_c
        ticks_bounds = tick_bounds3
        da = DS['omip2'].std(dim='model',skipna=False, **ddof_dic)
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        boundsc = bounds2_c
        ticks_bounds = tick_bounds2
        da = DS[item[panel]].mean(dim='model',skipna=False)
    else:
        bounds = bounds1
        boundsc = bounds1_c
        ticks_bounds = bounds1
        da = DS[item[panel]]

    da.plot.contourf(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend=extflg[panel],
            cbar_kwargs={'orientation': 'vertical',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'ticks': ticks_bounds,},
            add_labels=False,add_colorbar=True)
    da.plot.contour(ax=ax[panel],colors=['black'],levels=boundsc,linewidths=1)
#    if item[panel] == 'omip1' or item[panel] == 'omip2':
#        ct = da.plot.contour(ax=ax[panel],colors=['black'],levels=boundscb,linewidths=2)
#        ax[panel].clabel(ct,fontsize=10,fmt="%3.1f")

    ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top'})
    ax[panel].tick_params(labelsize=9)
    if (panel == 4 or panel == 5):
        ax[panel].set_xlabel('latitude')
    else:
        ax[panel].set_xlabel('')
    
    ax[panel].invert_yaxis()
    ax[panel].set_facecolor('lightgray')
    ax[panel].set_xlim([-12,17])

plt.subplots_adjust(left=0.08,right=0.98,top=0.92,bottom=0.08,wspace=0.12)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
