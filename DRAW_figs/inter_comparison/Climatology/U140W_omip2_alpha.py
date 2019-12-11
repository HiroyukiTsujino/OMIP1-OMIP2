# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4
from netCDF4 import Dataset, num2date


#if (len(sys.argv) < 2):
#    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
#    sys.exit()

#omip_out = int(sys.argv[1])
#nv_out = int(sys.argv[1]) - 1

#head_title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1' ]

suptitle = ' U@140$^\circ$W (average from 1980 to 2009)'

metainfo = json.load(open("./json/uo140w_omip2_alpha.json"))
model_list = metainfo.keys()


lattrop = np.linspace(-19.5,19.5,40)
lev14 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500. ]

print("Drawing "+suptitle)

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = np.linspace(1958,2018,61)


# Reference data (Johnson et al. 2002)

print( "Loading Johnson et al. 2002 data" )
reffile = '../refdata/Johnson_et_al_2002/meanfit_m.cdf'
da_ref = xr.open_dataset( reffile )["UM"]


d = np.empty( (len(model_list),14,40) )
nmodel = 0
for model in model_list:
    path = metainfo[model]['path']
    fname = metainfo[model]['fname']
    var = metainfo[model]['name']
    infile = path + '/' + fname

    print (model)
    DS_read = xr.open_dataset(infile,decode_times=False)
    DS_read['time'] = time

    tmpm = DS_read[var].sel(time=slice(1980,2009)).mean(dim='time')

    if (model == 'MRI.COM_alpha1_0' or model == 'MRI.COM_alpha0_7' or model == 'MRI.COM_alpha0_0'):
        tmp = tmpm
    if (model == 'CAS-LICOM3_alpha1_0' or model == 'CAS-LICOM3_alpha0_7'):
        tmp1 = tmpm.sel(lat=slice(None, None, -1))
        tmp = tmp1[:,1:41]

    d[nmodel] = tmp.values
    nmodel += 1

DS = xr.Dataset({'omip2': (['model','depth','lat'], d),},
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

# [left, bottom, width, height]
ax_cbar = plt.axes([0.15, 0.06, 0.7, 0.02])

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

title = [ '(a) ', '(b) ', '(c) ', '(d) ', '(e) ', '(f) Johnson et al. (2002)' ]


bounds = bounds1
boundsc = bounds1_c
ticks_bounds = tick_bounds1

da_ref.sel(XLON=220).plot.contourf(ax=ax[5],cmap=cmap[0],
                             levels=bounds,
                             extend='both',
                             add_labels=False,add_colorbar=False)

da_ref.sel(XLON=220).plot.contour(ax=ax[5],colors=['black'],levels=boundsc,linewidths=1)

ax[5].set_title(title[5],{'fontsize':10, 'verticalalignment':'top'})
ax[5].tick_params(labelsize=9)
ax[5].set_xlim([-12,17])
ax[5].set_xlabel('latitude')
ax[5].set_ylim([500,0])
ax[5].set_ylabel('depth')


nv_out = 1
nmodel = 0
for model in model_list:
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
        if (nmodel == 0):
            da.plot.contourf(ax=ax[nmodel],cmap=cmap[nv_out],
                             levels=bounds,
                             extend='both',
                             cbar_kwargs={'orientation': 'horizontal',
                                          'spacing':'uniform',
                                          'label': '[$\mathrm{m} \, \mathrm{s}^{-1}$]',
                                          'ticks': ticks_bounds,},
                             cbar_ax = ax_cbar,
                             add_labels=False,add_colorbar=True)
        else:
            da.plot.contourf(ax=ax[nmodel],cmap=cmap[nv_out],
                             levels=bounds,
                             extend='both',
                             add_labels=False,add_colorbar=False)

    da.plot.contour(ax=ax[nmodel],colors=['black'],levels=boundsc,linewidths=1)

    ax[nmodel].set_title(title[nmodel]+model,{'fontsize':10, 'verticalalignment':'top'})
    ax[nmodel].tick_params(labelsize=9)
    ax[nmodel].invert_yaxis()
    ax[nmodel].set_xlabel('')
    ax[nmodel].set_xlim([-12,17])
    ax[nmodel].set_facecolor('lightgray')
    if (nmodel == 4):
        ax[nmodel].set_xlabel('latitude')
    nmodel += 1

outfile = './fig/U140W_omip2_alpha'
plt.subplots_adjust(left=0.08,right=0.98,top=0.92,bottom=0.15,wspace=0.15)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

