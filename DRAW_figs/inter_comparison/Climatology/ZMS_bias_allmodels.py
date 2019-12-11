# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4


if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1 - WOA13v2', 'OMIP2 - WOA13v2', 'OMIP2 - OMIP1' ]

suptitle = head_title[nv_out]  + ' Zonal mean salinity (ave. from 1980 to 2009)'


metainfo = [ json.load(open("./json/zms_omip1.json")),
             json.load(open("./json/zms_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ 'Southern Ocean', 'Atlantic Ocean', 
          'Indian Ocean', 'Pacific Ocean' ]

xlim = [ [-90, -30], [-30, 90], [-30, 30], [-30, 60] ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]


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
            so_glb = np.where(so_glb > 50.0, np.NaN, so_glb)
            so_glb = np.where(so_glb < 10.0, np.NaN, so_glb)
            
            ncatl = netCDF4.Dataset(infile_atl,'r')
            so_atl = ncatl.variables['so_atl'][:,:,:]
            ncatl.close()
            so_atl = np.where(so_atl > 50.0, np.NaN, so_atl)
            so_atl = np.where(so_atl < 10.0, np.NaN, so_atl)

            ncind = netCDF4.Dataset(infile_ind,'r')
            so_ind = ncind.variables['so_ind'][:,:,:]
            ncind.close()
            so_ind = np.where(so_ind > 50.0, np.NaN, so_ind)
            so_ind = np.where(so_ind < 10.0, np.NaN, so_ind)

            ncpac = netCDF4.Dataset(infile_pac,'r')
            so_pac = ncpac.variables['so_pac'][:,:,:]
            ncpac.close()
            so_pac = np.where(so_pac > 50.0, np.NaN, so_pac)
            so_pac = np.where(so_pac < 10.0, np.NaN, so_pac)

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

        if model == 'NorESM-BLOM':
            tmp = tmp.transpose().interp(depth=lev33)
        if model == 'AWI-FESOM':
            tmp = tmp.transpose()
        if model == "MIROC-COCO4.9":
            tmp = tmp.sel(lat=slice(None, None, -1))
        if model == 'EC-Earth3-NEMO':
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

bounds1 = [-0.4, -0.3, -0.2, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4]
bounds2 = [-0.2, -0.15, -0.1, -0.07, -0.04, -0.02, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2 ]
bounds3 = [33.0, 34.0, 34.2, 34.4, 34.6, 34.7, 34.8, 34.9, 35.0, 35.2, 35.5, 35.8, 36.1, 36.5, 36.9 ]

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]
outfile = './fig/ZMS_bias_' + item[nv_out] + '.png'

# MMM

if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
    bounds = bounds1
    ticks_bounds = bounds1
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    ticks_bounds = bounds2
else:
    bounds = bounds3
    ticks_bounds = bounds3

da = DS[item[nv_out]].mean(dim='model',skipna=False)
for m in range(4):
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

    ax[11][m].set_title(title[m],{'fontsize':6,'verticalalignment':'top'})
    ax[11][m].tick_params(labelsize=7)
    ax[11][m].invert_yaxis()
    ax[11][m].set_xlim(xlim[m][0],xlim[m][1])
    ax[11][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
    ax[11][m].set_facecolor('lightgray')
for m in range(1,4):
    ax[11][m].tick_params(axis='y',labelleft=False)



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
        da.isel(basin=m).plot(ax=ax[nmodel][m],cmap=cmap[nv_out],
                              levels=bounds,
                              extend='both',
                              add_labels=False,add_colorbar=False)

        ax[nmodel][m].set_title(title[m],{'fontsize':6,'verticalalignment':'top'})
        ax[nmodel][m].tick_params(labelsize=7)
        ax[nmodel][m].invert_yaxis()
        ax[nmodel][m].set_xlim(xlim[m][0],xlim[m][1])
        ax[nmodel][m].set_xticks(np.arange(xlim[m][0],xlim[m][1]+0.1,30))
        ax[nmodel][m].set_facecolor('lightgray')
    for m in range(1,4):
        ax[nmodel][m].tick_params(axis='y',labelleft=False)

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
if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
