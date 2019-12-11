# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import netCDF4

if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3) or OMIP2-OMIP1 shallow (4)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1' ]

suptitle = head_title[nv_out] + ' Meridional Overturning Circulation (ave. from 1980 to 2009)'

metainfo = [ json.load(open("./json/moc_omip1.json")),
             json.load(open("./json/moc_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

#title = [ 'Southern Ocean', 'Atlantic Ocean', 'Indo-Pacific Ocean' ]

xlim  = [ [-90, -28], [-30, 90], [-30, 70] ]

lev33 = np.array([ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ])

latwoa = np.linspace(-89.5,89.5,num=180)

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

#for omip in range(2):
#    for nm in range(nummodel[omip]):
#        print(modnam[0][nm]+'-omip'+str(omip+1),DS['omip'+str(omip+1)].sel(model=nm,basin=0,depth=slice(2000,6500)).interp(lat=-30).min(dim="depth").values)


#J 描画

fig = plt.figure(figsize=(16,12))
fig.suptitle( suptitle, fontsize=20 )

# [left, bottom, width, height]
axes0 = np.array([0.05, 0.75, 0.25, 0.18])

ax = [ plt.axes(axes0),
       plt.axes(axes0+np.array([0.31,0,0,0])),
       plt.axes(axes0+np.array([0.62,0,0,0])),
       plt.axes(axes0+np.array([0,-0.23,0,0])),
       plt.axes(axes0+np.array([0.31,-0.23,0,0])),
       plt.axes(axes0+np.array([0.62,-0.23,0,0])),
       plt.axes(axes0+np.array([0,-0.46,0,0])),
       plt.axes(axes0+np.array([0.31,-0.46,0,0])),
       plt.axes(axes0+np.array([0.62,-0.46,0,0])),
       plt.axes(axes0+np.array([0,-0.69,0,0])),
       plt.axes(axes0+np.array([0.31,-0.69,0,0])),
       plt.axes(axes0+np.array([0.62,-0.69,0,0])),]

# [left, bottom, width, height]
ax_cbar = plt.axes([0.945,0.15,0.02,0.7])

model_title_locx = [0.17,0.48,0.79]
model_title_locy = [0.94,0.71,0.48,0.25]

bounds1 = np.linspace(-8,27,36)
bounds2 = np.linspace(-8,8,17)

cmap = [ 'gist_ncar', 'gist_ncar', 'RdBu_r', 'RdBu_r' ]

item = [ 'omip1', 'omip2', 'omip2-1', 'omip2-1' ]

if (nv_out == 4):
    outfile = './fig/AMOC_'+ item[nv_out] + '_shallow.png'
else:
    outfile = './fig/AMOC_'+ item[nv_out] + '.png'


# MMM

if item[nv_out] == 'omip1' or item[nv_out] == 'omip2':
    bounds = bounds1
    ticks_bounds = bounds1
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    ticks_bounds = bounds2

if (nv_out == 4):
    da = DS[item[nv_outl]].sel(depth=slice(0,500)).mean(dim='model',skipna=False)
else:
    da = DS[item[nv_out]].mean(dim='model',skipna=False)

da.isel(basin=1).plot(ax=ax[11],cmap=cmap[nv_out],
                      levels=bounds,
                      extend='both',
                      cbar_kwargs={'orientation': 'vertical',
                                   'spacing':'uniform',
                                   'ticks': ticks_bounds,},
                      cbar_ax=ax_cbar,
                      add_labels=False,add_colorbar=True)

#da.isel(basin=1).plot.contour(ax=ax[11],
#                              colors='black',
#                              levels=bounds,
#                              add_labels=False,
#                              linewidths=1.0)

#ax[11].set_title(title[1],{'fontsize':8,'verticalalignment':'baseline'})
ax[11].invert_yaxis()
ax[11].set_xlim(xlim[1][0],xlim[1][1])
ax[11].set_xticks(np.arange(xlim[1][0],xlim[1][1]+0.1,30))
ax[11].set_facecolor('lightgray')
ax[11].tick_params(axis='y',labelleft=False)

#######

nmodel = 0
nax = 0
for model in model_list[0]:

    if item[nv_out] == 'omip1' or item[nv_out] == 'omip2':
        bounds = bounds1
        ticks_bounds = bounds1
    elif item[nv_out] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2

    if (nv_out == 4):
        #da = DS[item[nv_outl]].sel(depth=slice(0,500)).mean(dim='model',skipna=False)
        da = DS[item[nv_outl]].isel(model=nmodel).sel(depth=slice(0,500))
    else:
        #da = DS[item[nv_out]].mean(dim='model',skipna=False)
        da = DS[item[nv_out]].isel(model=nmodel)

    if (nmodel == 0):
        da.isel(basin=1).plot(ax=ax[nax],cmap=cmap[nv_out],
                              levels=bounds,
                              extend='both',
                              cbar_kwargs={'orientation': 'vertical',
                                           'spacing':'uniform',
                                           'ticks': ticks_bounds,},
                              cbar_ax=ax_cbar,
                              add_labels=False,add_colorbar=True)
    else:
        da.isel(basin=1).plot(ax=ax[nax],cmap=cmap[nv_out],
                              levels=bounds,
                              extend='both',
                              add_labels=False,add_colorbar=False)
        
#    da.isel(basin=1).plot.contour(ax=ax[nax],
#                                  colors='black',
#                                  levels=bounds,
#                                  add_labels=False,
#                                  linewidths=1.0)

    #ax[nax].set_title(title[1],{'fontsize':8,'verticalalignment':'baseline'})
    ax[nax].invert_yaxis()
    ax[nax].set_xlim(xlim[1][0],xlim[1][1])
    ax[nax].set_xticks(np.arange(xlim[1][0],xlim[1][1]+0.1,30))
    ax[nax].set_facecolor('lightgray')
    ax[nax].tick_params(axis='y',labelleft=False)

    nax = nax + 1

    nmodel += 1
        
nmodel = 0
nax = 0
for model in model_list[0]:
    ny = int(nax / 3)
    nx = nax - ny * 3
    print(nx,ny)
    fig.text(model_title_locx[nx],model_title_locy[ny],model,fontsize=14,horizontalalignment='center',verticalalignment='center')
    nax = nax + 1

    nmodel += 1
    
fig.text(model_title_locx[2],model_title_locy[3],'MMM',fontsize=14,horizontalalignment='center',verticalalignment='center')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
