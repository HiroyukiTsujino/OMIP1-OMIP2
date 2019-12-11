# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import datetime

#ystr = 1980
#yend = 2009
ystr = 1988
yend = 2007

ylim = [ [-1.4, 2.6], [0, 1.5], [-1.8, 1.5] ]
yint = [ 0.2, 0.2, 0.2 ]

if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1', 'OMIP2' ]

suptitle = head_title[nv_out]  + ' (northward heat transport ave. from '+ str(ystr) + ' to ' + str(yend) + ')'

metainfo = [ json.load(open("./json/hfbasin_omip1.json")),
             json.load(open("./json/hfbasin_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ '(a) Global', '(b) Atlantic-Arctic', '(c) Indo-Pacific' ]

outfile = './fig/heat_transport_' + head_title[nv_out] + '.png'

print ( 'Model list for OMIP1:', model_list[0] )
print ( 'Model list for OMIP2:', model_list[1] )

lineinfo = json.load(open('../json/inst_color_style.json'))

#J NCAR-POP 補間用情報
y = np.linspace(-89.5,89.5,num=180)


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]

timem1 = np.empty((2010-1948)*12,dtype='object')
for yr in range(1948,2010):
    for mon in range(1,13):
        timem1[(yr-1948)*12+mon-1] = datetime.datetime(yr,mon,1)
timem2 = np.empty((2019-1958)*12,dtype='object')
for yr in range(1958,2019):
    for mon in range(1,13):
        timem2[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)
timem = [ timem1, timem2 ]


#J データ読込・平均

data = []

lincol = []
linsty = []
nummodel = []

for omip in range(2):
    var = np.empty( (len(model_list[omip]),3,180) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    coltmp = []
    stytmp = []
    nmodel = 0
    for model in model_list[omip]:
        path = metainfo[omip][model]['path']
        fname= metainfo[omip][model]['fname']
        infile =  path + '/' + fname
        factor = float(metainfo[omip][model]['factor'])

        coltmp +=[lineinfo[model]["color"]]
        stytmp +=[lineinfo[model]["style"]]

        if ( model == 'Kiel-NEMO' ):

            infile_glb = infile + '_global_update.nc'
            infile_atl = infile + '_atl_update.nc'
            infile_indpac = infile + '_indpac_update.nc'

            ncglb = netCDF4.Dataset(infile_glb,'r')
            #hfbasin_glb = ncglb.variables['hfbasin_GM_global'][:,:]
            hfbasin_glb = ncglb.variables['hfbasin_total_global'][:,:]
            ncglb.close()
            hfbasin_glb = np.where(hfbasin_glb > 9.9e36, np.NaN, hfbasin_glb)
            
            ncatl = netCDF4.Dataset(infile_atl,'r')
            #hfbasin_atl = ncatl.variables['hfbasin_GM_atl'][:,:]
            hfbasin_atl = ncatl.variables['hfbasin_total_atl'][:,:]
            ncatl.close()
            hfbasin_atl = np.where(hfbasin_atl > 9.9e36, np.NaN, hfbasin_atl)

            ncindpac = netCDF4.Dataset(infile_indpac,'r')
            #hfbasin_indpac = ncindpac.variables['hfbasin_GM_indpac'][:,:]
            hfbasin_indpac = ncindpac.variables['hfbasin_total_indpac'][:,:]
            ncindpac.close()
            hfbasin_inp = np.where(hfbasin_indpac > 9.9e36, np.NaN, hfbasin_indpac)

            if ( omip == 0 ):
                hfbasin_all = np.array(np.zeros((62,3,180)),dtype=np.float32)
                hfbasin_all[0:62,0,0:180] = hfbasin_glb[0:62,0:180]
                hfbasin_all[0:62,1,0:180] = hfbasin_atl[0:62,0:180]
                hfbasin_all[0:62,2,0:180] = hfbasin_indpac[0:62,0:180]
                print(hfbasin_glb[0:62,90])
                DS = xr.Dataset({'hfbasin': (['time','basin','lat'], hfbasin_all)},
                                 coords = {'time' : time[omip], 'lat': np.linspace(-89.5,89.5,num=180) } )

            else:
                hfbasin_all = np.array(np.zeros((61,3,180)),dtype=np.float32)
                hfbasin_all[0:61,0,0:180] = hfbasin_glb[0:61,0:180]
                hfbasin_all[0:61,1,0:180] = hfbasin_atl[0:61,0:180]
                hfbasin_all[0:61,2,0:180] = hfbasin_indpac[0:61,0:180]
                DS = xr.Dataset({'hfbasin': (['time','basin','lat'], hfbasin_all)},
                                 coords = {'time' : time[omip], 'lat': np.linspace(-89.5,89.5,num=180) } )

        elif (model == "CMCC-NEMO"):
            
            DS = xr.open_dataset( infile, decode_times=False )
            DS['time'] = timem[omip]
            lattmp = DS['lat'].values
            ny = len(lattmp)
            lattmp[ny-1] = lattmp[ny-2] + (lattmp[ny-2] - lattmp[ny-3])
            print(ny,lattmp)
            DS['lat'] = lattmp
            
        else:

            DS = xr.open_dataset( infile, decode_times=False )
            if (model == "GFDL-MOM"):
                DS = DS.rename({"year":"time"})

            DS['time'] = time[omip]
        
        if (model == "CMCC-NEMO"):
            tmp = DS.hfbasin.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)*factor
            tmp = tmp.interp(lat=y)
        else:
            tmp = DS.hfbasin.sel(time=slice(ystr,yend)).mean(dim='time',skipna=False)*factor

        if model == 'CESM-POP':
            tmp = tmp.interp(lat=y).isel(basin=[2,0,1])
        if model == 'NorESM-BLOM':
            tmp = tmp.rename({"region":"basin"})
            tmp = tmp.interp(lat=y).isel(basin=[3,1,2])
        if model == 'AWI-FESOM':
            tmp = tmp.transpose()
        if model == 'CAS-LICOM3':
            tmp = tmp.sel(lat=slice(None, None, -1))
        if model == 'GFDL-MOM':
            tmp = tmp.interp(yq=y).isel(basin=[2,0,1])
        if model == 'CMCC-NEMO':
            tmp = tmp.interp(lat=y)

        var[nmodel] = np.where( tmp.values == 0, np.nan, tmp.values )
        nmodel += 1

    data += [var]
    lincol += [coltmp]
    linsty += [stytmp]
    nummodel += [nmodel]

DS = xr.Dataset({'omip1': (['model','basin','lat'], data[0]), 
                 'omip2': (['model','basin','lat'], data[1]), },
                coords = {'lat': y} )


#J 描画
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

ax = [
    plt.subplot(3,1,1),
    plt.subplot(3,1,2),
    plt.subplot(3,1,3),
]

for n in range(3):
    if (omip_out == 1):        
        DS.omip1.mean(dim='model',skipna=False).isel(basin=n).plot(ax=ax[n], color='darkgrey', linewidth=4, label='MMM')
    else:
        DS.omip2.mean(dim='model',skipna=False).isel(basin=n).plot(ax=ax[n], color='darkgrey', linewidth=4, label='MMM')

    nmodel = 0
    for model in model_list[omip_out-1]:
        linecol=lincol[omip_out-1][nmodel]
        linesty=linsty[omip_out-1][nmodel]
        if (n == 1):
            DS['omip'+str(omip_out)].sel(model=nmodel).isel(basin=n).where(DS.lat>-32.0).plot(ax=ax[n],label=model,color=linecol,linewidth=1,linestyle=linesty)
        elif (n == 2):
            DS['omip'+str(omip_out)].sel(model=nmodel).isel(basin=n).where(DS.lat>-31.0).plot(ax=ax[n],label=model,color=linecol,linewidth=1,linestyle=linesty)
        else:
            DS['omip'+str(omip_out)].sel(model=nmodel).isel(basin=n).plot(ax=ax[n],label=model,color=linecol,linewidth=1,linestyle=linesty)

        nmodel += 1

    ax[n].set_title(title[n])
    ax[n].set_xlabel("Latitude")
    ax[n].set_xlim(-90,90)
    ax[n].set_xticks(np.linspace(-90,90,7))
    ax[n].set_ylabel("Heat transport [PW]")
    ax[n].set_ylim(ylim[n][0],ylim[n][1])
    ax[n].set_yticks(np.arange(ylim[n][0],ylim[n][1]+yint[n],yint[n]))
    ax[n].legend(bbox_to_anchor=(0.01,1.0),loc='upper left',ncol=2,fontsize=8)
    ax[n].grid()


ax[0].text( 47.5, 0.68,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text( 36.0, 1.11,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text( 24.0, 1.62,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(  9.5, 1.50,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(-10.5, 0.55,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(-19.5,-0.43,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(-31.0,-0.51,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')

ax[0].vlines( 47.5, 0.68 - 0.15, 0.68 + 0.15)
ax[0].vlines( 36.0, 1.11 - 0.37, 1.11 + 0.37)
ax[0].vlines( 24.0, 1.62 - 0.40, 1.62 + 0.40)
ax[0].vlines(  9.5, 1.50 - 1.54, 1.50 + 1.54)
ax[0].vlines(-10.5, 0.50 - 1.45, 0.50 + 1.45)
ax[0].vlines(-19.5,-0.43 - 0.61,-0.43 + 0.61)
ax[0].vlines(-31.0,-0.51 - 0.39,-0.51 + 0.39)

ax[1].text( 46.0, 0.58,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 37.0, 0.88,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 25.0, 1.20,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 10.5, 1.07,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text(-11.5, 0.56,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text(-32.0, 0.34,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 26.5, 1.24,"x",fontsize=17,horizontalalignment='center',verticalalignment='center')

ax[1].vlines( 46.0, 0.58 - 0.24, 0.58 + 0.24)
ax[1].vlines( 37.0, 0.88 - 0.22, 0.88 + 0.22)
ax[1].vlines( 25.0, 1.20 - 0.27, 1.20 + 0.27)
ax[1].vlines( 10.5, 1.07 - 0.33, 1.07 + 0.33)
ax[1].vlines(-11.5, 0.56 - 0.26, 0.55 + 0.26)
ax[1].vlines(-32.0, 0.34 - 0.18, 0.34 + 0.18)
ax[1].vlines( 26.5, 1.24 - 0.33, 1.24 + 0.33)

ax[2].text( 47.5, 0.04,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text( 23.0, 0.64,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text( 10.5, 0.51,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text(-18.5,-1.15,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text(-29.5,-0.91,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
    
ax[2].vlines( 47.5, 0.04 - 0.16, 0.04 + 0.16)
ax[2].vlines( 23.0, 0.64 - 0.29, 0.64 + 0.29)
ax[2].vlines( 10.5, 0.51 - 1.22, 0.51 + 1.22)
ax[2].vlines(-18.5,-1.15 - 0.61,-1.15 + 0.61)
ax[2].vlines(-29.5,-0.91 - 0.36,-0.91 + 0.36)

plt.subplots_adjust(left=0.1,right=0.98,top=0.92,bottom=0.05,hspace=0.25)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
