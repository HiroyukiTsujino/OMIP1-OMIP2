# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import datetime

if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' MRICOM (1) or LICOM (2)')
    sys.exit()

model_number = int(sys.argv[1]) - 1

model_name = [ 'MRI.COM', 'LICOM' ]

ystr = 1980
yend = 2009

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)

metainfo = [ json.load(open("./json/tau_omip1_alpha.json")),
             json.load(open("./json/tau_omip2_alpha.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ '(a) Global', '(b) Atlantic', '(c) Pacific' , '(d) Indian' ]

suptitle = 'OMIP2 zonal wind stress  (ave. from ' + str(ystr) + ' to ' + str(yend) + ')'
outfile = './fig/zonal_wind_alpha_'+ model_name[model_number] + '.png'

print ( 'Model list for OMIP1:', model_list[0] )
print ( 'Model list for OMIP2:', model_list[1] )

#J NCAR-POP 補間用情報
y = np.linspace(-89.5,89.5,num=180)


#J 時刻情報 (各モデルの時刻情報を上書きする)

nyr = yend - ystr + 1
time0 = np.empty(nyr,dtype='object')
for yr in range(ystr,yend+1):
    time0[yr-ystr] = datetime.datetime(yr,1,1)

time1 = np.empty((2010-1948)*12,dtype='object')
for yr in range(1948,2010):
    for mon in range(1,13):
        time1[(yr-1948)*12+mon-1] = datetime.datetime(yr,mon,1)

time2 = np.empty((2019-1958)*12,dtype='object')
for yr in range(1958,2019):
    for mon in range(1,13):
        time2[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)

time = [ time1, time2 ]


#J データ読込・平均
print( "Loading GRID data" )

arefile = '../AMIP/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

basinfile = '../WOA/annual/WOA13_1x1_mask.nc'
ncbas = netCDF4.Dataset(basinfile,'r')
basin = ncbas.variables['basin_mask'][:,:]
ncbas.close()

maskglb = np.array(np.zeros((ny,nx)),dtype=np.float64)
maskatl = np.array(np.zeros((ny,nx)),dtype=np.float64)
maskpac = np.array(np.zeros((ny,nx)),dtype=np.float64)
maskind = np.array(np.zeros((ny,nx)),dtype=np.float64)

taux_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
tauy_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)

for i in range(nx):
    for j in range(ny):
        if (basin[j,i] < 0 or basin[j,i] == 53):
            maskglb[j,i] = 0
        else:
            maskglb[j,i] = 1
            
        if (basin[j,i] == 1):
            maskatl[j,i] = 1
        else:
            maskatl[j,i] = 0
            
        if (basin[j,i] == 2 or basin[j,i] == 12):
            maskpac[j,i] = 1
        else:
            maskpac[j,i] = 0
            
        if (basin[j,i] == 3 or basin[j,i] == 8 or basin[j,i] == 56):
            maskind[j,i] = 1
        else:
            maskind[j,i] = 0
            
#

var1 = np.empty( (len(model_list[0]),4,180) )
var2 = np.empty( (len(model_list[1]),4,180) )

for omip in range(2):
    print( "Loading OMIP" + str(omip+1) + " data" )

    if (omip == 0):
        start_yr = 1948
    elif (omip == 1):
        start_yr = 1958

    nmodel = 0
    for model in model_list[omip]:

        print(model)

        path = metainfo[omip][model]['path']
        fnamex= metainfo[omip][model]['fnamex']
        fnamey= metainfo[omip][model]['fnamey']
        infilex =  path + '/' + fnamex
        infiley =  path + '/' + fnamey
        factor = float(metainfo[omip][model]['factor'])
        undef_value = float(metainfo[omip][model]['undef'])
        undef_nan = metainfo[omip][model]['undefnan']
        gtorlt = metainfo[omip][model]['gtorlt']

        
        nctaux = netCDF4.Dataset(infilex,'r')
        taux = nctaux.variables['tauuo'][:,:,:]
        nctaux.close()

        nctauy = netCDF4.Dataset(infiley,'r')
        tauy = nctauy.variables['tauvo'][:,:,:]
        nctauy.close()

        if (undef_nan == 'False'):
            if (gtorlt == 'gt'):
                undef_flags = (taux > undef_value)
            else:
                undef_flags = (taux < undef_value)
        else:
            undef_flags = np.isnan(taux)

        taux[undef_flags] = 0.0
        tauy[undef_flags] = 0.0
        mask_model = np.where(undef_flags, 0, 1)

        taux_annclim = 0.0
        tauy_annclim = 0.0
        day_annclim = 0.0

        for yr in range(ystr,yend+1):

            rec_base = (yr-start_yr)*12

            for mn in range(1,13):

                recn = rec_base + mn - 1
                if (yr == ystr or yr == yend):
                    if (mn == 12):
                        print (yr,mn,recn)

                taux_annclim = taux_annclim + mask_model[recn,:,:] * taux[recn,:,:] * mon_days[mn-1]
                tauy_annclim = tauy_annclim + mask_model[recn,:,:] * tauy[recn,:,:] * mon_days[mn-1]
                day_annclim = day_annclim + mask_model[recn,:,:] * mon_days[mn-1]

        taux_annclim = np.where(day_annclim == 0, 0.0, taux_annclim / day_annclim)
        tauy_annclim = np.where(day_annclim == 0, 0.0, taux_annclim / day_annclim)

        txglb = np.array(np.zeros((ny)),dtype=np.float64)
        txatl = np.array(np.zeros((ny)),dtype=np.float64)
        txpac = np.array(np.zeros((ny)),dtype=np.float64)
        txind = np.array(np.zeros((ny)),dtype=np.float64)
        arglb = np.array(np.zeros((ny)),dtype=np.float64)
        aratl = np.array(np.zeros((ny)),dtype=np.float64)
        arpac = np.array(np.zeros((ny)),dtype=np.float64)
        arind = np.array(np.zeros((ny)),dtype=np.float64)
        masktmp = np.where(taux_annclim == 0, 0, 1)

        for i in range(nx):
            for j in range(ny):
                txglb[j] = txglb[j] + area[j,i] * maskglb[j,i] * taux_annclim[j,i] * masktmp[j,i]
                arglb[j] = arglb[j] + area[j,i] * maskglb[j,i] * masktmp[j,i]
                txatl[j] = txatl[j] + area[j,i] * maskatl[j,i] * taux_annclim[j,i] * masktmp[j,i]
                aratl[j] = aratl[j] + area[j,i] * maskatl[j,i] * masktmp[j,i]
                txpac[j] = txpac[j] + area[j,i] * maskpac[j,i] * taux_annclim[j,i] * masktmp[j,i]
                arpac[j] = arpac[j] + area[j,i] * maskpac[j,i] * masktmp[j,i]
                txind[j] = txind[j] + area[j,i] * maskind[j,i] * taux_annclim[j,i] * masktmp[j,i]
                arind[j] = arind[j] + area[j,i] * maskind[j,i] * masktmp[j,i]

        txglb = np.where(arglb > 0, txglb / arglb, 0)
        txatl = np.where(aratl > 0, txatl / aratl, 0)
        txpac = np.where(arpac > 0, txpac / arpac, 0)
        txind = np.where(arind > 0, txind / arind, 0)
  
        #print(txglb)
        #print(txatl)
        #print(txpac)
        #print(txind)

        txbasin_all = np.array(np.zeros((4,ny)),dtype=np.float64)
        txbasin_all[0,0:ny] = txglb[0:ny]
        txbasin_all[1,0:ny] = txatl[0:ny]
        txbasin_all[2,0:ny] = txpac[0:ny]
        txbasin_all[3,0:ny] = txind[0:ny]
        for j in range(ny):
            print(j,txbasin_all[:,j])

        if (omip == 0):
            var1[nmodel] = np.where( txbasin_all == 0, np.nan, txbasin_all )
        else:
            var2[nmodel] = np.where( txbasin_all == 0, np.nan, txbasin_all )

        nmodel += 1

DS1 = xr.Dataset({'omip1': (['model','basin','lat'], var1), },
                  coords = {'lat': y} )

DS2 = xr.Dataset({'omip2': (['model','basin','lat'], var2), },
                  coords = {'lat': y} )

#J 描画
fig = plt.figure(figsize=(12,16))
fig.suptitle( suptitle, fontsize=20 )

linecol=["darkorange", "green", "red", "blue", "purple", "lightblue"]

ax = [
    plt.subplot(3,1,1),
    plt.subplot(3,1,2),
    plt.subplot(3,1,3),
]

ylim = [ [-0.1, 0.2], [-0.10, 0.14], [-0.10, 0.14] ]
yint = [ 0.05, 0.02, 0.02 ]

for n in range(3):
    #nmodel = 0
    #for model in model_list[0]:
    #    m = nmodel % 2
    #    if (m == 1):
    #        if (n == 1):
    #            DS1['omip1'].sel(model=nmodel).isel(basin=n).where(DS1.lat>-32.0).plot(ax=ax[n],label=model,color=linecol[nmodel],linewidth=1,linestyle="dashed")
    #        elif (n == 2):
    #            DS1['omip1'].sel(model=nmodel).isel(basin=n).where(DS1.lat>-31.0).plot(ax=ax[n],label=model,color=linecol[nmodel],linewidth=1,linestyle="dashed")
    #        else:
    #            DS1['omip1'].sel(model=nmodel).isel(basin=n).plot(ax=ax[n],label=model,color=linecol[nmodel],linewidth=1,linestyle="dashed")
    #
    #    nmodel += 1
    #
    nmodel = 0
    for model in model_list[1]:
        m = nmodel % 2
        if (m == model_number):
            if (n == 1):
                DS2['omip2'].sel(model=nmodel).isel(basin=n).where(DS2.lat>-32.0).plot(ax=ax[n],label=model,color=linecol[nmodel],linewidth=1)
            elif (n == 2):
                DS2['omip2'].sel(model=nmodel).isel(basin=n).where(DS2.lat>-31.0).plot(ax=ax[n],label=model,color=linecol[nmodel],linewidth=1)
            else:
                DS2['omip2'].sel(model=nmodel).isel(basin=n).plot(ax=ax[n],label=model,color=linecol[nmodel],linewidth=1)

        nmodel += 1

    ax[n].set_title(title[n])
    ax[n].set_xlabel("Latitude")
    ax[n].set_xlim(-90,90)
    ax[n].set_xticks(np.linspace(-90,90,7))
    ax[n].set_ylabel("Zonal wind stress [$\mathrm{N}\,\mathrm{m}^{-2}$]")
    ax[n].set_ylim(ylim[n][0],ylim[n][1])
    ax[n].set_yticks(np.arange(ylim[n][0],ylim[n][1]+yint[n],yint[n]))
    ax[n].axhline(y=0,color='k',linewidth=2)
    ax[n].legend()
    ax[n].grid()

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
