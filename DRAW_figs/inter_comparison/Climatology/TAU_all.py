# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import datetime

ystr = 1999
yend = 2009
mstr = 11
mend = 10

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)

metainfo = [ json.load(open("./json/tau_omip1.json")),
             json.load(open("./json/tau_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ '(a) Global', '(b) Atlantic', '(c) Pacific' , '(d) Indian' ]

suptitle = 'Zonal wind stress  (ave. from ' + str(ystr) + '-' + str(mstr) + ' to ' + str(yend) + '-' + str(mend) + ')'
outfile = './fig/TAUX_'+str(ystr)+'-'+str(yend)+'_MMM'

print ( 'Model list for OMIP1:', model_list[0] )
print ( 'Model list for OMIP2:', model_list[1] )

#J NCAR-POP 補間用情報
y = np.linspace(-89.5,89.5,num=180)

# Reference data (Johnson et al. 2002)

path_ref='../refdata/SCOW/'

print( "Loading SCOW data" )

infile_atl = path_ref + 'taux_atl_zm.nc'
infile_pac = path_ref + 'taux_pac_zm.nc'
infile_glb = path_ref + 'taux_glb_zm.nc'

ncglb = netCDF4.Dataset(infile_glb,'r')
time_ref = ncglb.variables['time'][:]
lat_ref = ncglb.variables['lat'][:]
tauuo_glb = ncglb.variables['tauuo_glb'][:,:]
ncglb.close()
tauuo_glb = np.where(tauuo_glb < -9.0e33, np.NaN, tauuo_glb)
            
ncatl = netCDF4.Dataset(infile_atl,'r')
tauuo_atl = ncatl.variables['tauuo_atl'][:,:]
ncatl.close()
tauuo_atl = np.where(tauuo_atl < -9.0e33, np.NaN, tauuo_atl)

ncpac = netCDF4.Dataset(infile_pac,'r')
tauuo_pac = ncpac.variables['tauuo_pac'][:,:]
ncpac.close()
tauuo_pac = np.where(tauuo_pac < -9.0e33, np.NaN, tauuo_pac)

tauuo_all = np.array(np.zeros((12,3,720)),dtype=np.float32)
tauuo_all[0:12,0,0:720] = tauuo_glb[0:12,0:720]
tauuo_all[0:12,1,0:720] = tauuo_atl[0:12,0:720]
tauuo_all[0:12,2,0:720] = tauuo_pac[0:12,0:720]
DS_ref = xr.Dataset({'tauuo': (['time','basin','lat'], tauuo_all)},
                    coords = {'time' : time_ref, 'lat': lat_ref } )


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

arefile = '../refdata/PCMDI-SST/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

basinfile = '../refdata/WOA13v2/1deg_L33/const/WOA13_1x1_mask.nc'
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

data = []

for omip in range(2):
    var = np.empty( (len(model_list[omip]),4,180) )
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

        if model == "Kiel-NEMO":
            varx='sozotaux'
            vary='sometauy'
        else:
            varx='tauuo'
            vary='tauvo'
        
        if (model == 'CMCC-NEMO'):
            nctaux = netCDF4.Dataset(infilex,'r')
            taux_annclim = nctaux.variables[varx][:,:]
            nctaux.close()
            nctauy = netCDF4.Dataset(infiley,'r')
            tauy_annclim = nctauy.variables[vary][:,:]
            nctauy.close()

            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flags = (taux_annclim > undef_value)
                elif (gtorlt == 'lt'):
                    undef_flags = (taux_annclim < undef_value)
                else:
                    undef_flags = (abs(taux_annclim) < 1.e-10)
            else:
                undef_flags = np.isnan(taux_annclim)

            taux_annclim[undef_flags] = 0.0
            tauy_annclim[undef_flags] = 0.0
            mask_model = np.where(undef_flags, 0, 1)

        else:

            nctaux = netCDF4.Dataset(infilex,'r')
            taux = nctaux.variables[varx][:,:,:]
            nctaux.close()

            nctauy = netCDF4.Dataset(infiley,'r')
            tauy = nctauy.variables[vary][:,:,:]
            nctauy.close()

            if model == "MIROC-COCO4.9":
                txtmp = np.flip(taux,1).copy()
                tytmp = np.flip(tauy,1).copy()
                taux = txtmp
                tauy = tytmp
        
            if model == "NorESM-BLOM":
                txtmp = np.roll(taux,180,axis=2).copy()
                tytmp = np.roll(tauy,180,axis=2).copy()
                taux = txtmp
                tauy = tytmp
        
            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flags = (taux > undef_value)
                elif (gtorlt == 'lt'):
                    undef_flags = (taux < undef_value)
                else:
                    undef_flags = (abs(taux) < 1.e-10)
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

                mst = 1
                med = 12
                if (yr == ystr):
                    mst = mstr
                if (yr == yend):
                    med = mend
            

                for mn in range(mst,med+1):

                    print(yr,mn)
                    recn = rec_base + mn - 1

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

        var[nmodel] = np.where( txbasin_all == 0, np.nan, txbasin_all )

        nmodel += 1

    data += [var]

DS = xr.Dataset({'omip1': (['model','basin','lat'], data[0]),
                 'omip2': (['model','basin','lat'], data[1]),},
                 coords = {'lat': y} )

#J 描画
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

linecol=["darkorange", "green", "red", "blue", "purple", "lightblue"]

ax = [
    plt.subplot(3,1,1),
    plt.subplot(3,1,2),
    plt.subplot(3,1,3),
]

ylim = [ [-0.1, 0.2], [-0.10, 0.14], [-0.10, 0.14] ]
yint = [ 0.05, 0.02, 0.02 ]

for n in range(3):
    DS_ref.tauuo.mean(dim='time',skipna=False).isel(basin=n).plot(ax=ax[n], color='palegreen', label='SCOW',linewidth=6)
    DS.omip1.mean(dim='model',skipna=False).isel(basin=n).plot(ax=ax[n], color='darkred', label='OMIP1', linewidth=2)
    ax[n].fill_between(x=DS.omip1["lat"],
                       y1=DS.omip1.mean(dim='model',skipna=False).isel(basin=n)-DS.omip1.std(dim='model',skipna=False).isel(basin=n),
                       y2=DS.omip1.mean(dim='model',skipna=False).isel(basin=n)+DS.omip1.std(dim='model',skipna=False).isel(basin=n),
                       alpha=0.5,facecolor='lightcoral')
    DS.omip2.mean(dim='model',skipna=False).isel(basin=n).plot(ax=ax[n], color='darkblue', label='OMIP2',linewidth=2)
    ax[n].fill_between(x=DS.omip2["lat"],
                       y1=DS.omip2.mean(dim='model',skipna=False).isel(basin=n)-DS.omip2.std(dim='model',skipna=False).isel(basin=n),
                       y2=DS.omip2.mean(dim='model',skipna=False).isel(basin=n)+DS.omip2.std(dim='model',skipna=False).isel(basin=n),
                       alpha=0.5,facecolor='lightblue')

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


plt.subplots_adjust(left=0.12,right=0.98,top=0.93,bottom=0.05,hspace=0.25)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
