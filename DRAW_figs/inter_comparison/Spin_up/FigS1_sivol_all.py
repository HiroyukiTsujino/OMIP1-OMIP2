# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../python")
import json
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from distutils.util import strtobool
import netCDF4
from netCDF4 import Dataset, num2date
from uncertain_Wakamatsu import uncertain

#########################################

title_list = [ "(a) OMIP1 NH", "(b) OMIP2 NH", "(c) MMM NH",
               "(d) OMIP1 SH", "(e) OMIP2 SH", "(f) MMM SH" ]

suptitle = 'Sea ice volume'
outfile = './fig/FigS1_sivol_ALL'

metainfo = [ json.load(open("./json/sivol_omip1.json")),
             json.load(open("./json/sivol_omip2.json")) ]

metainfo_full = [ json.load(open("./json/sivol_omip1_full.json")),
                  json.load(open("./json/sivol_omip2_full.json")) ]

lineinfo = json.load(open('../json/inst_color_style.json'))

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]
model_full_list = [ metainfo_full[0].keys(), metainfo_full[1].keys() ]

var_list = [ "sivoln", "sivols" ]

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.array([np.linspace(1948,2009,62)]*6),
         np.array([np.linspace(1958,2018,61)]*6) ]
for n in range(2):
    for i in range(6):
        time[n][i] = time[n][i] - (5-i)*71
timenan = [ time[1][:,60], time[0][:,0] ]
time[0] = time[0].reshape(6*62)
time[1] = time[1].reshape(6*61)


#J 間をあけるための Dataset 作成
varnan = np.full(6,np.nan)
DS_nan = [
    xr.Dataset({'var':(['time'],varnan),},coords={'time':timenan[0]}), 
    xr.Dataset({'var':(['time'],varnan),},coords={'time':timenan[1]})
]


#J 単一モデル用 dummy DS (json にエントリーがない場合に使用)
d_dummy = [ np.full( len(time[0]), np.nan ),
            np.full( len(time[1]), np.nan ) ]


mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)

#J データ読込 (n=0: OMIP1, n=1: OMIP2)
DS = []
DS_full = []

lincol = []
linsty = []
nummodel = []

for n in range(2):
    d = np.full( (len(var_list),len(model_list[n]),len(time[n])), np.nan )
    d_full = np.full( (len(var_list),len(model_full_list[n]),len(time[n])), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    coltmp = []
    stytmp = []

    nvar = 0
    for var in var_list:

        nmodel = 0
        nmodel_full = 0
        for model in model_list[n]:

            add_to_full = 'no'
            for model_full in model_full_list[n]:
                if (model == model_full):
                    add_to_full = 'yes'
                    break

            print(model + ' Added to full list ' + add_to_full)

            try:
                multidata = strtobool(metainfo[n][model]['multidata'])
            except:
                #J 単一モデル用エラー処理 (json にエントリーがない場合)
                d[nvar,nmodel] = d_dummy[n]
                continue

            if (nvar == 0):
                coltmp +=[lineinfo[model]["color"]]
                stytmp +=[lineinfo[model]["style"]]

            path = metainfo[n][model][var]['path']
            fname = metainfo[n][model][var]['fname']
            factor = float(metainfo[n][model][var]['factor'])
            infile = path + '/' + fname

            if (n == 1) and (model == "FSU-HYCOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[var][:]
                nc.close()

                d_fsu = np.full( len(time[n]), np.nan )
                for cyc in range(5):
                    isto = 0 + cyc * 61
                    iedo = isto + 57
                    istf = 0 + cyc * 58
                    iedf = istf + 57
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                cyc = 5
                isto = 0 + cyc * 61
                iedo = isto + 61
                istf = 0 + cyc * 58
                iedf = istf + 61
                print (isto, iedo)
                print (istf, iedf)
                d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor
                #print (d_fsu)
                d[nvar,nmodel] = d_fsu
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu

            elif (n == 0) and (model == "MIROC-COCO4.9" ):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[var][:]
                nc.close()

                d_coco = np.full( len(time[n]), np.nan )
                d_coco[0:62*5] = d_tmp[0:62*5] * factor
                d[nvar,nmodel] = d_coco
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_coco

            elif (model == "GFDL-MOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp_mon = nc.variables[var][:]
                nc.close()

                if ( n == 0 ) :
                    num_yr = 362
                else:
                    num_yr = 363

                d_tmp = np.array(np.zeros(num_yr),dtype=np.float64)
                nd = 0
                for yr in range(0,num_yr):
                    wgt_tmp = 0
                    for mn in range(1,13):
                        wgt_tmp = wgt_tmp + mon_days[mn-1]
                        d_tmp[yr] = d_tmp[yr] + d_tmp_mon[nd] * mon_days[mn-1]
                        nd += 1

                    d_tmp[yr] = d_tmp[yr] / wgt_tmp

                d_fsu = np.full( len(time[n]), np.nan )

                if ( n == 0 ):
                    for cyc in range(0,5):
                        isto = 0 + cyc * 62
                        iedo = isto + 59
                        istf = 0 + cyc * 60
                        iedf = istf + 59
                        print (cyc, isto, iedo)
                        print (cyc, istf, iedf)
                        d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                        for cyc in range(5,6):
                            isto = 0 + cyc * 62
                            iedo = isto + 61
                            istf = 0 + cyc * 62 - 10
                            iedf = istf + 61
                            print (cyc, isto, iedo)
                            print (cyc, istf, iedf)
                            d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                else:
                    for cyc in range(0,3):
                        isto = 0 + cyc * 61
                        iedo = isto + 59
                        istf = 0 + cyc * 60
                        iedf = istf + 59
                        print (isto, iedo)
                        print (istf, iedf)
                        d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                        for cyc in range(3,6):
                            isto = 0 + cyc * 61
                            iedo = isto + 60
                            istf = 0 + cyc * 61 - 3
                            iedf = istf + 60
                            print (isto, iedo)
                            print (istf, iedf)
                            d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                            #print (d_fsu)

                d[nvar,nmodel] = d_fsu
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu

            else:

                if multidata:
                    if ( model == 'Kiel-NEMO' ):
                        DS_read = xr.open_mfdataset(infile,concat_dim='time_counter',decode_times=False)
                    else:
                        DS_read = xr.open_mfdataset(infile,concat_dim='time',decode_times=False)

                else:
                    DS_read = xr.open_dataset(infile,decode_times=False)

                #J AWI-FESOM 用特別処理 (年平均計算)
                if model == "AWI-FESOM" and (var == 'sivoln' or var == 'sivols'):
                    d[nvar,nmodel] = np.average(
                        DS_read[var].values.reshape(math.ceil(DS_read[var].size/12),12),
                        weights=[31,28,31,30,31,30,31,31,30,31,30,31],
                        axis=1 ) * factor
                    if (add_to_full == 'yes'):
                        d_full[nvar,nmodel_full] = np.average(
                        DS_read[var].values.reshape(math.ceil(DS_read[var].size/12),12),
                        weights=[31,28,31,30,31,30,31,31,30,31,30,31],
                        axis=1 ) * factor
                #J その他
                else:
                    if ( model == 'Kiel-NEMO' ):
                        d[nvar,nmodel] = DS_read[var].values[:,0,0] * factor
                        if (add_to_full == 'yes'):
                            d_full[nvar,nmodel_full] = DS_read[var].values[:,0,0] * factor
                    else:
                        d[nvar,nmodel] = DS_read[var].values * factor
                        if (add_to_full == 'yes'):
                            d_full[nvar,nmodel_full] = DS_read[var].values * factor

                    
            #print (d[nvar,nmodel])

            nmodel += 1
            if (add_to_full == 'yes'):
                nmodel_full += 1

            print(nmodel, nmodel_full)
                
        nvar += 1

    #J サイクル間に NaN データ挿入
    d_new = np.concatenate(
        [d, np.tile(varnan,(len(var_list),len(model_list[n]),1))],
        axis = 2 )

    d_full_new = np.concatenate(
        [d_full, np.tile(varnan,(len(var_list),len(model_full_list[n]),1))],
        axis = 2 )

    time_new = np.concatenate( [time[n], timenan[n]] )

    #J xarray Dataset 再作成
    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new } ).sortby('time')

    DS += [DS_tmp]

    ######

    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d_full_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new } ).sortby('time')

    DS_full += [DS_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    nummodel += [nmodel]
    
    #nm = 0
    #for model in model_list[n]:
    #    print('----CHECK----')
    #    print(model,'sivoln','omip'+str(n+1))
    #    print(DS_tmp['sivoln'].isel(model=nm))
    #    nm +=1
        
#--------------------------------------------------
# omip1 - omip2

print("Compute uncertainty of the difference omip2-omip1")

ystr = 1980
yend = 2009
nyr = yend - ystr + 1
nmod = len(model_list[0])

d_sivoln  = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)
d_sivols  = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)

nmodel = 0
for model in model_list[0]:
    if (model == 'MIROC-COCO4.9'):
        sivoln_omip1 = DS[0]['sivoln'].sel(model=nmodel,time=slice(1909,1938)).values
        sivols_omip1 = DS[0]['sivols'].sel(model=nmodel,time=slice(1909,1938)).values
        sivoln_omip2 = DS[1]['sivoln'].sel(model=nmodel,time=slice(1909,1938)).values
        sivols_omip2 = DS[1]['sivols'].sel(model=nmodel,time=slice(1909,1938)).values
    else:
        sivoln_omip1 = DS[0]['sivoln'].sel(model=nmodel,time=slice(1980,2009)).values
        sivols_omip1 = DS[0]['sivols'].sel(model=nmodel,time=slice(1980,2009)).values
        sivoln_omip2 = DS[1]['sivoln'].sel(model=nmodel,time=slice(1980,2009)).values
        sivols_omip2 = DS[1]['sivols'].sel(model=nmodel,time=slice(1980,2009)).values

    d_sivoln[nmodel,:] = sivoln_omip2[:] - sivoln_omip1[:]
    d_sivols[nmodel,:] = sivols_omip2[:] - sivols_omip1[:]

    nmodel += 1

num_bootstraps = 10000
factor_5pct = 1.64  # 5-95%

dout_sivoln = uncertain(d_sivoln, "sivoln", nmod, nyr, num_bootstraps )
zval = dout_sivoln["mean"] / dout_sivoln["std"]
print(dout_sivoln)
print("z-value = ",zval)

dout_sivols = uncertain(d_sivols, "sivols", nmod, nyr, num_bootstraps )
zval = dout_sivols["mean"] / dout_sivols["std"]
print(dout_sivols)
print("z-value = ",zval)

#-------------------------------------------------------------------------
#J 描画
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )
ax = [ plt.subplot(2,3,1),
       plt.subplot(2,3,2),
       plt.subplot(2,3,3),
       plt.subplot(2,3,4),
       plt.subplot(2,3,5),
       plt.subplot(2,3,6) ]

nv = 0
for var in var_list:
    for omip in range(2):
        nf = 3 * nv + omip 
        nmodel = 0
        for model in model_list[omip]:
            print(nf,nmodel,var,model)
            linecol=lincol[omip][nmodel]
            linesty=linsty[omip][nmodel]
            if (omip == 0 and nv == 0):
                DS[omip][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=0.5,linestyle=linesty)
            else:
                DS[omip][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],color=linecol,linewidth=0.5,linestyle=linesty)

            nmodel += 1

        if (omip == 0 and nv == 0):
            leg = ax[nf].legend(bbox_to_anchor=(3.45,0.6),loc='upper left')
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

    nv += 1

nv = 0
for var in var_list:
    nf = 3 * nv + 2
    if (nv == 0): 
        DS_full[0][var].mean(dim='model').plot(ax=ax[nf],color='darkred',label='OMIP1-MMM')
    else:
        DS_full[0][var].mean(dim='model').plot(ax=ax[nf],color='darkred')

    ax[nf].fill_between(x=DS_full[0]['time'],
                        y1=DS_full[0][var].min(dim='model'),
                        y2=DS_full[0][var].max(dim='model'),
                        alpha=0.5, facecolor='lightcoral')
    if (nv == 0): 
        DS_full[1][var].mean(dim='model').plot(ax=ax[nf],color='darkblue',label='OMIP2-MMM')
    else:
        DS_full[1][var].mean(dim='model').plot(ax=ax[nf],color='darkblue')

    ax[nf].fill_between(x=DS_full[1]['time'],
                        y1=DS_full[1][var].min(dim='model'),
                        y2=DS_full[1][var].max(dim='model'),
                        alpha=0.5, facecolor='lightblue')
    if (nv == 0):
        ax[nf].legend(bbox_to_anchor=(1.05,1.0),loc='upper left')

    nv += 1


ylim = [ [6, 34], [2, 14] ]
ytick = [ np.linspace(6,34,8),
          np.linspace(2,14,7), ]

for nv in range(2):
    for nm in range(3):
        nf = 3 * nv + nm
        ax[nf].set_title(title_list[nf],{'fontsize':10,'verticalalignment':'top'})
        ax[nf].tick_params(labelsize=9)
        ax[nf].set_xlim(1592,2018)
        ax[nf].set_xticks(np.arange(1592,2018.1,71))
        if ( nv == 1 ):
            ax[nf].set_xlabel('year',fontsize=10)
        else:
            ax[nf].set_xlabel('',fontsize=10)

        ax[nf].grid()
        ax[nf].set_ylim(ylim[nv])
        ax[nf].set_yticks(ytick[nv])
        if (nm == 0):
            ax[nf].set_ylabel(r'$10^{3} \mathrm{km}^{3}$',fontsize=10)
        else:
            ax[nf].set_ylabel('')

#J titleとx軸ラベルが重なるのを防ぐ
#J tight_layout は suptitle を考慮しないので、上側を少しあける
plt.subplots_adjust(left=0.07,right=0.82,bottom=0.07,top=0.92)
#plt.tight_layout(rect=[0,0,1,0.95])

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0, dpi=300)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()
