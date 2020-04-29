# -*- coding: utf-8 -*-
import sys
import json
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from distutils.util import strtobool
import netCDF4
from netCDF4 import Dataset, num2date

#

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' 1 (OMIP1) or 2 (OMIP2) or 3 (MMM)')
    sys.exit()

if (int(sys.argv[1]) == 3):
    metainfo = [ json.load(open("./json/vat_omip1_full.json")),
                 json.load(open("./json/vat_omip2_full.json")) ]
    suptitle = 'Vertically averaged temperature (Multi Model Mean)'
    outfile = './fig/FigS1_vat_MMM.png'

else:
    metainfo = [ json.load(open("./json/vat_omip1.json")),
                 json.load(open("./json/vat_omip2.json")) ]
    suptitle = 'Vertically averaged temperature (OMIP-' + sys.argv[1] +')'
    outfile = './fig/FigS1_vat_OMIP'+sys.argv[1]+'.png'

lineinfo = json.load(open('../json/inst_color_style.json'))

title_list = [ "(a) 0 - 700 m", "(b) 0 - 2000 m", "(c) 2000 m - bottom", "(d) 0 m - bottom" ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

var_list = [ "thetaoga_700", "thetaoga_2000", "thetaoga_2000_bottom", "thetaoga_all" ]
volume_list = np.array([ 2.338e17, 6.216e17, 7.593e17, 1.381e18 ])
degC_to_ZJ = volume_list * 3.99e3 * 1.036e3 * 1.0e-21


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


#J データ読込 (n=0: OMIP1, n=1: OMIP2)
DS = []

lincol = []
linsty = []
nummodel = []

for n in range(2):

    d = np.full( (len(var_list),len(model_list[n]),len(time[n])), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    coltmp = []
    stytmp = []

    nvar = 0
    for var in var_list:

        nmodel = 0
        for model in model_list[n]:

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
            vname = metainfo[n][model][var]['varname']
            factor = float(metainfo[n][model][var]['factor'])
            infile = path + '/' + fname

            print(infile,vname)

            if (n == 1) and (model == "FSU-HYCOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
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

            elif (n == 1) and (model == "GFDL-MOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_fsu = np.full( len(time[n]), np.nan )
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

            
            elif (n == 0) and (model == "MIROC-COCO4.9" ):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_coco = np.full( len(time[n]), np.nan )
                d_coco[0:62*5] = d_tmp[0:62*5] * factor
                d[nvar,nmodel] = d_coco

            elif (n == 0) and (model == "GFDL-MOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_fsu = np.full( len(time[n]), np.nan )
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

                #print (d_fsu)
                d[nvar,nmodel] = d_fsu

            else:

                if multidata:
                    if ( model == 'Kiel-NEMO' ):
                        DS_read = xr.open_mfdataset(infile,concat_dim='time_counter',decode_times=False)
                    elif ( model == 'EC-Earth3-NEMO' ):
                        DS_read = xr.open_mfdataset(infile,concat_dim='time',decode_times=False)
                    else:
                        DS_read = xr.open_mfdataset(infile,decode_times=False)

                else:
                    DS_read = xr.open_dataset(infile,decode_times=False)

                #d[nvar,nmodel] = DS_read[vname].values * factor * degC_to_ZJ[nvar]
                if ( model == 'Kiel-NEMO' ):
                    d[nvar,nmodel] = DS_read[vname].values[:,0,0] * factor
                elif ( model == 'EC-Earth3-NEMO' ):
                    if (var == 'thetaoga_all'):
                        d[nvar,nmodel] = DS_read[vname].values * factor
                    else:
                        d[nvar,nmodel] = DS_read[vname].values[:,0,0] * factor
                else:
                    d[nvar,nmodel] = DS_read[vname].values * factor

                #if (n == 0):
                #    print(d[nvar,nmodel,-6:-1])
                #    d[nvar,nmodel] = d[nvar,nmodel] - (d[nvar,nmodel,-6:-1]).mean()
                #
                #if (n == 1):
                #    print(d[nvar,nmodel,-15:-9])
                #    d[nvar,nmodel] = d[nvar,nmodel] - (d[nvar,nmodel,-15:-9]).mean()
            nmodel += 1

        nvar += 1
            
    #J サイクル間に NaN データ挿入
    d_new = np.concatenate(
        [d, np.tile(varnan,(len(var_list),len(model_list[n]),1))],
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

    lincol += [coltmp]
    linsty += [stytmp]
    nummodel += [nmodel]
    

#J 描画
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )
ax = [ plt.subplot(4,1,1),
       plt.subplot(4,1,2),
       plt.subplot(4,1,3),
       plt.subplot(4,1,4) ]
#ax_r = [ ax[2].twinx(),
#         ax[3].twinx(), ]

#for n in range(2):

n = int(sys.argv[1]) - 1

if ( n < 2 ):
    if len(model_list[n]) > 1:
        nf = 0
        for var in var_list:
            nmodel = 0
            for model in model_list[n]:
                linecol=lincol[n][nmodel]
                linesty=linsty[n][nmodel]
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=1,linestyle=linesty)
                nmodel += 1


            if (nf == 0): 
                ax[nf].legend(bbox_to_anchor=(1.02,1.0),loc='upper left')
            nf += 1

    else:
        nf = 0
        for var in var_list:
            print(DS[n][var])
            DS[n][var].plot(ax=ax[nf])
            #        DS[n]['thetaoga'].plot(ax=ax[2+n],color='red')
            #        DS[n]['thetaoga'].plot(ax=ax_r[n],color='blue')
            nf += 1

else:
    nf = 0
    for var in var_list:
        DS[0][var].mean(dim='model').plot.line(x='time',ax=ax[nf],label='OMIP1',color='darkred')
        ax[nf].fill_between(x=DS[0]['time'],
                            y1=DS[0][var].min(dim='model'),
                            y2=DS[0][var].max(dim='model'),
                            alpha=0.5, facecolor='lightcoral')
        DS[1][var].mean(dim='model').plot.line(x='time',ax=ax[nf],label='OMIP2',color='darkblue')
        ax[nf].fill_between(x=DS[1]['time'],
                            y1=DS[1][var].min(dim='model'),
                            y2=DS[1][var].max(dim='model'),
                            alpha=0.5, facecolor='lightblue')

        if (nf == 0):
            ax[nf].legend(bbox_to_anchor=(1.02,1.0),loc='upper left')
        nf += 1
    
#ax[2].yaxis.label.set_color('red')
#ax[3].yaxis.label.set_color('red')
#ax_r[0].yaxis.label.set_color('blue')
#ax_r[1].yaxis.label.set_color('blue')

ylim = [ [9.6, 11.4], [5.5, 6.8], [0.6, 2.2], [3.0, 4.2] ]
ytick = [ np.linspace(9.6, 11.4, 10),
          np.linspace(5.5,  6.8, 14),
          np.linspace(0.6,  2.2,  9),
          np.linspace(3.0,  4.2, 13), ]

#ylim = [ [-1000, 1000], [-1000, 1000], [-1000, 1000], [-1500, 1500] ]
#ytick = [ np.linspace(-1000, 1000, 11),
#          np.linspace(-1000, 1000, 11),
#          np.linspace(-1000, 1000, 11),
#          np.linspace(-1500, 1500, 11), ]
#ytickr= [ np.linspace(12,28,9),
#          np.linspace(12,28,9), ]

for m in range(4):
    ax[m].set_title(title_list[m],{'fontsize':10,'verticalalignment':'top'})
    ax[m].set_xlim(1592,2018)
    ax[m].set_xticks(np.arange(1592,2018.1,71))
    if ( m == 3 ):
        ax[m].set_xlabel('year',fontsize=10)
    else:
        ax[m].set_xlabel('',fontsize=10)

    ax[m].grid()
    ax[m].set_ylim(ylim[m])
    ax[m].set_yticks(ytick[m])
    ax[m].set_ylabel(r'$^{\circ}$C',fontsize=10)

plt.subplots_adjust(left=0.10,right=0.75,top=0.93,bottom=0.05,hspace=0.2)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
