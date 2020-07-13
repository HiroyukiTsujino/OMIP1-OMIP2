# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["darkred", "darkblue"]
miparcs = ["orange", "lightblue"]
titles = {"interannual":"interannual variability", "monclim":"monthly climatology"}

zosstats=[]
num_models=[]

for intvl in ['annual','monthly']:
    for omip in ['omip1', 'omip2']:
        df=pd.read_csv('csv_zos/zos_filter_rbar_sites_' + intvl + '_' + omip + '_corrected.csv',index_col=0)
        zosstats += [df]
        num_models += [len(df.index)]

print(num_models)

#legcol = dict(interannual=[0.4,0.7],
#              monclim=[0.9,0.7])

fig = plt.figure(figsize=(8,10))
fig.suptitle("Sea Surface Height (zos)", size='x-large')

axes=fig.add_subplot(2,1,1)
#zosstats[0].plot.scatter(x='SITES',y='RBAR',c='darkred',ax=axes,xlim=[5,50],ylim=[0.5,0.9],title='annual wrt long-term mean')
#zosstats[1].plot.scatter(x='SITES',y='RBAR',c='darkblue',ax=axes,xlim=[5,50],ylim=[0.5,0.9],marker=markers)

i = 0
for index, row in zosstats[0].iterrows():
    print(row[0],row[1])
    sites=row[1]
    rbar=row[0]
    name=index + '-omip1'
    axes.scatter(sites,rbar,c='red',marker=markers[i],s=50,label=name)
    i += 1

i = 0
for index, row in zosstats[1].iterrows():
    print(row[0],row[1])
    sites=row[1]
    rbar=row[0]
    name=index + '-omip2'
    axes.scatter(sites,rbar,c='darkblue',marker=markers[i],s=50,label=name)
    i += 1

axes.set_title('(a) Time series of annual anomaly wrt long-term mean (1993-2009)')
axes.set_xlabel('Normalized error in overall time means (SITES)')
axes.set_ylabel('Space-time pattern similarity (RBAR)')
axes.set_xlim(5,45)
axes.set_ylim(0.5,0.9)
axes.legend(bbox_to_anchor=(1.02,0.98),loc='upper left',fontsize=8)

axes=fig.add_subplot(2,1,2)
#zosstats[2].plot.scatter(x='SITES',y='RBAR',c='darkred',ax=axes,xlim=[3,15],ylim=[0.5,0.9],title='monthly wrt long-term mean')
#zosstats[3].plot.scatter(x='SITES',y='RBAR',c='darkblue',ax=axes,xlim=[3,15],ylim=[0.5,0.9])
i = 0
for index, row in zosstats[2].iterrows():
    print(row[0],row[1])
    sites=row[1]
    rbar=row[0]
    name=index + '-omip1'
    axes.scatter(sites,rbar,c='red',marker=markers[i],s=50,label=name)
    i += 1

i = 0
for index, row in zosstats[3].iterrows():
    print(row[0],row[1])
    sites=row[1]
    rbar=row[0]
    name=index + '-omip2'
    axes.scatter(sites,rbar,c='darkblue',marker=markers[i],s=50,label=name)
    i += 1

axes.set_title('(b) Time series of monthly anomaly wrt long-term mean (1993-2009)')
axes.set_xlabel('Normalized error in overall time means (SITES)')
axes.set_ylabel('Space-time pattern similarity (RBAR)')
axes.set_xlim(1,15)
axes.set_ylim(0.5,0.9)
    
plt.subplots_adjust(left=0.1,right=0.73,top=0.92,bottom=0.1,hspace=0.3)

#fig.tight_layout()

outfile = 'fig/zos_sites_rbar'

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
