# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["darkred", "darkblue"]
miparcs = ["orange", "lightblue"]
titles = {"interannual":"interannual variability", "monclim":"monthly climatology"}

markinfo = json.load(open('../json/inst_color_style.json'))

mldcirc=[]
num_models=[]

for omip in ["omip1", "omip2"]:
    infl1 = "./csv_circulation/circulation_index_" + str(omip) + ".csv"
    df1=pd.read_csv(infl1,index_col=0)
    infl2 = "./csv_mld/winter_mld_" + str(omip) + ".csv"
    df2=pd.read_csv(infl2,index_col=0)
    df = pd.merge(df1,df2,left_index=True,right_index=True)
    print(df)
    mldcirc += [df]
    num_models += [len(df.index)]

print(num_models)

fig = plt.figure(figsize=(10,10))
fig.suptitle("Southern Hemisphere", size='x-large')

axes=fig.add_subplot(1,1,1)


for index, row in mldcirc[0].iterrows():
    print(row['NH-MLD'],row['AMOC'])
    btm=row['NH-MLD']
    side=row['AMOC']
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    name=index + '-omip1'
    axes.scatter(btm,side,c='red',marker=markers[mi],s=50,label=name)
    
for index, row in mldcirc[1].iterrows():
    print(row['NH-MLD'],row['AMOC'])
    btm=row['NH-MLD']
    side=row['AMOC']
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    name=index + '-omip2'
    axes.scatter(btm,side,c='darkblue',marker=markers[mi],s=50,label=name)

mld1=mldcirc[0]['NH-MLD']
amoc1=mldcirc[0]['AMOC']
mld2=mldcirc[1]['NH-MLD']
amoc2=mldcirc[1]['AMOC']

nm=0
for idx in mldcirc[1].index:
    a = [mld1[nm],mld2[nm]]
    b = [amoc1[nm],amoc2[nm]]
    print(idx,a,b)
    axes.plot(a,b,color='black',linewidth=0.5)
    nm += 1

axes.set_title('MLD - AMOC')
axes.set_xlabel('MLD')
axes.set_ylabel('AMOC')
axes.set_xlim(500,3000)
axes.set_ylim(5,25)
axes.legend(bbox_to_anchor=(1.35,0.9))

plt.subplots_adjust(left=0.10,right=0.75,bottom=0.15,top=0.90)

#fig.tight_layout()

plt.savefig('fig/mld_circ_nh.png')
plt.show()
