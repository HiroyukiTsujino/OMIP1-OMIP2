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

fig = plt.figure(figsize=(18,6))
fig.suptitle("Southern Hemisphere", size='x-large')

axes=fig.add_subplot(1,3,1)


i=0
for index, row in mldcirc[0].iterrows():
    print(row['SH-MLD'],row['GMOC'])
    btm=row['SH-MLD']
    rbar=-row['GMOC']
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    name=index + '-omip1'
    axes.scatter(btm,rbar,c='red',marker=markers[mi],s=50,label=name)
    
i=0
for index, row in mldcirc[1].iterrows():
    print(row['SH-MLD'],row['GMOC'])
    btm=row['SH-MLD']
    rbar=-row['GMOC']
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    name=index + '-omip2'
    axes.scatter(btm,rbar,c='darkblue',marker=markers[mi],s=50,label=name)
    i += 1

mld1=mldcirc[0]['SH-MLD']
gmoc1=mldcirc[0]['GMOC']
mld2=mldcirc[1]['SH-MLD']
gmoc2=mldcirc[1]['GMOC']

nm=0
for idx in mldcirc[1].index:
    #print(idx,gmoc1[nm],acc1[nm],gmoc2[nm],acc2[nm])
    a = [mld1[nm],mld2[nm]]
    b = [-gmoc1[nm],-gmoc2[nm]]
    print(idx,a,b)
    axes.plot(a,b,color='black',linewidth=0.5)
    nm += 1

axes.set_title('(a) MLD - GMOC')
axes.set_xlabel('MLD')
axes.set_ylabel('GMOC')
axes.set_xlim(0,5000)
axes.set_ylim(-1,20)
axes.legend(bbox_to_anchor=(4.0,0.9))

#plt.subplots_adjust(left=0.1,right=0.75)

axes=fig.add_subplot(1,3,2)

for index, row in mldcirc[0].iterrows():
    print(row['GMOC'],row['ACC'])
    btm=-row['GMOC']
    side=row['ACC']
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    name=index + '-omip1'
    axes.scatter(btm,side,c='red',marker=markers[mi],s=50,label=name)
    
i = 0
for index, row in mldcirc[1].iterrows():
    print(row['GMOC'],row['ACC'])
    btm=-row['GMOC']
    side=row['ACC']
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    name=index + '-omip2'
    axes.scatter(btm,side,c='darkblue',marker=markers[mi],s=50,label=name)
    i += 1

gmoc1=mldcirc[0]['GMOC']
acc1=mldcirc[0]['ACC']
gmoc2=mldcirc[1]['GMOC']
acc2=mldcirc[1]['ACC']

nm=0
for idx in mldcirc[1].index:
    #print(idx,gmoc1[nm],acc1[nm],gmoc2[nm],acc2[nm])
    a = [-gmoc1[nm],-gmoc2[nm]]
    b = [acc1[nm],acc2[nm]]
    axes.plot(a,b,color='black',linewidth=0.5)
    nm += 1
    
axes.set_title('(b) GMOC-ACC')
axes.set_xlabel('GMOC')
axes.set_ylabel('ACC')
axes.set_xlim(-1,20)
axes.set_ylim(100,200)

axes=fig.add_subplot(1,3,3)
i = 0
for index, row in mldcirc[0].iterrows():
    print(row['SH-MLD'],row['ACC'])
    btm=row['SH-MLD']
    side=row['ACC']
    name=index + '-omip1'
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    axes.scatter(btm,side,c='red',marker=markers[mi],s=50,label=name)
    i = i + 1

i = 0
for index, row in mldcirc[1].iterrows():
    print(row['SH-MLD'],row['ACC'])
    btm=row['SH-MLD']
    side=row['ACC']
    name=index + '-omip2'
    if (index == 'MRICOM'):
        index = 'MRI.COM'
    mi = int(markinfo[index]["marker"])
    axes.scatter(btm,side,c='darkblue',marker=markers[mi],s=50,label=name)
    i = i + 1

mld1=mldcirc[0]['SH-MLD']
acc1=mldcirc[0]['ACC']
mld2=mldcirc[1]['SH-MLD']
acc2=mldcirc[1]['ACC']

nm=0
for idx in mldcirc[1].index:
    #print(idx,gmoc1[nm],acc1[nm],gmoc2[nm],acc2[nm])
    a = [mld1[nm],mld2[nm]]
    b = [acc1[nm],acc2[nm]]
    print(idx,a,b)
    axes.plot(a,b,color='black',linewidth=0.5)
    nm += 1

axes.set_title('(c) MLD-ACC')
axes.set_xlabel('MLD')
axes.set_ylabel('ACC')

axes.set_xlim(0,5000)
axes.set_ylim(100,200)
#axes.legend(bbox_to_anchor=(1.35,1.0))
    
#plt.subplots_adjust(left=0.1,right=0.75,top=0.9,bottom=0.1,hspace=0.3)
plt.subplots_adjust(left=0.10,right=0.85,bottom=0.15,top=0.90)

#fig.tight_layout()

plt.savefig('fig/mld_circ_sh.png')
plt.show()
