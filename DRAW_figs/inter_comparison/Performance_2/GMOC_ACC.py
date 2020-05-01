# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["darkred", "darkblue"]
miparcs = ["orange", "lightblue"]

markinfo = json.load(open('../json/inst_color_style.json'))

mldcirc=[]
num_models=[]

infl1 = "./csv_spin/circulation_index_omip1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_spin/circulation_index_omip2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
dfm=df.drop('Z-STD',axis=0)
dfm.rename(index={'Z-MMM': 'MMM'}, inplace=True)

fig = plt.figure(figsize=(11,8))
fig.suptitle("ACC vs GMOC", size='x-large')

axes=fig.add_subplot(1,2,1)

for index, row in dfm.iterrows():
    print(-row['GMOC-OMIP1'],row['ACC-OMIP1'])
    btm=-row['GMOC-OMIP1']
    side=row['ACC-OMIP1']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='lightblue',marker=markers[mi],s=50,label=name)

r = dfm[['GMOC-OMIP1','ACC-OMIP1']].corr()
print(r)
x_np = -dfm[['GMOC-OMIP1']].values
y_np = dfm['ACC-OMIP1'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0,19,1)
y=np.arange(100,210,10)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(1,190,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('OMIP1 GMOC vs ACC')
axes.set_xlabel('OMIP1-GMOC [Sv]')
axes.set_ylabel('OMIP1-ACC [Sv]')
axes.set_xlim(0,18.1)
axes.set_xticks((np.arange(0,18,2)))
axes.set_ylim(100,200.1)
axes.set_yticks((np.arange(100,200,10)))
axes.grid(b=True,which='major',axis='both')

axes=fig.add_subplot(1,2,2)

for index, row in dfm.iterrows():
    print(-row['GMOC-OMIP2'],row['ACC-OMIP2'])
    btm=-row['GMOC-OMIP2']
    side=row['ACC-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='lightblue',marker=markers[mi],s=50,label=name)

r = dfm[['GMOC-OMIP2','ACC-OMIP2']].corr()
print(r)
x_np = -dfm[['GMOC-OMIP2']].values
y_np = dfm['ACC-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0,19,1)
y=np.arange(100,210,10)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(1,190,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('OMIP2 GMOC vs ACC')
axes.set_xlabel('OMIP2-GMOC [Sv]')
axes.set_ylabel('OMIP2-ACC [Sv]')
axes.set_xlim(0,18.1)
axes.set_xticks((np.arange(0,18,2)))
axes.set_ylim(100,200.1)
axes.set_yticks((np.arange(100,200,10)))
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))
axes.legend(bbox_to_anchor=(1.01,0.8),loc='upper left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.85,hspace=0.30)

#fig.tight_layout()

plt.savefig('fig/GMOC_ACC.png')
plt.show()
