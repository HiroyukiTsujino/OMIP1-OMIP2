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
fig.suptitle("Circulation index", fontsize=18)

axes=fig.add_subplot(2,2,1)

for index, row in dfm.iterrows():
    print(row['AMOC-OMIP1'],row['AMOC-OMIP2'])
    btm=row['AMOC-OMIP1']
    side=row['AMOC-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = dfm[['AMOC-OMIP1','AMOC-OMIP2']].corr()
print(r)
x_np = dfm[['AMOC-OMIP1']].values
y_np = dfm['AMOC-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(10,23,1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(11,20,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('AMOC')
axes.set_xlabel('OMIP1_AMOC [Sv]')
axes.set_ylabel('OMIP2_AMOC [Sv]')
axes.set_xlim(10,22)
axes.set_ylim(10,22)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,2)

for index, row in dfm.iterrows():
    print(row['GMOC-OMIP1'],row['GMOC-OMIP2'])
    btm=-row['GMOC-OMIP1']
    side=-row['GMOC-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = dfm[['GMOC-OMIP1','GMOC-OMIP2']].corr()
print(r)
x_np = -dfm[['GMOC-OMIP1']].values
y_np = -dfm['GMOC-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0,19,1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(1,15,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('GMOC')
axes.set_xlabel('OMIP1-GMOC [Sv]')
axes.set_ylabel('OMIP2-GMOC [Sv]')
axes.set_xlim(0,18.1)
axes.set_xticks((np.arange(0,18,2)))
axes.set_ylim(0,18.1)
axes.set_yticks((np.arange(0,18,2)))
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

axes=fig.add_subplot(2,2,3)

for index, row in dfm.iterrows():
    print(row['ACC-OMIP1'],row['ACC-OMIP2'])
    btm=row['ACC-OMIP1']
    side=row['ACC-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = dfm[['ACC-OMIP1','ACC-OMIP2']].corr()
print(r)
x_np = dfm[['ACC-OMIP1']].values
y_np = dfm['ACC-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(100,210,10)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(110,180,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('ACC')
axes.set_xlabel('OMIP1-ACC [Sv]')
axes.set_ylabel('OMIP2-ACC [Sv]')
axes.set_xlim(100,200)
axes.set_ylim(100,200)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,4)

for index, row in dfm.iterrows():
    print(row['ITF-OMIP1'],row['ITF-OMIP2'])
    btm=-row['ITF-OMIP1']
    side=-row['ITF-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = dfm[['ITF-OMIP1','ITF-OMIP2']].corr()
print(r)
x_np = -dfm[['ITF-OMIP1']].values
y_np = -dfm['ITF-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(5,21,1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(6,17,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('ITF')
axes.set_xlabel('OMIP1-ITF [Sv]')
axes.set_ylabel('OMIP2-ITF [Sv]')
axes.set_xlim(5,20)
axes.set_ylim(5,20)
axes.grid(b=True,which='major',axis='both')
axes.legend(bbox_to_anchor=(1.01,0.8),loc='lower left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.90,hspace=0.30)

#fig.tight_layout()

plt.savefig('fig/Circulation.png')
plt.show()
