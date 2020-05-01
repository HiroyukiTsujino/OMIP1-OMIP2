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

infl1 = "./csv_clim/SST_bias_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/SST_bias_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
#num_models len(df.index)
#print(num_models)

fig = plt.figure(figsize=(11,8))
fig.suptitle("SST and SSS bias", size='x-large')

axes=fig.add_subplot(2,2,1)

for index, row in df.iterrows():
    print(row['OMIP1_rmse'],row['OMIP2_rmse'])
    btm=row['OMIP1_rmse']
    side=row['OMIP2_rmse']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

dfm=df.drop('MMM',axis=0)
print(dfm)
r = dfm.loc[:,['OMIP1_rmse','OMIP2_rmse']].corr()
print(r)
x_np = dfm[['OMIP1_rmse']].values
y_np = dfm['OMIP2_rmse'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0.3,1.1,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(0.4,0.9,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('SST bias rmse')
axes.set_xlabel('OMIP1_rmse [$^{\circ}$C]')
axes.set_ylabel('OMIP2_rmse [$^{\circ}$C]')
axes.set_xlim(0.3,1.0)
axes.set_ylim(0.3,1.0)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,2)

for index, row in df.iterrows():
    print(row['OMIP1_mean'],row['OMIP2_mean'])
    btm = row['OMIP1_mean']
    side = row['OMIP2_mean']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

dfm=df.drop('MMM',axis=0)
print(dfm)
r = dfm.loc[:,['OMIP1_rmse','OMIP2_rmse']].corr()
print(r)
x_np = dfm[['OMIP1_mean']].values
y_np = dfm['OMIP2_mean'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-0.3,0.4,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-0.2,0.2,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('SST bias mean')
axes.set_xlabel('OMIP1_mean [$^{\circ}$C]')
axes.set_ylabel('OMIP2_mean [$^{\circ}$C]')
axes.set_xlim(-0.3,0.3)
axes.set_ylim(-0.3,0.3)
axes.grid(b=True,which='major',axis='both')
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
#axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

infl1 = "./csv_clim/SSS_bias_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/SSS_bias_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
axes=fig.add_subplot(2,2,3)

for index, row in df.iterrows():
    print(row['OMIP1_rmse'],row['OMIP2_rmse'])
    btm=row['OMIP1_rmse']
    side=row['OMIP2_rmse']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

dfm=df.drop(['MMM'],axis=0)
print(dfm)
r = dfm.loc[:,['OMIP1_rmse','OMIP2_rmse']].corr()
print(r)
x_np = dfm[['OMIP1_rmse']].values
y_np = dfm['OMIP2_rmse'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0.2,1.5,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(0.3,1.2,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('SSS bias rmse')
axes.set_xlabel('OMIP1_rmse [psu]')
axes.set_ylabel('OMIP2_rmse [psu]')
axes.set_xlim(0.2,1.4)
axes.set_ylim(0.2,1.4)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,4)

for index, row in df.iterrows():
    print(row['OMIP1_mean'],row['OMIP2_mean'])
    btm=row['OMIP1_mean']
    side=row['OMIP2_mean']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

dfm=df.drop('MMM',axis=0)
#dfm=df.drop(['MMM','Kiel-NEMO'],axis=0)
print(dfm)
r = dfm.loc[:,['OMIP1_mean','OMIP2_mean']].corr()
print(r)
x_np = dfm[['OMIP1_mean']].values
y_np = dfm['OMIP2_mean'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-0.2,0.5,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-0.1,0.3,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('SSS bias mean')
axes.set_xlabel('OMIP1_mean [psu]')
axes.set_ylabel('OMIP2_mean [psu]')
axes.set_xlim(-0.2,0.4)
axes.set_ylim(-0.2,0.4)
axes.grid(b=True,which='major',axis='both')
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
axes.legend(bbox_to_anchor=(1.05,0.8),loc='lower left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.85,hspace=0.30,wspace=0.25)

#fig.tight_layout()

plt.savefig('fig/SST_SSS_bias.png')
plt.show()
