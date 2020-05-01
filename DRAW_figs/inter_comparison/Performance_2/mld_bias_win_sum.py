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
titles = {"interannual":"interannual variability", "monclim":"monthly climatology"}

markinfo = json.load(open('../json/inst_color_style.json'))

### Winter

infl1 = "./csv_clim/MLD_winter_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/MLD_winter_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
dfm=df.drop('MMM',axis=0)
print(dfm)

fig = plt.figure(figsize=(11,8))
fig.suptitle("MLD bias", size='x-large')

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

r = dfm.loc[:,['OMIP1_rmse','OMIP2_rmse']].corr()
print(r)
x_np = dfm[['OMIP1_rmse']].values
y_np = dfm['OMIP2_rmse'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(20,100,10)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(30,80,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('Winter MLD bias rmse')
axes.set_xlabel('OMIP1_rmse [m]')
axes.set_ylabel('OMIP2_rmse [m]')
axes.set_xlim(20,90)
axes.set_ylim(20,90)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,2)

for index, row in df.iterrows():
    print(row['OMIP1_mean'],row['OMIP2_mean'])
    btm=row['OMIP1_mean']
    side=row['OMIP2_mean']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='blue'
        edgecol='blue'
    else:
        markcol='red'
        edgecol='orange'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = dfm.loc[:,['OMIP1_mean','OMIP2_mean']].corr()
print(r)
x_np = dfm[['OMIP1_mean']].values
y_np = dfm['OMIP2_mean'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-30,40,10)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-25,20,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('Winter MLD bias mean')
axes.set_xlabel('OMIP1_mean [m]')
axes.set_ylabel('OMIP2_mean [m]')
axes.set_xlim(-30,30)
axes.set_ylim(-30,30)
axes.grid(b=True,which='major',axis='both')
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
#axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

### Summer

infl1 = "./csv_clim/MLD_summer_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/MLD_summer_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
dfm=df.drop('MMM',axis=0)
print(dfm)

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

r = dfm.loc[:,['OMIP1_rmse','OMIP2_rmse']].corr()
print(r)
x_np = dfm[['OMIP1_rmse']].values
y_np = dfm['OMIP2_rmse'].values
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
axes.text(6,18,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('Summer MLD bias rmse')
axes.set_xlabel('OMIP1_rmse [m]')
axes.set_ylabel('OMIP2_rmse [m]')
axes.set_xlim(5,20)
axes.set_ylim(5,20)
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
        markcol='blue'
        edgecol='blue'
    else:
        markcol='red'
        edgecol='orange'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = dfm.loc[:,['OMIP1_mean','OMIP2_mean']].corr()
print(r)
x_np = dfm[['OMIP1_mean']].values
y_np = dfm['OMIP2_mean'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-15,16,1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-13,12,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('Summer MLD bias mean')
axes.set_xlabel('OMIP1_mean [m]')
axes.set_ylabel('OMIP2_mean [m]')
axes.set_xlim(-15,15)
axes.set_ylim(-15,15)
axes.grid(b=True,which='major',axis='both')
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
axes.legend(bbox_to_anchor=(1.01,0.8),loc='lower left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.85,hspace=0.30,wspace=0.25)

#fig.tight_layout()

plt.savefig('fig/mld_bias_win_sum.png')
plt.show()
