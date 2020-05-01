# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D","x"]
mipcols = ["darkred", "darkblue"]
miparcs = ["orange", "lightblue"]

markinfo = json.load(open('../json/inst_color_style.json'))

#---------------------------

fig = plt.figure(figsize=(11,8))
fig.suptitle("sea ice extent", size='x-large')

axes=fig.add_subplot(2,2,1)

infl1 = "./csv_var/siextent_omip1-NH-MAR-mean.csv"
df1=pd.read_csv(infl1,index_col=0)
df1.rename(index=lambda x: x[:-6],inplace=True)
print(df1)
infl2 = "./csv_var/siextent_omip2-NH-MAR-mean.csv"
df2=pd.read_csv(infl2,index_col=0)
df2.rename(index=lambda x: x[:-6],inplace=True)
print(df2)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
#dfm=df.drop('MMM',axis=0)
#print(dfm)

for index, row in df.iterrows():
    print(row['OMIP1-NH-MAR'],row['OMIP2-NH-MAR'])
    btm=row['OMIP1-NH-MAR']
    side=row['OMIP2-NH-MAR']
    if ( index == 'OBS'):
        markcol = 'red'
        edgecol = 'red'
    else:
        markcol = 'blue'
        edgecol = 'lightblue'

    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df.loc[:,['OMIP1-NH-MAR','OMIP2-NH-MAR']].corr()
print(r)
x_np = df[['OMIP1-NH-MAR']].values
y_np = df['OMIP2-NH-MAR'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(10.0,21.0)
y=x.copy()
axes.plot(x,y,color="grey",linestyle="dotted",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(11,19,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('sea ice extent NH March')
axes.set_xlabel('OMIP1')
axes.set_ylabel('OMIP2')
axes.set_xlim(10.0,20.0)
axes.set_ylim(10.0,20.0)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,2)

infl1 = "./csv_var/siextent_omip1-NH-SEP-mean.csv"
df1=pd.read_csv(infl1,index_col=0)
df1.rename(index=lambda x: x[:-6],inplace=True)
infl2 = "./csv_var/siextent_omip2-NH-SEP-mean.csv"
df2=pd.read_csv(infl2,index_col=0)
df2.rename(index=lambda x: x[:-6],inplace=True)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
#dfm=df.drop('MMM',axis=0)
#print(dfm)

for index, row in df.iterrows():
    print(row['OMIP1-NH-SEP'],row['OMIP2-NH-SEP'])
    btm=row['OMIP1-NH-SEP']
    side=row['OMIP2-NH-SEP']
    if ( index == 'OBS'):
        markcol = 'red'
        edgecol = 'red'
    else:
        markcol = 'blue'
        edgecol = 'lightblue'

    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df.loc[:,['OMIP1-NH-SEP','OMIP2-NH-SEP']].corr()
print(r)
x_np = df[['OMIP1-NH-SEP']].values
y_np = df['OMIP2-NH-SEP'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(4.0,11.0)
y=x.copy()
axes.plot(x,y,color="grey",linestyle="dotted",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(4.5,9.5,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('sea ice extent NH September')
axes.set_xlabel('OMIP1')
axes.set_ylabel('OMIP2')
axes.set_xlim(4.0,10.0)
axes.set_ylim(4.0,10.0)
axes.grid(b=True,which='major',axis='both')

axes=fig.add_subplot(2,2,3)

infl1 = "./csv_var/siextent_omip1-SH-SEP-mean.csv"
df1=pd.read_csv(infl1,index_col=0)
df1.rename(index=lambda x: x[:-6],inplace=True)
infl2 = "./csv_var/siextent_omip2-SH-SEP-mean.csv"
df2=pd.read_csv(infl2,index_col=0)
df2.rename(index=lambda x: x[:-6],inplace=True)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
#dfm=df.drop('MMM',axis=0)
#print(dfm)

for index, row in df.iterrows():
    print(row['OMIP1-SH-SEP'],row['OMIP2-SH-SEP'])
    btm=row['OMIP1-SH-SEP']
    side=row['OMIP2-SH-SEP']
    if ( index == 'OBS'):
        markcol = 'red'
        edgecol = 'red'
    else:
        markcol = 'blue'
        edgecol = 'lightblue'
        
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df.loc[:,['OMIP1-SH-SEP','OMIP2-SH-SEP']].corr()
print(r)
x_np = df[['OMIP1-SH-SEP']].values
y_np = df['OMIP2-SH-SEP'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(13.0,27.0)
y=x.copy()
axes.plot(x,y,color="grey",linestyle="dotted",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(14,24,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('sea ice extent SH September')
axes.set_xlabel('OMIP1')
axes.set_ylabel('OMIP2')
axes.set_xlim(13.0,26.0)
axes.set_ylim(13.0,26.0)
axes.set_xticks(np.arange(14,26,2))
axes.grid(b=True,which='major',axis='both')

axes=fig.add_subplot(2,2,4)

infl1 = "./csv_var/siextent_omip1-SH-MAR-mean.csv"
df1=pd.read_csv(infl1,index_col=0)
df1.rename(index=lambda x: x[:-6],inplace=True)
infl2 = "./csv_var/siextent_omip2-SH-MAR-mean.csv"
df2=pd.read_csv(infl2,index_col=0)
df2.rename(index=lambda x: x[:-6],inplace=True)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
#dfm=df.drop('MMM',axis=0)
#print(dfm)

for index, row in df.iterrows():
    print(row['OMIP1-SH-MAR'],row['OMIP2-SH-MAR'])
    btm=row['OMIP1-SH-MAR']
    side=row['OMIP2-SH-MAR']
    mi = int(markinfo[index]["marker"])
    name=index
    if ( index == 'OBS'):
        markcol = 'red'
        edgecol = 'red'
    else:
        markcol = 'blue'
        edgecol = 'lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df.loc[:,['OMIP1-SH-MAR','OMIP2-SH-MAR']].corr()
print(r)
x_np = df[['OMIP1-SH-MAR']].values
y_np = df['OMIP2-SH-MAR'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0.0,8.0)
y=x.copy()
axes.plot(x,y,color="grey",linestyle="dotted",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(1,6,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('sea ice extent SH March')
axes.set_xlabel('OMIP1')
axes.set_ylabel('OMIP2')
axes.set_xlim(0.0,7.0)
axes.set_ylim(0.0,7.0)
axes.grid(b=True,which='major',axis='both')
axes.legend(bbox_to_anchor=(1.01,0.8),loc='lower left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.85,hspace=0.30,wspace=0.25)

#fig.tight_layout()

plt.savefig('fig/si_extent.png')
plt.show()
