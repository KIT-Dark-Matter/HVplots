import os
import sys
import numpy as np
import pandas as pd
import PhyPraKit as ppk
import cv2
import re
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
import skimage
from scipy.ndimage import gaussian_filter
from PIL import Image, ExifTags
import exifread
import imageio
from datetime import timedelta
import pickle
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import scipy.fft
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from functools import partial
#from minisom import MiniSom
import json
from datetime import datetime
import reliability
sys.path.append('../modules')
import processing
from run_handler import Campaign, Run
from image_handler import Picture
from uncertainties import ufloat
from uncertainties import umath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.pyplot import figure
import os

def gettime(exstr): #converts time to usable units
    examplet=np.fromstring(exstr,dtype=int,sep=':')
    if examplet.size==3:
        return 3600*examplet[0]+60*examplet[1]+examplet[2]
    if examplet.size==4:
        return 3600*examplet[0]+60*examplet[1]+examplet[2]+0.1*examplet[3]

def fixtime(fname,delim='\t";"',skip=2): #gives table for csv file with time in s since start
    data=np.genfromtxt(fname,delimiter=delim,dtype=str,skip_header=skip)[:,0:5]
    offset=gettime(data[0,0][1:11])
    for i in range(data.shape[0]):
        data[i,0]=gettime(data[i,0][1:11])-offset
    return data

def HVplot(fname,maxindex=-1,skiph=2,delim='\t";"',d=ufloat(0.89,0.01)):  #plots voltage and current for csv file
    dat=fixtime(fname,delim=delim,skip=skiph).astype(float)
    if maxindex==-1:
        maxindex=dat[:,2].shape[0]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('Bulk field [kV/cm]', color=color)
    ax1.plot(dat[1:(maxindex-1),0], -dat[1:(maxindex-1),2]/(1000*d.n), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('current[mA]', color=color)
    ax2.plot(dat[1:(maxindex-1),0], -dat[1:(maxindex-1),4], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    print('the maximum voltage reached in this ramp was', np.max(-dat[:,2])/d.n,'V/cm')
    plt.show()

def kaplan(x,b):    #calculates the survival function at voltage x for dataset b
    n=b[0,:].size
    c=np.empty((5,n))  #(max of section, did it die, number alive at start of section, S in area, dS in area)
    for i in range(n):
        c[0:2,i]=b[:,np.argsort(b[0,:])[i]]  
    c[2,0]=n
    for i in range(1,n):
        c[2,i]=n+1-i
    c[3,0]=1
    for i in range(1,n):
        c[3,i]=c[3,i-1]*(c[2,i]-c[1,i-1])/c[2,i]
    c[4,0]=0
    for i in range(1,n):
        h=0
        for j in range(1,i+1):
            h=h+c[1,j-1]/(c[2,j]*(c[2,j]-c[1,j-1]))
        c[4,i]=c[3,i]*np.sqrt(h)
    if x>=0 and x<c[0,0]:
        return np.array([c[3,0],c[4,0]])
    for i in range(n-1):
        if x>=c[0,i] and x<c[0,i+1]:
            return np.array([c[3,i+1],c[4,i+1]])
    return np.array([0,0])
def meier(b,colour='b',name='data'):    #gives a kaplan meier curve with 95% confidence region
    ma=np.max(b[0,:])
    mi=np.min(b[0,:])
    x=np.linspace(0.9*mi,ma,2000)
    f=np.empty((2000,3))
    for i in range(2000):
        f[i,0]=kaplan(x[i],b)[0]
        f[i,1]=f[i,0]+1.96*kaplan(x[i],b)[1]
        if f[i,1] >1 :
            f[i,1]=1
        f[i,2]=f[i,0]-1.96*kaplan(x[i],b)[1]
        if f[i,2] <0 :
            f[i,2]=0 
    plt.plot(x,f[:,0], color=colour, label=name)
    plt.fill_between(x,f[:,1],f[:,2],color=colour,alpha=.1)
    plt.xlabel('Voltage[kV]')
    plt.ylabel('Survival Rate')
    plt.legend()
def find_peak(fname,delim='\t";"',skiph=2,d=ufloat(0.89,0.01)): #gives breakdown voltage for the file
    peaktest=fixtime(fname,delim=delim,skip=skiph).astype(float)  #todo: when no breakdown, give max voltage and 0
    for i in range(np.size(peaktest[:,2])):
        if -peaktest[i,2]+peaktest[i-1,2] <-500:
            return (i,ufloat(-peaktest[i-1,2],1)/d,1)
    return (np.argmax(-peaktest[:,2]),ufloat(np.max(-peaktest[:,2]),1)/d,0)

def survival_data(fname,delim='\t";"'): #gives a breakdown voltage for every ramp in json file
    data=json.load(open(fname))
    p=np.empty((3,data['nr-of-runs']))
    count=0
    for i in data['datafiles']:
        p[0,count]=find_peak(os.path.join(data['hv_path'],i),delim=delim)[1].n/1000
        p[2,count]=find_peak(os.path.join(data['hv_path'],i),delim=delim)[1].s/1000
        p[1,count]=find_peak(os.path.join(data['hv_path'],i),delim=delim)[2]
        count=count+1
    return p

def logrank(a,b):      #plots kaplan-meier curves for both datasets and gives chi^2 and p-value
    n=np.size(a[0,:])
    l=np.size(a[0,:])+np.size(b[0,:])
    c=np.empty((8,l)) #(voltage of breakdown,breakdown in a, breakdown in b,number alive in a, number alive in b)
    count=0
    #print(l)
    for i in range(n):
        c[0,count]=a[0,i]
        c[1,count]=a[1,i]
        c[2,count]=0
        c[7,count]=1
        count=count+1
    for i in range(n):
        c[0,count]=b[0,i]    
        c[1,count]=0
        c[2,count]=b[1,i]
        c[7,count]=0
        count=count+1
    csort=np.empty((8,l))
    for i in range(2*n):
        csort[:,i]=c[:,np.argsort(c)[0,i]]
    c=csort
    c[3,0]=c[4,0]=n
    for i in range(1,l):
        c[3,i]=c[3,i-1]-c[1,i-1]
        c[4,i]=c[4,i-1]-c[2,i-1]
        if c[1,i-1]==c[2,i-1]==0:
            c[3,i]=c[3,i]-c[7,i-1]
            c[4,i]=c[4,i]-1+c[7,i-1]
    for i in range(l):
        c[5,i]=c[3,i]*(c[1,i]+c[2,i])/(c[3,i]+c[4,i])
        c[6,i]=c[4,i]*(c[1,i]+c[2,i])/(c[3,i]+c[4,i])
    print(np.sum(c[5,:]))
    print(np.sum(c[6,:]))
    cs=((np.sum(c[1,:])-np.sum(c[5,:]))**2/np.sum(c[5,:]))+((np.sum(c[2,:])-np.sum(c[6,:]))**2/np.sum(c[6,:]))
    meier(a,'b',name='dataset 1')
    meier(b,'r', name='dataset 2')
    plt.text(4.5,0.2,'X^2='+round(cs,4).astype(str), bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
    plt.text(4.5,0.05,'p='+round(1-scipy.stats.chi2.cdf(cs,1),4).astype(str), bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
    plt.show()
    print(cs)

def breakdown_plot(fname,delim='\t";"'):       #plots the breakdowns for each ramp in the json file +mean
    data=json.load(open(fname))
    mean=np.zeros(50)
    for i in range(data['nr-of-runs']):
        bd=find_peak(os.path.join(data['hv_path'],data['datafiles'][i]),delim=delim,skiph=2)
        dat=fixtime(os.path.join(data['hv_path'],data['datafiles'][i]),delim=delim,skip=2).astype(float)
        mean=mean-dat[bd[0]-40:bd[0]+10,2]/data['nr-of-runs']
        plt.plot(dat[bd[0]-40:bd[0]+10,0]-dat[bd[0],0],-dat[bd[0]-40:bd[0]+10,2],color='Gray')
        xdat=dat[bd[0]-40:bd[0]+10,0]-dat[bd[0],0]
    plt.plot(xdat,mean,color='b', label='mean')
    plt.xlabel('time after breakdown[s]')
    plt.ylabel('Field Strength [V/cm]')
    plt.legend()
    plt.show()

def readhum(fname, skiph=3):
   hum=np.genfromtxt(fname,skip_header=skiph,delimiter='\t',dtype=str)   #reads humidity file and converts time to seconds
   for i in range(hum.shape[0]):  #if you get "can't decode" error, copy and paste into new txt file, idk why this happens
      hum[i,1]=hum[i,1][12:]
      hum[i,1]=gettime(hum[i,1])
   return hum

def VCHplot(HVfile,humfile,delim='\t";"',skiph=2,maxindex=-1): #plots Voltage, current and humidity data
    dat=np.genfromtxt(HVfile,delimiter=delim,dtype=str,skip_header=skiph)[:,0:5]
    offset=gettime(dat[0,0][1:11])
    for i in range(dat.shape[0]):
        dat[i,0]=gettime(dat[i,0][1:11])-offset
    dat=dat.astype(float)
    hdat=readhum(humfile).astype(float)
    maxt=np.max(dat[:,0])+offset
    hmin=0
    hmax=0
    for i in range(hdat[:,0].size):
        if (offset-hdat[i,1])<8:
            hmin=i
            break
    for i in range(hmin,hdat[:,0].size):
      if (maxt-hdat[i,1])<8:
         hmax=i
         break
    if maxindex==-1:
        maxindex=dat[:,2].shape[0]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time[s]')
    ax1.set_ylabel('Voltage[kV]', color=color)
    ax1.plot(dat[:,0], -dat[:,2]/1000, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('current[mA]', color=color)
    ax2.plot(dat[:,0],-dat[:,4], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax3=ax1.twinx()
    color= 'tab:green'
    ax3.set_ylabel('humidity[%]',color=color)
    ax3.plot(hdat[:,1]-offset,hdat[:,3],color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.xlim([np.min(dat[:,0]),np.max(dat[:,0])])
    plt.show()

def FN(file,a=20,b=0,smooth=1):     #gives out FN plot of file, as well as field amplifiction factor
    bdt=find_peak(file,delim='\t";"')[0]
    t=fixtime(file,skip=0).astype(float)[bdt-a:bdt+b,0]
    V=-fixtime(file,skip=0).astype(float)[bdt-a:bdt+b,2]
    I=-fixtime(file,skip=0).astype(float)[bdt-a:bdt+b,4]*0.001
    I=ppk.meanFilter(I,width=smooth)
    plt.plot(1000/V,np.log(np.abs(I/V**2)),'-o')
    plt.xlabel('(1/V) [1/kV]')
    plt.ylabel('ln(I/V^2)')
    plt.show()
    phi=4.5
    def line(x,a,b):
        return a*x+b
    fn=ppk.phyTools.k2Fit(line,1/V,np.log(np.abs(0.001*I/V**2)),sy=np.sqrt((1/(V**2))+(25*10**(-16))/I**2),sx=1/V**2,axis_labels=['1/V [1/V]', 'ln(I/V^2)'],fit_info=True,model_legend='linear fit (y=a*x+b)')
    ares=ufloat(fn[0][0],fn[1][0][1])
    print(-6.831*10**9*phi**(3/2)*ufloat(0.0089,0.0001)/ares)
def conditioning(fname,delim='\t";"',color='b',name='id'):
    if name=='id':
        plotlabel=json.load(open(fname))['id']
    else:
        plotlabel=name
    bd=survival_data(fname,delim=delim)[0,:]
    bderr=survival_data(fname,delim=delim)[2,:]
    n=np.size(bd)
    x=np.linspace(1,n,n)
    plt.errorbar(x,bd,yerr=bderr,fmt='--',color=color, label=plotlabel)
    plt.xticks(ticks=x)
    plt.xlabel('Ramp Nr.')
    plt.ylabel('Breakdown field [kV/cm]')
    plt.ylim(4.5,9)
def prebreakdown_current(fname,delim='\t";"'):       #plots the pre-breakdown for each ramp in the json file +mean
    data=json.load(open(fname))
    mean=np.zeros(39)
    for i in range(data['nr-of-runs']):
        bd=find_peak(data['hv_path']+'\\'+data['datafiles'][i],delim=delim,skiph=2)
        dat=fixtime(data['hv_path']+'\\'+data['datafiles'][i],delim=delim,skip=2).astype(float)
        mean=mean-dat[bd[0]-40:bd[0]-1,4]/data['nr-of-runs']
        plt.plot(dat[bd[0]-40:bd[0]-1,0]-dat[bd[0],0],-dat[bd[0]-40:bd[0]-1,4],color='Gray')
        xdat=dat[bd[0]-40:bd[0]-1,0]-dat[bd[0],0]
    plt.plot(xdat,mean,color='b', label='mean')
    plt.xlabel('Time after breakdown [s]')
    plt.ylabel('Current [mA]')
    plt.legend()
    plt.show()
def breakdown_loc(videofile,calfile,bright=0):
    plt.figure(figsize=(40,24))
    source_file = videofile
    videocapture = cv2.VideoCapture(source_file)
    times = []
    max_brightness = []
    count = 0
    max=0
    while True:
        t_now = count * 40/1000   #40 with 25fps, 10 with 100
        count += 1
        success, img = videocapture.read()
        if success:
            if cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[1]> max:
                max=cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[1]
                brightest_frame=count-1
        if not success:
            print(f'Could not open frame {count}')
            break
        if count == 1300: #13000   
            example_img = img
        times.append(t_now)
        min_v, max_v, _, _ = cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        max_brightness.append(max_v)#np.max(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten()))
        if count > 16000:
            break
    cap = cv2.VideoCapture(source_file) #video_name is the video being called
    cap.set(1,brightest_frame); # Where frame_no is the frame you want
    ret, frame = cap.read()
    fig,axs=plt.subplots(2,2,figsize=(10,6))
    axs[1,0].imshow(frame)
    axs[1,0].set(xlabel='Pixel Nr',ylabel='Pixel Nr')
    axs[0,1].plot(times, max_brightness, marker='o')
    axs[0,1].set(xlabel='Time after video start [s]', ylabel='Max brightness of grayscale')
    bright_image=cv2.VideoCapture(calfile)
    success,bright_img=bright_image.read()  #next time use an image lol
    bright_img=processing.increase_brightness(bright_img,bright)
    axs[0,0].imshow(bright_img)
    axs[0,0].set(xlabel='Pixel Nr',ylabel='Pixel Nr')
    axs[1,1].imshow(bright_img+frame)
    axs[1,1].set(xlabel='Pixel Nr',ylabel='Pixel Nr')
    plt.show()
    plt.imshow(bright_img+frame)
    plt.xlabel('Pixel Nr')
    plt.ylabel('Pixel Nr')
    plt.show()
def IVplot(file,a=15,b=0,d=ufloat(0.89,0.01),ytype='log'):     #gives out FN plot of file, as well as field amplifiction factor
    bdt=find_peak(file,delim='\t";"')[0]
    t=fixtime(file,skip=2).astype(float)[bdt-a:bdt+b,0]
    V=fixtime(file,skip=2).astype(float)[bdt-a:bdt+b,2]
    I=fixtime(file,skip=2).astype(float)[bdt-a:bdt+b,4]
    plt.plot(-V/d.n,np.abs(-I),'--o')
    plt.xlabel('Bulk field [V/cm]')
    plt.ylabel('Current [mA]')
    plt.yscale(ytype)
    plt.show()
def IVsmooth(file,a=15,b=0,d=ufloat(0.89,0.01),ytype='log',smooth=10):     #gives out FN plot of file, as well as field amplifiction factor
    bdt=find_peak(file,delim='\t";"')[0]
    t=fixtime(file,skip=2).astype(float)[bdt-a:bdt+b,0]
    V=fixtime(file,skip=2).astype(float)[bdt-a:bdt+b,2]
    I=fixtime(file,skip=2).astype(float)[bdt-a:bdt+b,4]
    I=ppk.meanFilter(I,width=smooth)
    plt.plot(-V/d.n,np.abs(-I),'-')
    plt.xlabel('Bulk field [V/cm]')
    plt.ylabel('Current [mA]')
    plt.yscale(ytype)
    plt.show()
def meanIV(fname,delim='\t";"'):       #plots the men pre-breakdown current over voltage for each ramp in the json file
    data=json.load(open(fname))
    mean=np.zeros(34)
    meanv=np.zeros(34)
    for i in range(data['nr-of-runs']):
        bd=find_peak(data['hv_path']+'\\'+data['datafiles'][i],delim=delim,skiph=2)
        dat=fixtime(data['hv_path']+'\\'+data['datafiles'][i],delim=delim,skip=2).astype(float)
        mean=mean-dat[bd[0]-40:bd[0]-6,4]/data['nr-of-runs']
        meanv=meanv-dat[bd[0]-40:bd[0]-6,2]/data['nr-of-runs']
        xdat=dat[bd[0]-40:bd[0]-6,4]-dat[bd[0],0]
    plt.plot(meanv,mean,'bo', label='mean')
    plt.xlabel('Voltage [s]')
    plt.ylabel('Current [mA]')
    plt.legend()
    plt.show()
def zoominplot(examplefile,d=ufloat(0.89,0.01)):
    fig, ax = plt.subplots(figsize=(8,6),dpi=80)
    axins = inset_axes(ax,1.6,1.6 , loc='upper left',bbox_to_anchor=(130,420))
    x = fixtime(examplefile)[:,0].astype(float)
    y = -fixtime(examplefile)[:,2].astype(float)/1000
    ax.plot(x, y/d.n)
    ax.plot(np.array([x[find_peak(examplefile)[0]-1]]),np.array([find_peak(examplefile)[1].n])/1000,'o',color='black',label='Breakdown location determined by algorithm')
    ax.set(xlabel='Time since start [s]',ylabel='Bulk field [kV/cm]')
    ax.legend(bbox_to_anchor=(0.66,1.08),fontsize=11)
    axins.plot(x, y/d.n)
    axins.plot(np.array([x[find_peak(examplefile)[0]-1]]),np.array([find_peak(examplefile)[1].n])/1000,'o',color='black')
    axins.set_xlabel('Time [s]')
    axins.set_ylabel('Bulk field [kV]')
    x0=x[find_peak(examplefile)[0]]
    y0=find_peak(examplefile)[1].n
    x1=x0-5
    x2=x0+5
    y1=(y0-1500)/1000
    y2=(y0+200)/1000
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1,y2) # apply the y-limits
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
def zoomincurrentplot(examplefile,d=ufloat(0.89,0.01)):
    fig, ax = plt.subplots()
    axins = inset_axes(ax,1.5,1.5 , loc='upper left',bbox_to_anchor=(170,380))
    x = fixtime(examplefile)[:,0].astype(float)
    y = -fixtime(examplefile)[:,4].astype(float)
    ax.plot(x, y)
    ax.set(xlabel='Time since start [s]',ylabel='Current [mA]')
    axins.plot(x, y)
    axins.set_xlabel('Time [s]')
    axins.set_ylabel('Current [mA]')
    x0=x[find_peak(examplefile)[0]]
    #y0=find_peak(examplefile)[1].n
    x1=x0-5
    x2=x0+5
    y1=np.min(y)-0.001
    y2=np.max(y)+0.001
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1,y2) # apply the y-limits
    plt.xticks(visible=True)
    plt.yticks(visible=True)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
def breakdown_frame(videofile):
    source_file = videofile
    videocapture = cv2.VideoCapture(source_file)
    times = []
    max_brightness = []
    count = 0
    max=0
    while True:
        t_now = count * 40/1000   #40 with 25fps, 10 with 100
        count += 1
        success, img = videocapture.read()
        if success:
            if cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[1]> max:
                max=cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[1]
                brightest_frame=count-1
        if not success:
            print(f'Could not open frame {count}')
            break
        if count == 1300: #13000   
            example_img = img
        times.append(t_now)
        min_v, max_v, _, _ = cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        max_brightness.append(max_v)#np.max(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten()))
        if count > 16000:
            break
    cap = cv2.VideoCapture(source_file) #video_name is the video being called
    cap.set(1,brightest_frame); # Where frame_no is the frame you want
    ret, frame = cap.read()
    return frame