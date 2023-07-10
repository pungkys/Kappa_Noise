#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:55:26 2022

@author: psuroyo
"""
# Calculating kappa-0 from Noise data per station
## IMPORT TRACE PER STATION PER DAY (Noise cutted : FIRST 10 MINUTES A DAY)

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from spectra import fas_psd_smooth
from gradient_to_kappa import gradient, kappa
from stats import bootstrap
from obspy import read, read_inventory
import seaborn as sns
import warnings
warnings.simplefilter("ignore", UserWarning)

# 1. Reading all 10 minutes noise records from 2018 - 2019 
direc='/Users/psuroyo/Downloads/Noise/'
yearl= ['2018','2019']
#yearl=["2018"]
net= 'LV'
sta= 'L001'
upperf, lowerf = 15 , 35
pinv  = "/Users/psuroyo/Documents/Study_DataWork/PNR_TLS_Array/selected_mseed_p/Inventory/"
inv = read_inventory(os.path.join(pinv,'pnr_inventory.xml'),'stationxml')

# 2. Plot amplitude spectra plot vs frequency for each traces 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,sharex=True, sharey=True, figsize=(10,4))
fig2, bx = plt.subplots()


store_hvsr=[]
store_loge=[];store_logn=[];store_logz=[]
grade=[];gradn=[]; gradz=[]
ce=[];cn=[];cz=[]

nfft=1000
ne=nn=nz=0
for year in yearl:
    path = os.path.join(direc, year, net, sta)
    list_ev = [f for f in os.listdir(path) if not f.startswith('.')]
    for ev in list_ev:
        list_rec = [f for f in os.listdir(os.path.join(path,ev)) if not f.startswith('.')]
        if len(list_rec)==3:
            stE, stN = read(os.path.join(path,ev, '*HHE*'), format='mseed'), read(os.path.join(path,ev, '*HHN*'), format='mseed')
            hor = stE+ stN
            ver = read(os.path.join(path,ev, '*HHZ*'), format='mseed')

            for tr in stE:
                tr.remove_response(inv, output='VEL',pre_filt=[0.05,0.1,40,45], 
                                   zero_mean=True, taper=True, taper_fraction=0.05)
                f_e, e_fas, e_psd,e_smooth= fas_psd_smooth(tr, nfft, smoothing='konno-ohmachi',
                                                           smoothing_count=1, smoothing_constant=40)
                coefe,poly1de, xe, ye = gradient (upperf, lowerf, np.log(e_smooth), f_e)
                
                grade.append(coefe[0])
                ce.append(coefe[1])
                store_loge.append(np.log(e_smooth))
                ax1.plot (f_e, np.log(e_smooth),'grey')
                ax1.set_title('E-comp')
                ne = ne+1
    
            for tr in stN:
                tr.remove_response(inv, output='VEL',pre_filt=[0.05,0.1,40,45],
                                   zero_mean=True, taper=True, taper_fraction=0.05)
                f_n, n_fas, n_psd,n_smooth= fas_psd_smooth(tr, nfft, smoothing='konno-ohmachi',
                                                           smoothing_count=1, smoothing_constant=40)
                coefn,poly1dn, xn, yn = gradient (upperf, lowerf, np.log(n_smooth), f_n)
                gradn.append(coefn[0])
                cn.append(coefn[1])
                store_logn.append(np.log(n_smooth))
                ax2.plot (f_n, np.log(n_smooth),'grey')
                ax2.set_title('N-comp')
                nn= nn+1
       

            for tr in ver:
                tr.remove_response(inv, output='VEL',pre_filt=[0.05,0.1,40,45],
                                   zero_mean=True, taper=True, taper_fraction=0.05)
                f_z, z_fas, z_psd,z_smooth= fas_psd_smooth(tr, nfft, smoothing='konno-ohmachi',
                                                           smoothing_count=1, smoothing_constant=40)
                coefz,poly1dz, xz, yz = gradient (upperf, lowerf, np.log(z_smooth), f_z)
                gradz.append(coefz[0])
                cz.append(coefz[1])
                store_logz.append(np.log(z_smooth))
                ax3.plot (f_z, np.log(z_smooth),'grey')
                ax3.set_title('Z-comp')
                nz = nz+1



            hvsr = np.sqrt(pow(e_smooth,2) + pow(n_smooth,2)) / (2 * z_smooth)
            store_hvsr.append(hvsr)
            bx.plot (f_z,hvsr,'grey')
            bx.set_xscale('log')
            bx.set_xlim(0, max(f_z) / 2)

# 3. average of spectra in natural log scale
avg_logampe = np.mean(store_loge, axis=0)
avg_coefe, poly1e, avg_xe, avg_ye = gradient (upperf, lowerf, avg_logampe, f_e)
avg_logampn = np.mean(store_logn, axis=0)
avg_coefn, poly1n, avg_xn, avg_yn = gradient (upperf, lowerf, avg_logampn, f_n)
avg_logampz = np.mean(store_logz, axis=0)
avg_coefz, poly1z, avg_xz, avg_yz = gradient (upperf, lowerf, avg_logampz, f_z)

ax1.plot (f_e, (avg_logampe),'k');ax1.plot( avg_xe, poly1e(avg_xe), '--y')
ax2.plot (f_n, (avg_logampn),'k');ax2.plot( avg_xn, poly1n(avg_xn), '--y')
ax3.plot (f_z, (avg_logampz),'k');ax3.plot( avg_xz, poly1z(avg_xz), '--y')

fig.supylabel("log Amplitude spectra")
fig.supxlabel("Freq (Hz)")
fig.suptitle(net+'.'+sta)
fig.savefig(os.path.join(direc,str(net)+'.'+str(sta)+" log amplitude spectra in f.png"),dpi=300)

hvsr_avg=np.mean(store_hvsr, axis=0)
bx.plot (f_z,hvsr_avg,'k')
fig2.supylabel('H/V')
fig2.supxlabel("Freq (Hz)")
fig2.suptitle(net+'.'+sta)

#4. Calculate statistic of kappa 
ke=[];kn=[];kz=[]
ce_plot=[];cn_plot=[];cz_plot=[]

for i in range (len(grade)):
    ke.append(kappa(grade[i]))
    kn.append(kappa(gradn[i]))
    kz.append(kappa(gradz[i]))
    
    ce_plot.append(ce[i])
    cn_plot.append(cn[i])
    cz_plot.append(cz[i])
    
nbins, low, up = 20, 10, 90
kmean_e, ke_interval, fig = bootstrap(ke, np.mean, nbins, low, up, (net+"."+sta+".E"))
kmean_n, kn_interval, fig = bootstrap(kn, np.mean, nbins, low, up, (net+"."+sta+".N"))
kmean_z, kz_interval, fig = bootstrap(kz, np.mean, nbins, low, up, (net+"."+sta+".Z"))
cmean_e, ce_interval, fig = bootstrap(ce_plot, np.mean, nbins, low, up, (net+"."+sta+".E"))
cmean_n, cn_interval, fig = bootstrap(cn_plot, np.mean, nbins, low, up, (net+"."+sta+".N"))
cmean_z, cz_interval, fig = bootstrap(cz_plot, np.mean, nbins, low, up, (net+"."+sta+".Z"))

store_kappa = [ke, kn, kz]
dk = pd.DataFrame({'k_e': ke, 'k_n': kn, 'k_z':kz})
dk.to_excel(os.path.join(direc, 'store_k0noise3.xlsx'), sheet_name=sta, index=False)
# Plotting for kmean 
ch =['HHE','HHN','HHZ']
c=[cmean_e, cmean_n, cmean_z]
k=[-kmean_e, -kmean_n, -kmean_z]
kstd=[-ke_interval, -kn_interval, -kz_interval]
f=[f_e, f_n, f_z]
x= [xe, xn, xz]
amp=[avg_logampe,avg_logampn, avg_logampz]

ks= -0.018
ksc= -0.019
kc= -0.020
fig,ax = plt.subplots(1, 3, figsize=(15,6), sharey=True)
fig.suptitle(net+'.'+sta,fontsize=18)

for i in range (len(ax)):
        pols=np.poly1d([ks *(2*np.pi),c[i]])
        polsc=np.poly1d([ksc*(2*np.pi),c[i]])
        polc=np.poly1d([kc*(2*np.pi),c[i]])
        ax[i].set_title(str(ch[i]), fontsize = 16)
        ax[i].tick_params(labelsize=13)
        ploy1= np.poly1d([k[i]*(2*np.pi),c[i]])
        sns.kdeplot(f[i], (amp[i]), cmap="Blues", shade=True, thresh=0,ax=ax[i])
        ax[i].plot(x[i], (ploy1(x[i])),'--k', label='$k_{noise}$ '+"{:.3f}".format(abs(k[i])))
        ax[i].plot(x[i], (pols(x[i])),'--r',label='$k_{s}$= 0.018')
        ax[i].plot(x[i], (polsc(x[i])),'--b',label='$k_{sc}$= 0.019')
        ax[i].plot(x[i], (polc(x[i])),'--g',label='$k_{c}$= 0.020')
        
        pol_stdk=np.poly1d([kstd[i][0]*(2*np.pi),c[i]])
        pol_stdkm=np.poly1d([kstd[i][1]*(2*np.pi),c[i]])
        #ax[i].plot(x[i], (pol_stdk(x[i])),'-.k',alpha=0.3,label=' convidence interval $k_{noise}$ ')
        #ax[i].plot(x[i], (pol_stdkm(x[i])),'-.k',alpha=0.3)
        ax[i].legend(loc='upper right')
        
fig.supylabel(" log Amplitude spectra",fontsize=18)
fig.supxlabel("Freq (Hz)",fontsize=18)
fig.savefig(os.path.join(direc,str(net)+'.'+str(sta)+"A.png"),dpi=300)
plt.show()


c= [avg_coefe, avg_coefn, avg_coefz]
p= [poly1e, poly1n, poly1z]

fig,ax = plt.subplots(1, 3, figsize=(15,6), sharey=True)
fig.suptitle(net+'.'+sta,fontsize=18)
# plt.ylim(-6.5,-3.5)
for i in range (len(ax)):
        pols=np.poly1d([-0.018 *(2*np.pi),c[i][1]])
        polsc=np.poly1d([-0.019*(2*np.pi),c[i][1]])
        polc=np.poly1d([-0.020*(2*np.pi),c[i][1]])
        ax[i].set_title(str(ch[i]), fontsize = 16)
        ax[i].tick_params(labelsize=13)
        
        sns.kdeplot(f[i], (amp[i]), cmap="Reds", shade=True, thresh=0,ax=ax[i])
        ax[i].plot(x[i], (p[i](x[i])),'--k', label='$k_{noise}$ '+"{:.3f}".format(abs(c[i][0])/(2*np.pi)))
        ax[i].plot(x[i], (pols(x[i])),'--r',label='$k_{s}$= 0.018')
        ax[i].plot(x[i], (polsc(x[i])),'--b',label='$k_{sc}$= 0.019')
        ax[i].plot(x[i], (polc(x[i])),'--g',label='$k_{c}$= 0.020')
#         ax[i].plot(x[i], (ploy1m(x[i])),'-.k',alpha=0.3,label='k= '+"{:.3f}".format(k[0])+"+/-"+"{:.4f}".format(k[1]))
        ax[i].legend(loc='upper right')
        
fig.supylabel(" log Amplitude spectra",fontsize=18)
fig.supxlabel("Freq (Hz)",fontsize=18)
fig.savefig(os.path.join(direc,str(net)+'.'+str(sta)+"k_meanspectra.png"),dpi=300)
plt.show()

import pandas as pd 
data_file= {'hvsr_avg':hvsr_avg,'f': f_z}
datafile= pd.DataFrame(data_file,columns=['hvsr_avg','f'])


datafile.to_csv(os.path.join(direc, str(sta)+ ".csv" ), sep='\t')
