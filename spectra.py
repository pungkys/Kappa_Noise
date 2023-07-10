#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:33:43 2022

@author: psuroyo
"""

import numpy as np
from konno_ohmachi_smoothing import calculate_smoothing_matrix


def nextpow2(n): 
    count = 0; 
#if n is already a power of 2, return n immediately
    if (n and not(n & (n - 1))): 
        return n 
# this logic below only works if n is not already a power of 2   
    while( n != 0): 
        n >>= 1
        count += 1     
    return 1 << count


def psd_to_amp(sig,_sigpsd,fsig):
    sig_amp = np.sqrt((_sigpsd*len(fsig))/sig.stats['sampling_rate'])
    return sig_amp

def fas_psd_smooth (tr,nfft, smoothing=None,
                  smoothing_count=1, smoothing_constant=40):
    
#     nfft  = nextpow2(len(tr.data))
    norm = len(tr.data)
    nfft = nextpow2(nfft)
    npositive = nfft//2
    pslice = slice(1, npositive)
#     padded_data = np.pad(tr.data, (0, nfft - len(tr.data)), 'constant')
    freq_sig_ = np.fft.fftfreq(nfft, d=1/tr.stats['sampling_rate'])[pslice] 
    sig_psd_ = (np.abs((np.fft.fft(tr.data,n =nfft)[pslice]))**2) 
    sig_amp_= psd_to_amp(tr,sig_psd_,freq_sig_) 

     # Apply smoothing.
    if smoothing:
        if 'konno-ohmachi' in smoothing.lower():
            
            sm_matrix = calculate_smoothing_matrix(freq_sig_,smoothing_constant)
                
            for _j in range(smoothing_count):
                smooth_fas = np.dot(sig_amp_, sm_matrix)
            return freq_sig_, sig_amp_, sig_psd_,smooth_fas
    else:
        return freq_sig_, sig_amp_, sig_psd_