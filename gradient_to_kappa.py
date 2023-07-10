#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:42:52 2022

@author: psuroyo
"""
import numpy as np

def gradient (low, up, spec, freq):
    # find where low freq, first index
    i_low = np.where(np.round(freq)== low)[0][0] 
    # find where upper freq, last index
    i_up = np.where(np.round(freq)== up)[0][-1]
    # cut the data from i_low to i_up
    x = freq [i_low: i_up ]
    y = spec [i_low: i_up ]
    coef = np.polyfit(x, y, 1)
    poly1d = np.poly1d(coef)
    return coef,poly1d, x, y 

def kappa(gradient_coef):
    m = gradient_coef
    return -m/(np.pi)
    