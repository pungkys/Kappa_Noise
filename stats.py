#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:45:19 2022

@author: psuroyo
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt

def draw_bs_replicates(data,func,size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data,size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample) 
    return bs_replicates

# Create a function to get x, y for of ecdf
def get_ecdf(data):
    
    # Get lenght of the data into n
    n = len(data)
    
    # We need to sort the data
    x = np.sort(data)
    
    # the function will show us cumulative percentages of corresponding data points
    y = np.arange(1,n+1)/n
    
    return x,y

def resampling(data,size,repetition=0.2):
    import random
    # Create an empty array to store replicates
    bs_sample = np.empty(size)
    k_repetition = int(round(repetition* len(data)))
    k_retention = int(round((1-repetition)* len(data)))
    #Generate n samples from a sequence with the possibility of repetition. 
    bs_repetition = random.choices(data,k=k_repetition) 
    #Generate n unique samples (multiple items) from a sequence without repetition a.k.a retention
    bs_retention = random.sample(list(data),k=k_retention)
    
    bs_sample = np.concatenate((bs_repetition, bs_retention), axis=None)
    # check length output
    if len(bs_sample)!= size:
        print('check size')
    else:
        return bs_sample
    
def bootstrap (grad, func, nbins, low, up, suptitle = None):
    if suptitle is not None:
        fig, ax = plt.subplots(1, 2, figsize=(14,12))
        for i in range(1000):
    
            # Generate a bootstrap sample
            bs_sample_ = np.random.choice(np.asarray(grad),size=len(np.asarray(grad)))
    
            # Plot ecdf for bootstrap sample
            x, y = get_ecdf(bs_sample_)
            ax[0].scatter(x, y, s=1, c='b', alpha=0.3)
            ax[0].set_title("Empirical Cumulative Distribution Function", fontsize=18)
            ax[0].set_xlabel(r"$ \kappa$")
            ax[0].set_ylabel("ECDF")
        
        # Draw 10000 bootstrap replicates
        bs_replicates_ = draw_bs_replicates(np.asarray(grad), func, 1000)
        conf_interval = np.percentile(bs_replicates_,[low, up])
        emp = func (np.asarray(grad))
        # Plot probability density function
        ax[1].hist(bs_replicates_, bins= nbins, density=True)
        ax[1].axvline(x=np.percentile(bs_replicates_,[low]), ymin=0, ymax=1,label=str(low)+' th percentile',c='y')
        ax[1].axvline(x=np.percentile(bs_replicates_,[up]), ymin=0, ymax=1,label=str(up)+ ' th percentile',c='r')
        plt.figtext(0.93, 0.5, "Empirical mean: " + str(round(emp,3))
                                +'\n'+"Bootstrap replicates mean: " + str(round(func(bs_replicates_),3))
                                +'\n'+"The confidence interval: "+ str(np.around(conf_interval,4)),
                                bbox=dict(facecolor='white'))
    
        ax[1].set_xlabel(r"$ \kappa$",fontsize=14)
        ax[1].set_ylabel("PDF",fontsize=14)
        ax[1].set_title("Probability Density Function", fontsize=18)
        ax[1].legend()
        
        con_interval = np.around(conf_interval,4)
        boostrap_value = round(func(bs_replicates_),3)
    
        plt.suptitle(suptitle,fontsize=20)
        plt.show()
        
        return boostrap_value, con_interval, fig
    
    else:
        # Draw 10000 bootstrap replicates
        bs_replicates_ = draw_bs_replicates(np.asarray(grad), func, 1000)
        conf_interval = np.percentile(bs_replicates_,[low, up])
        emp = func (np.asarray(grad))
        
        con_interval = np.around(conf_interval,4)
        boostrap_value = round(func(bs_replicates_),3)
        
        return boostrap_value, con_interval, None

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def meanstd_k (kappa, c, percentage): 
    k, kstd   = mean_confidence_interval((np.asarray(kappa)), confidence= percentage)
    coe, cstd = mean_confidence_interval((np.asarray(c)), confidence= percentage)
    ploy1 = np.poly1d([kappa,coe])
    ploy1p= np.poly1d([kappa+kstd,coe+cstd])
    ploy1m= np.poly1d([kappa-kstd,coe-cstd])
    
    return ploy1, ploy1p, ploy1m