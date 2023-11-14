# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:27:39 2023

@author: f.divruno
Modified : S. S. Varghese
"""

import numpy as np
import matplotlib.pyplot as plt
    
def plot_all_waterfall(data, t, freq, logged, title='', savefolder='' ,savename='', savefig=False):
    '''
    function to plot a waterfall plot    

    Parameters
    ----------
    data : array
        waterfall data
    time : array
        time array
    freq : array
        frequency array
    title : str, optional
        Title of the plot
    savefolder : str
        folder to save the plot
    savename : str
        plot name
    savefig : boolean
        flag to save the figure

    Returns
    -------
    None.

    '''       
    if not logged: #convert to mW
        data = 10.0**(data/10.0)
        lab = "Power [mW]"
    else:
        lab =  "Power [dB]"
             
    vmax = np.nanpercentile(data,98) # setting min and max percentile of the data
    vmin = np.nanpercentile(data,2)
    
    fig, ax = plt.subplots(2,2,figsize=(12, 10),
                            gridspec_kw={'height_ratios': [1, 0.25], 'width_ratios':[1,0.25]}, constrained_layout=True)
    
    ax[1,0].sharex(ax[0,0])
    ax[0,1].sharey(ax[0,0])
    
    img = ax[0,0].imshow(data, aspect='auto', vmin=vmin, vmax=vmax, 
                          origin='lower',extent=[freq[0], freq[-1], t[0], t[-1]])
      
    ax[0,0].set_ylabel("Time [Hours]")
                      
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(lab, rotation=270)
    
    if not logged:
    
        Pmax_f = np.nanmax(data,axis=0)  # max value
        #Pmean_f = np.nanpercentile(data,50,axis=0)
        Pmean_f = np.nanmean(data,axis=0) # mean value
        P99_f = np.nanpercentile(data,99.9,axis=0) # 99 percentile

        Pm_t = np.nanmean(data,axis=1) # mean
        Pmax_t = np.nanmax(data,axis=1) # max 
 
    else:
        Pmax_f = 10*np.log10(np.nanmax(10**(data/10),axis=0))
        #Pmean_f = 10*np.log10(np.nanpercentile(10**(data/10),50,axis=0))
        Pmean_f = 10*np.log10(np.nanmean(10**(data/10),axis=0))
        P99_f = 10*np.log10(np.nanpercentile(10**(data/10),99.9,axis=0))

        Pm_t = 10*np.log10(np.nanmean(10**(data/10),axis=1))
        Pmax_t = 10*np.log10(np.nanmax(10**(data/10),axis=1))

    ax[1,0].plot(freq,Pmax_f,'-', color =  'b')
    ax[1,0].plot(freq,Pmean_f,'-', color = 'r')
    ax[1,0].plot(freq,P99_f,'-', color = 'c')
    ax[1,0].set_ylabel(lab)
    ax[1,0].set_xlabel("Frequency [GHz]")
    ax[1,0].grid('both',alpha=0.5)
    ax[1,0].legend(['Max','Mean','99.9 percentile'],fontsize=10)
    
    
    ax[0,1].plot(Pm_t,t,'-', label = 'Mean') #np.arange(0,data.shape[0]),'-')
    #ax[0,1].plot(Pmax_t,t,'-', label = 'Max') #np.arange(0,data.shape[0]),'-')
    ax[0,1].set_xlabel(lab)
    
    ax[0,1].grid('both',alpha=0.5)
    ax[0,1].legend()

    ax[0,0].set_title(title)
    
    ax[1,1].axis('off')
    
    if savefig: #if the name is provided, save the figure
        plt.savefig(savefolder+'/'+savename, dpi=200)    
        # plt.close()
    else:
        plt.show()
    

