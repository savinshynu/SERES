"""
Created on Fri Nov 10 

@author:  S. S. Varghese

A module to process the observations of satellites and 
conduct a bunch of diagnostics

Input:
telescope data in .npz format
satellite coordinates in .npz format

"""
import sys
import argparse
import numpy as np
from scipy.stats import median_abs_deviation as mad
from matplotlib import pyplot as plt
from RSS_assignment import plot_all_waterfall

plt.rcParams.update({'font.size': 14})


fov_rad = 0.0782 # Primary beam radius, FWHM of Onsala 13.2 m telescope at 10 GHz = 0.15634 deg 

def ang_dist(x1,y1,x2,y2):
    """
    Calcualt the angular distance between an array of (azimuth, elevation) points on a sphere 
    wrt to a (ref azimuth and ref elevation) and returns coordinates within a radius of the 
    reference point
    
    Parameters:
    ----------
    x1 : numpy float array
         azimuth
    y1 : numpy float array
         elevation
    x2 : float
         reference azimuth
    y2 : float
         reference elevation
    Returns:
    -------
    Index array 
    angular distance in degrees
    """
    num = len(x1) # Number of the coordinates 
    x2 = np.ones(num)*x2 # Duplicating ref point into the first array shape
    y2 = np.ones(num)*y2

    # Using the great circle distance equation
    a = np.sin(y1*(np.pi/180.0))*np.sin(y2*(np.pi/180.0))
    b = np.cos(y1*(np.pi/180.0))*np.cos(y2*(np.pi/180.0))*np.cos((x1-x2)*(np.pi/180.0))
    z=a+b
    ang_dist = np.arccos(z)
    ang_dist *= (180.0/np.pi) # Converting back to degrees
    good_ind = np.where((np.abs(ang_dist) < fov_rad))[0] # Indices with distance less the fov_rad
    # returning indices and angular distances
    return good_ind, ang_dist[good_ind]



def plot_coord_polar(tel_az, tel_el, sat_az, sat_el, sat_ind):

    """
    Function to plot the telescope pointing and three random
    satellite coordinates

    Parameters:
    -----------
    tel_az : float array
             telescope azimuth
    tel_el : float array
             telescop elevation
    sat_az : float array
             azimuth of all satellites
    sat_el : float array
             elevation of all satellites
    sat_ind : list
              Indices of 3 satellites
    Returns:
    -------
    A plot 
    """
    fig, ax = plt.subplots(figsize=(8, 8),subplot_kw={'projection': 'polar'})
    ax.scatter(np.radians(tel_az), tel_el, s = 10, marker = 'o', facecolors = 'none', edgecolors='b', label = 'tel') # Plotting telescope pointings
    clr = ['red', 'black', 'orange']
    for i, ind in enumerate(sat_ind): # Plotting satellite coordinates
        ax.scatter(np.radians(sat_az[:, ind]), sat_el[:, ind], s = 2, marker = 'o', facecolors = 'none', edgecolors=clr[i], label = f'sat-{ind}' )
    ax.set_rmax(0) # Setting the max and min radius value
    ax.set_rmin(90)
    ax.set_rticks([80, 60, 40, 20])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.legend(loc = 'upper right')
    ax.grid(True, alpha=0.5)
    ax.set_title("Telescope Pointings and Satellite transits ", va='bottom')
    plt.show()

class search_emission:
    """
    A class to process the radio observation of satellites
    """
    
    def __init__(self, radio_data, sat_coords):
        """
        Intializing class variables
        """
        self.radio_wf = radio_data['data'] # Radio waterfall data
        self.radio_tel_az = radio_data['telescope_az'] # Telescope azimuth pointings
        self.radio_tel_el = radio_data['telescope_el'] # Telescope elevation pointings
        self.radio_freqs = radio_data['f_GHz'] # Frequency in GHz
        self.radio_mjd = radio_data['mjd'] # Time in MJD
        self.radio_times = (self.radio_mjd - self.radio_mjd[0])*24 # Time in hours
        self.sat_az = sat_coords['sat_az'] # Satellite azimuth
        self.sat_el = sat_coords['sat_el'] # Satellite elevation
        self.sat_range = sat_coords['sat_range'] # Satellite range
        self.nsat = self.sat_az.shape[1] # No. of satellites
        self.ntobs = len(radio_data['telescope_az']) # No. of time integrations

    def print_metadata(self):
        """
        print basic metadata of the 
        radio data and satellite data
        """
        print(f" Radio data: \n\
                ********** \n\
                Observations started on {self.radio_mjd[0]} \n\
                Number of time integrations: {self.ntobs} \n\
                Duration of observations : {self.radio_times[-1]} hours \n\
                Time integration : {(self.radio_times[1] - self.radio_times[0])*3600.0} s, {self.radio_times[-1]*3600.0/self.ntobs} s \n\
                Number of frequencies: {len(self.radio_freqs)} \n\
                Frequency channel width : {self.radio_freqs[1] - self.radio_freqs[0]} GHz \n\
                Satellite Data: \n\
                *************** \n\
                Number of Satellites : {self.nsat} \n")

    def plot_all(self, data, times, freqs, save_fold, save_name, logged = True, savefig = False):
        """
        Plot waterfall, time series and the spectra

        Parameters:
        -----------
        data: array
              waterfall data
        times: array
               time data
        freqs : array
                frequency data
        save_fold : str
                    folder to save the plot
        save_name : str
                    plot name to save
        logged : bool
                 True: plot everything in decibels
                 False: plot in mW
        savefig: bool
                 flag to save image

        Returns:
        -------
        plot the figure or save the figure
        """
        print("Plotting the waterfall, time series and spectra now:")
        plot_all_waterfall(data, times, freqs, logged,'', save_fold, save_name, savefig)
           

    def plot_waterfall(self, data, times, freqs, outdir, outname, logged = True, savefig = False ):
        """
        Plot waterfall
        Parameters:
        ----------
        data: array
              waterfall data
        times: array
               time data
        freqs : array
                frequency data
        outdir : str
                    folder to save the plot
        outname : str
                    plot name to save
        logged : bool
                 True: plot everything in decibels
                 False: plot in mW
        savefig: bool
                 flag to save image

        Returns:
        --------
        plot the figure or save the figure
        """
        print("Plotting the waterfall now:")
        fig = plt.figure(figsize = [12,8])
        if logged: #plot in dB
            img = plt.imshow(data.T, aspect='auto', origin='lower',extent=[times[0], times[-1], freqs[0], freqs[-1]])
            cbar = fig.colorbar(img)
            cbar.set_label("Power [dB]", rotation = 90)
        else: #plot in mW
            img = plt.imshow(10.0**(data/10.0).T, aspect='auto', origin='lower',extent=[times[0], times[-1], freqs[0], freqs[-1]])
            cbar = fig.colorbar(img)
            cbar.set_label("Power [mW]", rotation = 90)
        
        plt.xlabel("Time [Hours]") #plot axis labels
        plt.ylabel("Frequency [MHz]")
        
        if savefig: #if the name is provided, save the figure
            plt.savefig(outdir+'/'+outname, dpi=200)    
            plt.close()
        else:
            plt.show()
        

    def get_waterfall_statistics(self, data, times, freqs):
        """
        Print basic statistics of a waterfall file
        Parameters:
        ----------
        data: array
              waterfall data
        times: array
               time data
        freqs : array
                frequency data
        
        Returns:
        --------
        None
        """

        data = 10.0**(data/10.0) # converting to mW
        std = np.std(data)  # standard deviation
        tseries = np.nanmean(data, axis = 1) # time series
        fspec = np.nanmean(data, axis = 0)   # mean spectra
        maxt = np.argmax(tseries)    # max arg of time series
        maxf = np.argmax(fspec)      # mas arg of spectra
        print(f"Standard deviation of the data: {std} \n\
                Peak of the time series: {times[maxt]} s \n\
                Peak of the spectra : {freqs[maxf]} GHz  \n ")

    def plot_timeseries(self, data, times, freqs, outdir, outname, logged = True, savefig = False):
        """
        Plot Time series
        Parameters:
        ----------
        data: array
              waterfall data
        times: array
               time data
        freqs : array
                frequency data
        outdir : str
                    folder to save the plot
        outname : str
                    plot name to save
        logged : bool
                 True: plot everything in decibels
                 False: plot in mW
        savefig: bool
                 flag to save image

        Returns:
        -------
        plot the figure or save the figure

        """
        print("Plotting the time series now:")
        series = 10.0*np.log10(np.nanmean(10**(data/10.0), axis = 1)) # Averaging along frequency axis

        fig = plt.figure(figsize = [12,8])
        if logged:
             plt.plot(times, series)
             plt.ylabel("Power [dB]")
        else:    
            plt.plot(times, 10.0**(series/10.0))
            #plt.plot(10.0**(series/10.0))
            plt.ylabel("Power [mW]")
            plt.xlabel("Time [Hours]")
            #plt.xlabel("TIme integrations")

        if savefig: #if the name is provided, save the figure
            plt.savefig(outdir+'/'+outname, dpi=200)    
            plt.close()
        else:
            plt.show()

    def plot_spectra(self, data, times, freqs, outdir, outname, logged = True, savefig = False):   
        """
        Plot spectra
        Parameters:
        -----------
        data: array
              waterfall data
        times: array
               time data
        freqs : array
                frequency data
        outdir : str
                    folder to save the plot
        outname : str
                    plot name to save
        logged : bool
                 True: plot everything in decibels
                 False: plot in mW
        savefig: bool
                 flag to save image

        Returns:
        --------
        plot the figure or save the figure

        """
        print("Plotting the spectra now:")
        spectra = 10.0*np.log10(np.nanmean(10.0**(data/10.0), axis = 0)) # averaging along time axis to get the spectra
        
        fig = plt.figure(figsize = [12,8])
        
        if logged:
             plt.plot(freqs, spectra)
             plt.ylabel("Power [dB]")
        else:    
            plt.plot(freqs, 10.0**(spectra/10.0))
            plt.ylabel("Power [mW]")
        
        plt.xlabel("Frequency [GHz]")
        
        if savefig: #if the name is provided, save the figure
            plt.savefig(outdir+'/'+outname, dpi=200)    
            plt.close()
        else:
            plt.show()
    
    def calc_sat_within_beam(self, outdir):
        """
        Calculate the number of satellites within the telescope beam radius at each time
        and if found one,  write out the results as a text file in the form of
        time integration, satellite number, angular distance of satellite from telescope beam 
        and range of satellite during transit
        Parameters:
        ourdir : str
                 Output directory
        Returns:
            A text file with results
        """ 
        transit = 0  #Initialize the number of satellite transits
        with open(f"{outdir}/transit.txt", "w") as fh:
            fh.write("Time_integration Time_hrs Satellite_index angular_distance  Satellite_range\n")
            for i in range(self.ntobs):
                az_ref = self.radio_tel_az[i]
                el_ref = self.radio_tel_el[i]
                sat_az = self.sat_az[i,:]
                sat_el = self.sat_el[i,:]
                index, ang_sep = ang_dist(sat_az, sat_el, az_ref, el_ref)
                if len(index) > 0:
                    fh.write(f"{i} {self.radio_times[i]} {index} {ang_sep} {self.sat_range[i, index]} \n")
                    transit += len(index)
                    print(f"{len(index)} satellites transit at integration {i}")
            fh.close()
        print(f"{transit} satellites transited during the whole obseravation")

    def grab_wf(self, tbeg = None, tend = None, fbeg = None, fend = None):
        """
        Grab a region of the waterfalll data
        Parameters:
        tbeg : float
               starting time in hours
        tend : float
               stopping time in hours
        fbeg : float
               starting frequency in GHz
        fend : float
               Stopping frequency in GHz

        Returns:
        A sliced waterfall data
        """
        if tbeg is None: # If input parameters not specified, use the original start and stop values
            tbeg = self.radio_times[0]
        if tend is None:
            tend = self.radio_times[-1]
        if fbeg is None:
            fbeg = self.radio_freqs[0]
        if fend is None:
            fend = self.radio_freqs[-1]

        assert tbeg >= self.radio_times[0]  and tend <= self.radio_times[-1] #check if input time and freq values fall within the observed range
        assert fbeg >= self.radio_freqs[0] and fend <= self.radio_freqs[-1]
        
        delt = self.radio_times[1] - self.radio_times[0]  # time resolution
        delf = self.radio_freqs[1] - self.radio_freqs[0]  # frequency resolution
        tin = self.radio_times[0] # starting time
        fin = self.radio_freqs[0] # starting frequency
        
        tind_beg = int((tbeg - tin)/delt)  # corresponding time and frequency index
        tind_end = int((tend - tin)/delt)

        find_beg = int((fbeg - fin)/delf)
        find_end = int((fend - fin)/delf)

        return self.radio_wf[tind_beg:tind_end, find_beg:find_end], self.radio_times[tind_beg:tind_end], self.radio_freqs[find_beg:find_end]

    def search_timeseries(self, data, times, freqs, outdir, plot_hist = True):
        """
        Search for single pulses in time series, writes out a text file with detections and
        plot a histogram of SNR
        Parameters:
        ----------
        data: array
              waterfall data
        times: array
               time data
        freqs : array
                frequency data
        outdir : str
                    folder to save the plot
        plot_hist: bool
                 flag to plot histogram

        Returns:
        --------
        A text file with detections and 
        a histogram plot
        """
        series = np.nanmean(10**(data/10.0), axis = 1) # average in frequency to get time series

        med_series = np.median(series) # median of the series
        sigma_series = mad(series)    # median absolute deviation of the series

        if sigma_series != 0:    # zero values can lead to zero mads. Check that
            snr_series = np.abs(series - med_series)/sigma_series # Calculate SNR of each integration
        else:
            sys.exit("MAD zero, check the dataset") # Stop the function for zero MADs

        cand = np.where((snr_series > 5.0))[0] # Indices of series with SNR > 5.0
        print(f"Total detection of strong pulses within frequency range: {freqs[0]} - {freqs[-1]}: {len(cand)}")
        
        # writing results into a text file
        with open(f"{outdir}/pulse_detection.txt", "w") as fh:
            fh.write("Time_integration Time_hrs SNR\n")
            for n in cand:
                fh.write(f"{n} {times[n]} {snr_series[n]}\n")
            fh.close()
        
        if plot_hist: # Plot histogram of SNR
            fig = plt.figure(figsize = [12,8])
            plt.hist(np.log10(snr_series[cand]), 50)
            plt.ylabel("Count")
            plt.xlabel(r"$\rm Log_{10}$(SNR)")
            plt.show()

     
    def down_channelize_data(self, data, times, freqs, outdir, outname, logged, factor, savefig = False):
        """
        Down channelize the data, plot waterfall, spectra and time series.
        Print statistic as well
        Parameters:
        -----------
        data: array
              waterfall data
        times: array
               time data
        freqs : array
                frequency data
        outdir : str
                    folder to save the plot
        outname : str
                    plot name to save
        logged : bool
                 True: plot everything in decibels
                 False: plot in mW
        factor : int
                 Channelizing factor
        savefig: bool
                 flag to save image

        Returns:
        --------
        plot the output files and prints statistics

        """
        data = 10**(data/10.0) # converts to mW
        fnew = int(len(freqs)/factor) # new freq axis after averaging
        data_down = np.zeros((data.shape[0], fnew)) # shape of array after averaging
        freqs_down = np.zeros(fnew) # new frequency array
        for f in range(fnew):
            data_down[:,f] = 10.0*np.log10(np.nanmean(data[:,f*factor:(f+1)*factor], axis = 1)) # average waterfall data
            freqs_down[f] = np.nanmean(freqs[f*factor:(f+1)*factor])  # average frequencies
        
        plot_all_waterfall(data_down, times, freqs_down, logged,'', outdir, outname, savefig) # plot everything 
        #self.plot_timeseries(data_down, times, freqs_down, outdir, outname, logged = False)
        #self.search_timeseries(data_down, times, freqs_down, outdir)
        self.get_waterfall_statistics(data_down, times, freqs_down) # get statistics of channelized data

   
    def plot_tel_sat_coor(self, sat_ind = [277, 279, 508]):
        """
        Plot telescope pointing and satellite coordinates (3)
        from integrations 100:1000 
        Parameters:
        ---------
        Index of 3 interesting satellites

        Returns:
        -------
        plot
        """
        plot_coord_polar(self.radio_tel_az[100:1000], self.radio_tel_el[100:1000], self.sat_az[100:1000, :], self.sat_el[100:1000, :], sat_ind)
        


def main(args): 
    
    #Load the radio and satellite data
    radio_data = np.load(args.dat_file)
    sat_coords = np.load(args.sat_file)

    #Class instance to load the radio and satellite data
    obj = search_emission(radio_data, sat_coords)

    #print metadata
    if args.meta:
        obj.print_metadata()
    
    #plot satellite coordinates 
    if args.plot_sat:
        obj.plot_tel_sat_coor()
    
    # Grabbing the right data 
    if args.cin:
        tbeg, tend, fbeg, fend = args.cin[:]
        data, times, freqs = obj.grab_wf(float(tbeg), float(tend), float(fbeg), float(fend))
    else:
        data, times, freqs = obj.grab_wf()
    
    # print waterfall stats
    if args.wf_stat:
        obj.get_waterfall_statistics(data, times, freqs)
    
    # plot files
    if args.plot:
        if 'a' in args.plot:
            obj.plot_all(data, times, freqs, args.out_dir, "all_logged.png", logged = True, savefig = True)
        if 'w' in args.plot:
            obj.plot_waterfall(data, times, freqs, args.out_dir, 'waterfall_logged.png', logged = True, savefig = True)
        if 't' in args.plot:
            obj.plot_timeseries(data, times, freqs, args.out_dir, 'tseries.png', logged = False, savefig = True)
        if 'i' in args.plot:
            obj.plot_spectra(data, times, freqs, args.out_dir, 'spectra_logged.png', logged = False, savefig = True)
    
    # find pulses in the data
    if args.find:
        obj.search_timeseries(data, times, freqs, args.out_dir, plot_hist= True)

    # calculates satellites within the beam
    if args.calc_sat:
        obj.calc_sat_within_beam(args.out_dir)
    
    # down channelize the data
    if args.down:
        obj.down_channelize_data(data, times, freqs, args.out_dir, "channelized_data.png", True, 250, savefig = False)


if __name__ == '__main__':

    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads radio observation and satellite coordinates in .npz format to produce diagnostic plots and calculate the statistics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dat_file', type = str, required = True, help = 'Radio observation in .npz format')
    parser.add_argument('-s','--sat_file', type = str, required = True, help = 'Satellite coordinates in .npz format')
    parser.add_argument('-o','--out_dir', type = str, required = True, help = 'Output directory to save the plots and write out text files')
    parser.add_argument('-m','--meta',  action = "store_true", help = 'Print basic metadata')
    parser.add_argument('-p','--plot', nargs='+', required = False ,  help=' plotting: w : waterfall, i : spectra, t : time series, a : all' )
    parser.add_argument('-c','--cin', nargs='+', required = False ,  help=' plotting: tbeg, tend, fbeg, fend, if given give all the entries' )
    parser.add_argument('-f','--find', action = "store_true", help = 'Find single pulses')
    parser.add_argument('-dc','--down', action = "store_true", help = 'Down channelize data')
    parser.add_argument('-cs','--calc_sat', action = "store_true", help = 'Calculate satellites with the primary beam')
    parser.add_argument('-ps','--plot_sat', action = "store_true", help = 'Plot satellites and telescope pointings')
    parser.add_argument('-ws','--wf_stat', action = "store_true", help = 'print waterfall statistics')
    args = parser.parse_args()
    main(args)
