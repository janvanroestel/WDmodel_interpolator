import numpy as np
import glob
import os
import scipy.ndimage.filters as filters



def bilinear_interpolation(x, y, X, Y):
    '''Interpolate (x,y) in a rectangular grid defined by X and Y

    input: 
        x : float, x position
        y : float, y position
        X : 1D-array of x-grid values
        Y : 1D-array of y-grid values

    output:
        tuple of corner points, x,y,w

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    # find corners
    x_1 = X[np.searchsorted(X,x,side='right')-1] 
    x_2 = X[np.searchsorted(X,x,side='right')] 
    y_1 = Y[np.searchsorted(Y,y,side='right')-1] 
    y_2 = Y[np.searchsorted(Y,y,side='right')] 

    # calculate weights
    dx1 = (x-x_1)/(x_2-x_1)
    dx2 = 1-dx1
    dy1 = (y-y_1)/(y_2-y_1)
    dy2 = 1-dy1

    w_a = dx2*dy2 # lower left
    w_b = dx1*dy2 # lower right
    w_c = dx2*dy1 # upper left
    w_d = dx1*dy1 # upper right 

    # return the corner values, x,y and weight
    return ([x_1,y_1,w_a],
            [x_2,y_1,w_b],
            [x_1,y_2,w_c],
            [x_2,y_2,w_d])





class WDmodels:
    
    def __init__(self,filedir,
            dw=None,smooth=0,wmin=None,wmax=None):
        """ Initial the class with the data. dw is the wavelength stepsize and
        smooth is the sigma of the Gaussian smoothing kernel
        

        input:
            filedir : str, the directory which contains the KoesterDA models. You 
                      can download them here
                      http://svo2.cab.inta-csic.es/theory/newov/index.php
                      don't change the names of the files
            dw      : float, wavelength step. Use None to keep the orginal grid
            smooth  : float, Gaussian sigma for convolution kernel. Check your 
                      instrument for what this value should be 
                      (be carefull to convert FWHM to sigma)
            wmin    : float, the mimimum wavelength for the model
            wmax    : float, the maximum wavelength for the model

        """

        # get the datafiles
        files = glob.glob(filedir+'*')
        files.sort()

        # get T and logg values
        T = np.array([float(os.path.basename(f)[2:7]) for f in files])
        logg = np.array([float(os.path.basename(f)[8:11]) for f in files])*0.01

        # load everything into ram; these spectra are on different grids...
        print('Loading model data...')
        modelspectra = [np.loadtxt(f) for f in files]
        N = np.array([np.size(d[:,0]) for d in modelspectra])

        # make the final grid of the spectrum
        w = np.loadtxt(files[np.argmax(N)],usecols=0)
        if wmin is None:
            wmin = np.min(w)
        if wmax is None:
            wmax = np.max(w)
        if dw is not None:
            # make a regular grid
            w = np.arange(np.min(w),np.max(w),dw)
        else:
            w = w[(w>wmin)*(w<wmax)]


        # smooth models
        if smooth>0:
            wf = np.arange(np.min(w),np.max(w),smooth/5.)
            k = smooth*5 # smoothing kernel for regular grid
            s = lambda y: filters.gaussian_filter(y,k)
            l = lambda d: np.interp(w,wf,s(np.interp(wf,d[:,0],d[:,1])))
        else:
            l = lambda d: np.interp(w,d[:,0],d[:,1])

        # store in dict for easy lookup
        print('Smoothing and interpolating...')
        modeldict = dict()
        for _T,_logg,m in zip(T,logg,modelspectra):
            modeldict[(_T,_logg)] = l(m)
        print('Done')

        self.w = w
        self.spectra = modeldict
        self.T_grid = np.sort(np.unique(T))
        self.logg_grid = np.sort(np.unique(logg))



    # for a given T and logg, return a spectrum
    def get_spectrum(self,T,logg,K=0,kernel=0):
        """ generate a spectrum by interpolating the model
        input:
            T    : float, temperature of the WD in K
            logg : float, surface gravity in log[cgs]
            K    : float, RV shift if km/s

        output:
            s    : 1D-array, the flux in 4pi*Eddington flux in erg/cm2/s/A
                   use self.w to get the wavelength grid

        """
        
        if T>np.max(self.T_grid) or T<np.min(self.T_grid):
            raise ValueError('%dK is out of bounds' %T)

        if logg>np.max(self.logg_grid) or logg<np.min(self.logg_grid):
            raise ValueError('%g is out of bounds' %logg)

        # check if requested T and logg are in the grid
        try:
            return self.spectra[(T,logg)]
        except:
            pass

        # do bilinear interpolation in T logg grid
        corners = bilinear_interpolation(T,logg,self.T_grid,self.logg_grid)
        s = np.array([c[2]*self.spectra[(c[0],c[1])] for c in corners])
        s = np.sum(s,axis=0)

        if K is not 0:
            wn = self.w * (1.+K/(3.*10**5))
            s = np.interp(self.w,wn,s) # resample to wl grid
        
        return s
