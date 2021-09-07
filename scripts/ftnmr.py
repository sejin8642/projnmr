# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import binom
from itertools import product
from scipy.stats import truncnorm
from scipy import interpolate

# Larmor angular frequency function
def larmor(B=1.5, unit='MHz'):
    """ Returns Larmor angular frequency based on external B field 

    Parameters
    ----------
    B: float
        Magnetic field in unit of Tesla
    unit: str
        Unit for ordinary frequency (default MHz)

    Returns
    -------
        Larmor angular frequency in either MHz or kHz
    """
    if unit=='MHz':
        return 267.52218744*B
    elif unit=='kHz':
        return 267.52218744*pow(10,3)*B
    else:
        raise ValueError("Frequency unit must be either MHz or kHz")

# molecule class 
class molecule(): 
    """
    Molecule class 

    This class contains hydrogen groups with the number and chemical shifts of each group.
    Based on J-coupling constants, spectral splits with their distribution is also created

    Attributes
    ----------
    hydrogens: dict[str]: (int, float)
        Dictionary of hydrogen groups with a number and a chemical shifts of the members
    couplings: list[(str, str, float)]
         List of J-couplings between two hydrogen groups. The last element of the tuple
         is the J-coupling, and the unit for it is Hz
    splits: 
    """

    # molecule constructor
    def __init__(
            self,
            hydrogens={'a':(1, 10.0)},
            couplings=[]):
        """ molecule constructor

        Parameters
        ----------
        hydrogens: dict[str]: (int, float)
            Dictionary of hydrogen groups with a number and a chemical shifts of the members
            (default a:(1, 10.0))
        couplings: list[(str, str, float)]
            List of J-couplings between two hydrogen groups. The last element of the tuple 
            is the J-coupling, and the unit for it is Hz (default None)
        splits: dict[str]: (array, array)
            Spectral splits for each hydrogen group and their probability distribution within each split
        """ 
        A = list(dict.fromkeys([k for b in couplings for k in b[:-1]]))
        B = {k:[ (b[2], hydrogens[ b[ 1-b.index(k) ] ][0]) for b in couplings if k in b] for k in A}

        C0 = {k:[ [n*b/2 for n in range(-d, d+1, 2) ] for b, d in B[k]] for k in A}
        C1 = {k:[ [binom(d, x)/pow(2,d) for x in range(0, d+1)] for b, d in B[k] ] for k in A}
        D0 = {k:[ sum(i) for i in product(*C0[k]) ] for k in A}
        D1 = {k:[ np.prod(i) for i in product(*C1[k]) ] for k in A}
        E0 = {k:[ D0[k][n] for n in np.argsort(D0[k]) ] for k in A}
        E1 = {k:[ D1[k][n] for n in np.argsort(D0[k]) ] for k in A}

        ind = lambda k: filter(lambda i: not np.isclose(E0[k][i-1], E0[k][i]), range(0, len(E0[k])))
        F0 = {k:[E0[k][i] for i in ind(k)] for k in A}
        F1 = {k:[E1[k][i] for i in ind(k)] for k in A}

        dup = lambda k: filter(lambda i: np.isclose(E0[k][i-1], E0[k][i]), range(0, len(E0[k])))
        for k in A:
            n = 0
            for i in dup(k):
                F1[k][i-1-n] += E1[k][i]
                n += 1
        
        # molecule constructor attributes
        self.hydrogens = hydrogens
        self.couplings = couplings
        self.splits = {k:(np.array(F0[k]), np.array(F1[k])) for k in A}

# spectrometer Class
class spectrometer():
    """
    Spectrometer class

    Attributes
    ----------
    B: float
        External magnetic field
    timeunit: str
        Unit for time variable, t
    shift_maximum: float
        Maximum chemical shift the spectrometer can observer
    p_l: int
        Power of two to narrow the frequency range of the processed signal
    shift_cutoff: float
        Highest shift of the frequency domain for the FFT
    f_unit: str
        Unit for ordinary frequency used
    ep: int
        Exponent of 10 to convert seconds into miliseconds or microseconds
    w_l: float
        angular Larmor frequency
    f_s: float
        Sampling frequency
    dt: float
        Sampling interval, also called timestep
    gamma: float
        Gyromagnetic ratio
    ns: integer
        Total number of signal samples (different from the number of FFT output)
    p: integer
        Power of two that yields the number of processed data points 
    t: numpy array[float]
        Times at which signal is sampled
    df: float
        Frequency resolution
    nf: int
        Number of FFT output
    f: numpy array[float]
        Frequency domain for FFT output
    shift: numpy array[float]
        Chemical shift doamin for FFT output
    hr: float
        One over number of hydrogens of the reference molecule, usually TMS which has 12 hydrogens
    std: float
        Standard deviation of signal noise
    noise: complex float
        Signal noise
    splits: list[tuple(float, float)]
        List of angular Larmor frequencies and relative abundances for sample molecules
    signal: numpy array[compelx float]
        NMR sample signal
    FFT: numpy array[complex float]
        FFT output of the signal
    spectra: numpy array[complex float]
        NMR spectra output with the artifact and noise
    spectra_artifact: numpy array[float]
        Spectra artifact

    Methods
    -------
    sampling_frequency()
        Returns a sampling frequency
    signal_frequency()
        Returns an adjusted signal frequency
    time()
        Returns a list of sampling times and sampling rate f_s
    signal_output()
        Returns FID signal
    call()
        Returns the sampled FID signal
    """

    # spectrometer class attributes
    gamma = 267.52218744*pow(10,6)
    ip2 = 0.5/np.pi

    # spectrometer constructor
    def __init__(
            self,
            B=10.0,
            timeunit='msec',
            shift_maximum=128.0,
            shift_minimum=15,
            t_cut=1500,
            f_min=0.2,
            RH=12,
            std=0.00005):
        """ spectrometer constructor

        Parameters
        ----------
        B: float
            External magnetic field (default 1.5 Tesla) 
        timeunit: str
            Unit string for time variable. It is either msec or micron (default msec)
        shift_maximum: float
            Maximum chemical shift to set the maximum frequency (default 128.0 ppm)
        shift_minimum: float
            Minimum chemical shift that the frequency range must include (default 15.0)
        t_cut: float
            Cutoff time that the maximum t value must exceed (default 1500.0)
        f_min: float
            Minimum frequency resolution for high resolution NMR spectrum (default 0.2 Hz)
        RH: integer
            Number of hydrogens the reference molecule contains (default 12)
        std: float
            Standard deviation of signal noise
        """

        # spectrometer constructor attributes
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift_maximum = shift_maximum
        self.p_l = int(np.log2(shift_maximum/shift_minimum))
        self.shift_cutoff = shift_maximum*pow(2, -self.p_l)
        self.f_unit, self.ep = self.unit(timeunit)
        self.w_l = self.gamma*B*pow(10, self.ep)
        self.f_s = self.ip2*shift_maximum*pow(10, -6)*self.w_l
        self.dt = 1/self.f_s
        self.ns, self.p, self.t = self.time(t_cut, f_min)
        self.df = self.f_s*pow(2, -self.p)
        self.nf = pow(2, self.p-self.p_l)
        self.f = self.df*np.arange(0, self.nf)
        self.shift = (self.shift_cutoff/self.nf)*np.arange(0, self.nf)
        self.hr = 1/RH
        self.std = std

    # spectrometer unit method
    def unit(self, timeunit):
        """ Unit method

        Returns frequency unit and time exponent that turns seconds into miliseconds or microseconds
        
        Parameters
        ----------
        timeunit: str
            Unit for time. It is either msec or micron
        
        Returns
        -------
        f_unit: str
            Unit for frequency
        ex: int
            Exponent of 10 to convert seconds into miliseconds or microseconds
        """

        if timeunit == 'msec':
            return 'kHz', -3
        elif timeunit == 'micron':
            return 'MHz', -6
        else:
            raise ValueError('incorrect time unit is specified: use msec or micron')
        
    # spectrometer time method
    def time(self, t_cut, f_min):
        """
        A list of times at which the signal is sampled with sampling interval and rate.

        Returns
        -------
        ns: integer
            Number of signal samples
        p: integer
            Power of two data points for sampled signal to be processed before clipping at
            cutoff chemical shift (usually more than ns, thus zero padding occurs)
        t: numpy array[float]
            Sampled times
        """
        p = int(np.log2(t_cut*self.f_s + 1)) + 1
        ns = pow(2, p) # total number of signal samples including t = 0 
        p += int(np.log2(self.f_s*pow(10, -self.ep)/(ns*f_min))) + 1
        return ns, p, np.arange(0, ns)/self.f_s

    # spectrometer calibrate method
    def calibrate(
            self,
            B=10.0,
            timeunit='msec',
            shift_maximum=128.0,
            shift_minimum=15,
            t_cut=1500,
            f_min=0.2,
            RH=12,
            std=0.00005):
        """
        Spectrometer calibrate method

        This method will calibrate spectrometer settings to default if no inputs were provided.
        It is essentially __init__. The parameters for this method is the same as the constructor
        """
        self.__init__(
                B=B,
                timeunit=timeunit,
                shift_maximum=shift_maximum,
                shift_minimum=shift_minimum,
                t_cut=t_cut,
                f_min=f_min,
                RH=RH,
                std=std)

    # spectrometer artifact method
    def artifact(self, baseline=False):
        """
        Artifact method

        For now, only baseline distortion artifact is implemented
        Parameters
        ----------
        Baseline: Bool
            If true, baseline distortion artifact is created

        Returns
        -------
        splev: numpy array[float]
            Linear or spline interpolation for baselinse distortion artifact
        """
        self.spectra_artifact = np.zeros(self.nf)

        if baseline:
            n = np.random.randint(2, 25)
            sd = 0.15
            w = 0.3/sd
            upper_bound = truncnorm(-w, w, loc=0.3, scale=sd).rvs(1)[0]
            y = np.random.uniform(0.0, upper_bound, n+1)
            if (2 < n) and (n < 21):
                bin_size = self.shift_cutoff/n
                std = bin_size/10
                b = 0.5*bin_size/std
                x = np.array(
                        [0]+
                        [truncnorm(-b, b, loc=bin_size*mu, scale=std).rvs(1)[0] for mu in range(1, n)]+
                        [self.shift_cutoff])
                tck = interpolate.splrep(x, y, s=0)
                self.spectra_artifact += interpolate.splev(self.shift, tck, der=0)
            else: 
                self.spectra_artifact += ( (y[-1] - y[0])/self.shift_cutoff*self.shift + y[0] )

    # spectrometer measure method
    def measure(self, moles, noise=True):
        """" 
        Measures FID signal from the sample
        
        Parameter
        ---------
        noise: bool
            If true, noise is introduced with std
        moles: dict[str]:(molecule, float)
            Sample object that contains molecules and T2, r, and timeunit
        """

        # relaxivities for corresponding hydrogen groups
        relaxivity = {x:{y: 1/moles[x][0].hydrogens[y][2] for y in moles[x][0].hydrogens} 
                     for x in moles}

        # Split frequencies and their relative abundance (relative to RH)
        A = [(
            moles[x][0].hydrogens[y][1]*pow(10, -6)*self.w_l,
            moles[x][1]*moles[x][0].hydrogens[y][0]*self.hr,
            relaxivity[x][y])
            for x in moles for y in moles[x][0].hydrogens if y not in moles[x][0].splits] \
        +   [(
            pow(10, -6)*moles[x][0].hydrogens[y][1]*self.w_l + 2*pow(10, self.ep)*np.pi*z,
            moles[x][1]*moles[x][0].hydrogens[y][0]*k*self.hr,
            relaxivity[x][y])
            for x in moles for y in moles[x][0].splits
            for z, k in zip(moles[x][0].splits[y][0], moles[x][0].splits[y][1])]
       
        # Final signal and its spectra (FFT of signal) from all hydrogen FID
        self.splits = A
        separate_fid = [self.dt*r*N*np.exp(1j*w*self.t)*np.exp(-r*self.t) for w, N, r in A] 

        self.noise = np.zeros(self.ns, dtype=np.complex128)
        if noise:
            self.noise += np.random.normal(0, self.std, self.ns)+ \
                          1j*np.random.normal(0, self.std, self.ns)

        self.signal = np.sum(separate_fid, axis=0) + self.noise
        self.FFT = np.fft.fft(self.signal, n=pow(2, self.p))[:self.nf]
        self.spectra = self.FFT + self.spectra_artifact

    def __repr__(self):
        return "Spectrometer class that measures a sample solution with organic molecules in it"

    def __call__(self):
        try:
            return self.spectra
        except AttributeError:
            return None

# Lorentzian Class
class lorentzian():
    """
    Absorption Lorentzian class

    This class will create an absorption Lorentzian profile with frequency domain

    Attributes
    ----------
    f: list[float]
    unit: str
        Unit string for ordinary frequency (default kHz)
    ns: integer
        Total number of frequencies (default pow(2, 15))
    r: float
        Relaxivity
    f0: float
        Ordinary Larmor frequency
    lorentz: list[float]
        Lorentzian function output

    Methods
    -------
    lorz()
        Returns the lorentz attribute
    """

    # Lorentzian constructor
    def __init__(
            self,
            unit='kHz',
            ns=pow(2,15),
            f_max=55.0,
            r=0.01,
            f0=4.25,
            f_l=425000,
            ob=object()):
        """
        Constructor

        Parameters
        ----------
        unit: str
            Unit string for ordinary frequency (default kHz)
        ns: integer
            Total number of frequencies (default pow(2, 15))
        f_max: float
            Maximum frequency for f-domain (default 55.0)
        r: float
            Relaxivity (default 0.01)
        f0: float
            Adjusted detected frequency (default 4.25)
        f_l: float
            Ordinary Larmor frequency (default 425000)
        ob: fid class
            fid object from which to extract its attributes (default object())
        """

        # Lorentzian object attributes
        if isinstance(ob, fid):
            self.unit = ob.frequency_unit
            self.ns = ob.ns
            self.p = ob.p
            self.f = np.arange(0, ob.ns)*ob.f_s/ob.ns # the last f excludes fmax
            self.cs = pow(10, 6)*self.f/ob.f_l
            self.r = ob.r
            self.f0 = ob.f0
            self.lorentz = self.lorz()
        else:
            self.unit = unit
            self.ns = ns
            self.p = np.log2(ns)
            self.f = np.arange(0, ns)*f_max/ns # the last f excludes f_max
            self.cs = pow(10, 6)*self.f/f_l
            self.r = r
            self.f0 = f0
            self.lorentz = self.lorz()

    # lorz method of Lorentzian
    def lorz(self):
        """
        Lorentzian function

        Returns
        -------
        lorentz: list[float]
            Lorentzian output
        """
        A = 2*np.pi*(self.f0 - self.f)
        B = pow(self.r, 2) + 4*pow(np.pi, 2)*pow((self.f - self.f0), 2)

        return self.r/B + 1j*A/B

    def __call__(self):
        """ returns lorentz """
        return self.lorentz

