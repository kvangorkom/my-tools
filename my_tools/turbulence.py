import numpy as np
from .signal import get_radial_dist

def von_Karman_mod_psd(r0, L0, l0, f):
    '''
    Generate a two-sided 2D von Karman PSD, accounting
    for inner and outer scales.
    
    If generating for a one-sided frequency array, I
    believe you'll have to multiply by a factor of 2
    to preserve the total power under the PSD.
    
    Parameters:
        r0 : float
            Fried's coherence length (meters)
        L0 : float
            Outer scale frequency (1/meter)
        l0 : float
            Inner scale frequency (1/meter)
        f : array-like
            Grid of frequencies to evalute the PSD
            at. (1/meter)
    Returns:
        PSD in units of rad^2/meters^-2
    '''
    fm = 5.92 / l0 / (2 * np.pi) # inner scale frequency [1/m]
    f0 = 1. / L0 # outer scale frequency [1/m]
    return 0.023 * r0**(-5.0/3.0) * np.exp(-(f/fm)**2) * (f**2 + f0**2)**(-11.0/6.0) 

def von_Karman_fft_phase_screen(N, delta, r0, L0, l0):
    '''
    Generate a von Karman phase screen using the FFT
    approach. This is essentially a direct translation of
    MATLAB code in Schmidt's  Numerical Simulation of 
    Optical Wave Propagation.
    
    Note: this approach doesn't reproduce low frequency
    content accurately. Use
    von_Karman_fft_subharmonics_phase_screen instead.
        
    Parameters:
        N : int
            Size of one-axis of the square phase
            screen to generate (pixels)
        delta : float
            Sampling of output array (meter/pixel)
        r0 : float
            Fried's coherence length (meters)
        L0 : float
            Outer scale frequency (1/meter)
        l0 : float
            Inner scale frequency (1/meter)
    Returns:
        NxN phase screen in units of radians
    '''
    # setup the PSD
    del_f = 1. / (N * delta) # frequency grid spacing [1/m]
    
    # frequency grid [1/m]
    f = get_radial_dist((N, N), scaleyx=(del_f, del_f))

    # modified von Karman atmospheric phase PSD
    PSD_phi = von_Karman_mod_psd(r0, L0, l0, f) 
    
    # random draws of Fourier coefficients
    noise = (np.random.normal(size=(N,N)) + 1j*np.random.normal(size=(N,N))) #/ np.sqrt(2)
    cn = noise  * np.sqrt(PSD_phi) * del_f # synthesize the phase screen
    phz = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(cn))).real * N**2
    
    # I remove the piston term in the spatial domain to accommodate even-sized
    # arrays for which there is no (0, 0) frequency pixel in the 2D PSD
    return phz - phz.mean()

def von_Karman_fft_subharmonics_phase_screen(N, delta, r0, L0, l0):
    '''
    Generate a modified von Karman phase screen via
    the FFT approach and then add low-frequency correction
    by accumulating subharmonics.
    
    Essentially a direct translation of the MATLAB approach
    in Schmidt's Numerical Simulation of Optical Wave Propagation 
        
    Parameters:
        N : int
            Size of one-axis of the square phase
            screen to generate (pixels)
        delta : float
            Sampling of output array (meter/pixel)
        r0 : float
            Fried's coherence length (meters)
        L0 : float
            Outer scale frequency (1/meter)
        l0 : float
            Inner scale frequency (1/meter)
    Returns:
        NxN phase screen in units of radians
    '''

    D = N * delta
    
    # high-frequency screen from FFT method
    phz_hi = von_Karman_fft_phase_screen(N, delta, r0, L0, l0)
    
    # spatial grid [m]
    xi = np.linspace(-N / 2., N / 2., num=N, endpoint=False) * delta
    y, x = np.meshgrid(xi, xi)
    # initialize low-freq screen
    phz_lo = np.zeros_like(phz_hi)

    # loop over frequency grids with spacing 1/(3^p*L)
    for p in range(1,4):
        # set up the PSD
        del_f = 1 / (3.0**p * D); #frequency grid spacing [1/m]
        fx = np.asarray([-1, 0, 1]) * del_f
        # frequency grid [1/m]
        fy, fx = np.meshgrid(fx, fx)
        f = get_radial_dist((3, 3), scaleyx=(del_f, del_f))
        
        # modified von Karman atmospheric phase PSD
        PSD_phi = von_Karman_mod_psd(r0, L0, l0, f)
        PSD_phi[1,1] = 0.
        
        # random draws of Fourier coefficients
        cn = (np.random.normal(size=(3,3)) + 1j*np.random.normal(size=(3,3))) * np.sqrt(PSD_phi) * del_f
        SH = np.zeros((N,N))
        
        # loop over frequencies on this grid to generate subharmonics
        for ii in range(9):
            SH = SH + cn.flat[ii] * np.exp(1j*2*np.pi * (fx.flat[ii]*x + fy.flat[ii]*y) )
        phz_lo = phz_lo + SH # accumulate subharmonics
    
    # remove piston
    phz_lo = phz_lo.real - np.mean(phz_lo.real)
    
    # add subharmonics to high-frequency screen
    return phz_hi + phz_lo # radians