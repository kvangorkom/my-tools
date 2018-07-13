import numpy as np

def flexible_fft(data, dt, normalize=True, apply_window=True, onesided=True, return_psd=False):
    '''
    Perform an FFT of a column or multiple columns of data with the
    options of windowing, returning the one-sided FFT, and properly
    normalizing to preserve energy.
    Parameters:
        data : 1d or 2d nd array
            N or N x k array, where k is the number of columns.
        dt : float
            data timestep
        normalize : bool
            Apply normalization(s)?
        window : bool
            Apply time-series Hanning window before FFTing?
        onesided : bool
            Return only positive frequency values?
    Returns:
        freq, fft
    '''

    n = data.shape[0]
    #always compute norm and window but only apply if requested
    norm = 1.
    window = 1.
    
    if apply_window:
        window = np.hanning(n)
        norm *= np.mean(window)
        if np.ndim(data) > 1.:
            window = window[:,None]

    #transform
    fft = np.fft.fft(data * window, axis=0)
    freq = np.fft.fftfreq(n,d=dt)

    if onesided:
        pos_freq = np.where(freq >= 0.)
        fft = fft[pos_freq]
        freq = freq[pos_freq]
        norm /= np.sqrt(2)

    if normalize:
        norm *= n
        fft /= norm

    if return_psd:
        return freq, ( fft * np.conjugate(fft)).real / freq[1]
    
    return freq, fft

def han2d(shape, fraction=1./np.sqrt(2), normalize=False):
    '''
    Radial Hanning window scaled to a fraction 
    of the array size.
    
    Fraction = 1. for circumscribed circle
    Fraction = 1/sqrt(2) for inscribed circle (default)
    '''
    #return np.sqrt(np.outer(np.hanning(shape[0]), np.hanning(shape[0])))

    # get radial distance from center
    radial = get_radial_dist(shape)

    # scale radial distances
    rmax = radial.max() * fraction
    scaled = (1 - radial / rmax) * np.pi/2.
    window = np.sin(scaled)**2
    window[radial > fraction * radial.max()] = 0.
    return window

def radial_psd(surface, window=None, sampling=1., nbins=200, return_2dpsd=False):
    '''
    Compute the 2D PSD of a surface and return the radial average

    Units: [z units]^2 / freq^2

    Parameters:
        surface : 2D array-like
            Surface to find PSD of
        window : 2d array-like, opt.
            2D window to minimize spectral leakage (for
            example, the output of han2d)
        sampling : float, opt.
            Pixel sampling [physical surface width / n pixels]
        nbins : int
            Number of radial bins for the radial average. This
            should be a factor of several smaller than surface
            shape
        return_2dpsd : bool, opt. (Default: False)
            Return the 2D PSD instead of the radial average?

    Returns:
        frequency : 1D array-like
            radial frequencies at which the PSD is reported
        psd_ravg : 1D array-like
            Radially-averaged PSD values [z units^2] / freq^2 
    '''

    if window is None:
        window = np.ones_like(surface)

    # frequency spacing
    ny, nx = surface.shape
    deltaky = 1. / sampling / ny
    deltakx = 1. / sampling / nx
    
    # take the fft
    fft2d = np.fft.fftshift(np.fft.fft2(surface * window))
    
    # brute force window normalization factor by matching
    # rms values before and after window (in spatial domain)
    surface_rms = analysis.rms(surface)
    windowed_rms = analysis.rms(surface * window)
    factor = surface_rms / windowed_rms * 1. / (nx * ny)
    fft2d *= factor

    # square modulus before averaging
    psd = (fft2d * np.conjugate(fft2d)).real
    
    if return_2dpsd:
        #deltak = np.sqrt(deltaky**2 + deltakx**2) # I don't know if this is the right thing to do
        return deltakx, psd / deltaky / deltakx
    
    # get radial distances and divide into nbins sets
    radial = get_radial_dist((ny, nx), scaleyx=(deltaky, deltakx)) # frequency space
    bins = np.linspace(0, radial.max(), num=nbins, endpoint=True)
    digrad = np.digitize(radial, bins)
    
    # average over radial bins and then multiply by 2 for one-sided PSD
    #psd_ravg = np.asarray([np.sum(psd[digrad == i]) / (np.pi * i**2) for i in np.unique(digrad)]) * 2.0
    psd_ravg = np.asarray([np.mean(psd[digrad == i]) for i in np.unique(digrad)]) * 2.0
    
    #normalize by frequency spacing
    psd_ravg /= deltaky * deltakx #/= bins[1]
    
    return bins, psd_ravg

def get_radial_dist(shape, scaleyx=(1.0, 1.0)):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial