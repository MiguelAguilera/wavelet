import numpy as np
from numpy import pi,linspace,exp,sqrt
from numpy import math
import matplotlib.pyplot as plt

def cwt(x,wavelet,width):
	"""
	Continuous wavelet transform.

	Performs a continuous wavelet transform on `data`,
	using the `wavelet` function. A CWT performs a convolution
	with `data` using the `wavelet` function, which is characterized
	by a width parameter and length parameter.

	Parameters
	----------
	data : (N,) ndarray
	data on which to perform the transform.
	wavelet : function
	Wavelet function, which should take 2 arguments.
	The first argument is the number of points that the returned vector
	will have (len(wavelet(width,length)) == length).
	The second is a width parameter, defining the size of the wavelet
	(e.g. standard deviation of a gaussian). See `ricker`, which
	satisfies these requirements.
	widths : (M,) sequence
	Widths to use for transform.

	Returns
	-------
	cwt: (M, N) ndarray
	Will have shape of (len(data), len(widths)).

	Notes
	-----
	>>> length = min(10 * width[ii], len(data))
	>>> cwt[ii,:] = scipy.signal.convolve(data, wavelet(length,
	... width[ii]), mode='same')

	Examples
	--------
	>>> from scipy import signal
	>>> sig = np.random.rand(20) - 0.5
	>>> wavelet = signal.ricker
	>>> widths = np.arange(1, 11)
	>>> cwtmatr = signal.cwt(sig, wavelet, widths)
	"""
	
	y=np.zeros([len(width), len(x)]).astype(complex)
	for i in range(len(width)):
		w = wavelet(min(10 * width[i], len(x)),width[i])
		y[i,:]=np.convolve(np.array(x), w, mode='same')
	return y

def bandpls(cwt1,cwt2,widths,eps=1,nc=8):
	"""
	Wavelet phase-lock analysis
	Detects transent moments of phase locking at different scales by
	comparing instantaneous frequencies for a scaling value
	""" 
	N=len(cwt1)
	dph=np.angle(cwt1)-np.angle(cwt2)
	dph+=(np.abs(cwt1)*np.abs(cwt2)==0).astype(int)*np.random.rand(N)*2*pi
	dz=np.exp(1j*dph)
	f=eps/widths
	d=np.ceil(nc/f)
	
	win=np.ones(d)/d
	phi=np.convolve(dz,win, mode='same')
	r=np.abs(phi)
	ph=np.mod(np.angle(phi)+pi,2*pi)-pi

	return r,ph
	
def pls(cwt1,cwt2,widths,eps=1,nc=8):
	"""
	Wavelet phase-lock analysis
	Detects transent moments of phase locking at different scales by
	comparing instantaneous frequencies for each scaling value
	""" 
	(M,N)=cwt1.shape
	r=np.zeros((M,N))
	ph=np.zeros((M,N))
	for s in range(M):
		(r[s,:],ph[s,:])=bandpls(cwt1[s,:],cwt2[s,:],widths[s],eps,nc)
			
	return r,ph

def morlet(M, s=1.0, f=1.0, complete=True):
	"""
	Complex Morlet wavelet.

	Parameters
	----------
	M : int
	Length of the wavelet.
	w : float
	Omega0. Default is 5
	s : float
	Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
	complete : bool
	Whether to use the complete or the standard version.

	Returns
	-------
	morlet : (M,) ndarray

	See Also
	--------
	scipy.signal.gausspulse

	Notes
	-----
	The standard version::

	pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

	This commonly used wavelet is often referred to simply as the
	Morlet wavelet. Note that this simplified version can cause
	admissibility problems at low values of w.

	The complete version::

	pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

	The complete version of the Morlet wavelet, with a correction
	term to improve admissibility. For w greater than 5, the
	correction term is negligible.

	Note that the energy of the return wavelet is not normalised
	according to s.

	The fundamental frequency of this wavelet in Hz is given
	by ``f = 2*s*w*r / M`` where r is the sampling rate.

	"""
	
	x = linspace(-M/2.0, M/2.0, M)
	output = exp(1j * 2*pi*f * (x/s))
	if complete:
		output -= exp(-0.5 * ((2*pi*f)**2))
		
	output *= exp(-0.5 * ((x/s)**2)) * pi**(-0.25)
	output /= sqrt(s)
	return output
    
def ricker(points, a):
    """
	Return a Ricker wavelet, also known as the "Mexican hat wavelet".

	It models the function:

	``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,

	where ``A = 2/sqrt(3a)pi^1/4``.

	Parameters
	----------
	points : int
	Number of points in `vector`.
	Will be centered around 0.
	a : scalar
	Width parameter of the wavelet.

	Returns
	-------
	vector : (N,) ndarray
	Array of length `points` in shape of ricker curve.

	Examples
	--------
	>>> from scipy import signal
	>>> import matplotlib.pyplot as plt

	>>> points = 100
	>>> a = 4.0
	>>> vec2 = signal.ricker(points, a)
	>>> print(len(vec2))
	100
	>>> plt.plot(vec2)
	>>> plt.show()

	"""
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


def paul(M, s=1.0, n=4):
    """
	Paul wavelet.

	Parameters
	----------
	M : int
	Length of the wavelet.
	n : int
	Order of the Paul wavelet. Default is 4
	s : float
	Scaling factor . Default is 1.

	Returns
	-------
	Paul : (M,) ndarray


	Notes
	-----

	"""
    x = linspace(-M/2.0, M/2.0, M)
    output = 2**n * math.factorial(n) * (1-1j*(x/s))**-(n+1)
    output /= 2*pi*sqrt(math.factorial(2*n)/2)
    output /= sqrt(s)
 
    return output
    
