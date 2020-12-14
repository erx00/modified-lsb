import numpy as np

def gaborfilter(theta, wavelength=12, phase=0, sigma=3, aspect=2, ksize=None):

	"""
	GB = GABORFILTER(THETA, WAVELENGTH, PHASE, SIGMA, ASPECT, KSIZE)
	creates a Gabor filter GB with orientation THETA (in radians),
	wavelength WAVELENGTH (in pixels), phase offset PHASE (in radians),
	envelope standard deviation SIGMA, aspect ratio ASPECT, and dimensions
	KSIZE x KSIZE. KSIZE is an optional parameter, and if omitted default
	dimensions are selected.
	"""

	if ksize is None:
	 	ksize = 8*sigma*aspect


	if type(ksize) == int or len(ksize) == 1:
	 	ksize = [ksize, ksize]


	xmax = np.floor(ksize[1]/2.)
	xmin = -xmax
	ymax = np.floor(ksize[0]/2.)
	ymin = -ymax

	xs, ys = np.meshgrid(np.arange(xmin,xmax+1), np.arange(ymax,ymin-1,-1))

	xrs = np.cos(theta)*xs + np.sin(theta)*ys
	yrs = -1*np.sin(theta)*xs + np.cos(theta)*ys

	g = np.sin((2*np.pi/wavelength)*yrs + phase) * \
		np.exp((np.power(xrs/aspect, 2) + np.power(yrs, 2))/(-2*np.power(sigma, 2)))

	# sum equal to 0
	g = g - np.mean(g)

	# sum of squares equal to 1
	g = g / np.sqrt(np.sum(np.power(g, 2)))

	return g
