import batman-package as batman
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const

# Get the solar radius
solar_radius = const.R_sun
jupiter_radius = const.R_jup
R_star=0.79*solar_radius

params = batman.TransitParams()
params.t0 = 0.                       #time of inferior conjunction
params.per = 1.75                    #orbital period
params.rp = 1.27*jupiter_radius/R_star                     #planet radius (in units of stellar radii)
params.a = 0.03*1.5e11/R_star                      #semi-major axis (in units of stellar radii)
params.inc = 85.35                   #orbital inclination (in degrees)
params.ecc = 0.                      #eccentricity
params.w = 90.                       #longitude of periastron (in degrees)
params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"       #limb darkening model

t = np.linspace(-0.05, 0.05, 100)

m = batman.TransitModel(params, t)    #initializes model
flux = m.light_curve(params)          #calculates light curve

plt.plot(t, flux)
plt.xlabel("Time from central transit")
plt.ylabel("Relative flux")
plt.show()