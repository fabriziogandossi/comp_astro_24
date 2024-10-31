import batman
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const

# Constants
solar_radius = const.R_sun.value  # in meters
jupiter_radius = const.R_jup.value  # in meters
R_star = 0.79 * solar_radius  # Stellar radius of WASP-52 in meters (estimated)

# Define transit parameters for WASP-52 b
params = batman.TransitParams()
params.t0 = 0.                       # Time of inferior conjunction
params.per = 1.74977083               # Orbital period (days)
params.rp = (1.27 * jupiter_radius) / R_star  # Planet radius (in units of stellar radii)
params.a = (0.0272 * 1.496e11) / R_star  # Semi-major axis in meters, then in stellar radii
params.inc = 85.35                    # Orbital inclination (in degrees)
params.ecc = 0.                       # Eccentricity
params.w = 90.                        # Longitude of periastron (in degrees)
params.u = [0.1, 0.3]                 # Limb darkening coefficients
params.limb_dark = "quadratic"        # Limb darkening model

# Time array for light curve calculation
t = np.linspace(-0.05, 0.05, 1000)

# Initialize the transit model and calculate the light curve
m = batman.TransitModel(params, t)
flux = m.light_curve(params)

# Plot the light curve
plt.plot(t, flux)
plt.xlabel("Time from central transit (days)")
plt.ylabel("Relative flux")
plt.ylim((0.970, 1.001))
plt.title("Transit Light Curve of WASP-52 b")
plt.grid()
#plt.savefig("wasp52b_lightcurve.png")
plt.show()