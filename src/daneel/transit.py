import batman
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import yaml

# Function to load parameters from YAML
def load_parameters(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Constants (only needed if you need them for other parts of the code)
solar_radius = const.R_sun.value  # in meters
jupiter_radius = const.R_jup.value  # in meters
R_star = 0.79 * solar_radius  # Stellar radius of WASP-52 in meters (estimated)

# Path to your YAML file (replace with actual path if needed)
yaml_file_path = '/home/silvia/Desktop/Magistrale/Esami_da_dare/Computational_astrophysics/comp_astro_24_prova/src/daneel/transitparameters.yaml'

# Load parameters from the YAML file
params = load_parameters(yaml_file_path)

# Print loaded parameters to check
print("Loaded Parameters:")
print(params)

def plot_transit(input_pars):
    """
    Function to plot the transit light curve using batman model.
    Takes in input_pars, a dictionary of parameters from YAML.
    """

    # Define transit parameters from input_pars
    t0 = input_pars['t0']  # Time of inferior conjunction
    per = input_pars['per']  # Orbital period (days)
    rp = (input_pars['rp']* jupiter_radius) / R_star  # Planet radius (in units of stellar radii)
    a = (input_pars['a'] * 1.496e11) / R_star # Semi-major axis in stellar radii
    inc = input_pars['inc']  # Orbital inclination (degrees)
    ecc = input_pars['ecc']  # Eccentricity
    w = input_pars['w']  # Longitude of periastron (degrees)
    u = input_pars['u']  # Limb darkening coefficients
    limb_dark = input_pars['limb_dark']  # Limb darkening model

    # Time array for light curve calculation
    t = np.linspace(-0.05, 0.05, 1000)

    # Initialize the transit model and calculate the light curve
    batman_params = batman.TransitParams()
    batman_params.t0 = t0
    batman_params.per = per
    batman_params.rp = rp
    batman_params.a = a
    batman_params.inc = inc
    batman_params.ecc = ecc
    batman_params.w = w
    batman_params.u = u
    batman_params.limb_dark = limb_dark

    m = batman.TransitModel(batman_params, t)
    flux = m.light_curve(batman_params)

    # Plot the light curve
    plt.plot(t, flux)
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.ylim((0.970, 1.001))
    plt.title("Transit Light Curve")
    plt.grid()
    plt.savefig("transit_light_curve.png") 
    plt.show()
    print('ho funzionatoooo')
    
