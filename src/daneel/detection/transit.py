import batman
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import yaml

# Function to load parameters from YAML
def load_parameters(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)




def plot_transit(input_pars):
    """
    Function to plot the transit light curve using batman model.
    Takes in input_pars, a dictionary of parameters from YAML.
    """

    # Define transit parameters from input_pars
    planet_name = input_pars['name']
    t0 = input_pars['t0']  # Time of inferior conjunction
    per = input_pars['per']  # Orbital period (days)
    rp = input_pars['rp']  # Planet radius (in units of stellar radii)
    a = input_pars['a']  # Semi-major axis in stellar radii
    inc = input_pars['inc']  # Orbital inclination (degrees)
    ecc = input_pars['ecc']  # Eccentricity
    w = input_pars['w']  # Longitude of periastron (degrees)
    Rstar=input_pars['R_star']
    u_list = input_pars['u']  # Limb darkening coefficients
    limb_dark = input_pars['limb_dark']  # Limb darkening model

    # Constants (only needed if you need them for other parts of the code)
    solar_radius = const.R_sun.value  # in meters
    jupiter_radius = const.R_jup.value  # in meters
    

    # Time array for light curve calculation
    t = np.linspace(-0.06, 0.06, 1000)

    # Initialize the transit model and calculate the light curve
    

    # Plot the light curve
    batman_params = batman.TransitParams()

    plt.figure()

    for i in range(len(rp)):
        R_star = Rstar[i] * solar_radius  # Stellar radius of WASP-52 in meters
        batman_params.t0 = t0[i]
        batman_params.per = per[i]
        batman_params.a = a[i]* 1.496e11 / R_star
        batman_params.inc = inc[i]
        batman_params.ecc = ecc[i]
        batman_params.w = w[i]
        batman_params.u = u_list[i]  # Use the ith set of limb darkening coefficients
        batman_params.limb_dark = limb_dark


        batman_params.rp = rp[i]* jupiter_radius/ R_star 

        m = batman.TransitModel(batman_params, t)
        flux = m.light_curve(batman_params)

    
        label = f"Planet Radius = {rp[i]:.3f} Stellar Radii"
        plt.plot(t, flux, label=label)
        
        
    plt.legend()
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.ylim((0.880, 1.005))
    plt.title(f"Transit Light Curve {planet_name}")
    plt.grid()

    filename = input("Enter the filename to save the figure (e.g., 'lightcurve.png'): ")
    plt.savefig(filename) 
    plt.show()
   


