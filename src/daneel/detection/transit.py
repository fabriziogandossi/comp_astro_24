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
    t = np.linspace(-0.06, 0.06, 1000)

    # Initialize the transit model and calculate the light curve
    radius=[rp]
    radii = [rp, rp / 2]
    radii_2 = [rp, rp / 2,2*rp]

    # Plot the light curve
    batman_params = batman.TransitParams()
    batman_params.t0 = t0
    batman_params.per = per
    batman_params.a = a
    batman_params.inc = inc
    batman_params.ecc = ecc
    batman_params.w = w
    batman_params.u = u
    batman_params.limb_dark = limb_dark

   
    plt.figure()

    for rp in radius:

        batman_params.rp = rp

        m = batman.TransitModel(batman_params, t)
        flux = m.light_curve(batman_params)

    
        label = f"Planet Radius = {rp:.3f} Stellar Radii"
        plt.plot(t, flux, label=label)
        
        
    plt.legend()
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.ylim((0.968, 1.005))
    plt.title("Transit Light Curve WASP-52 b")
    plt.grid()
    plt.savefig("assignment2_taskA.png") 
    plt.show()



    plt.figure()

    for rp in radii:

        batman_params.rp = rp

        m = batman.TransitModel(batman_params, t)
        flux = m.light_curve(batman_params)

    
        label = f"Planet Radius = {rp:.3f} Stellar Radii"
        plt.plot(t, flux, label=label)
        
        
    plt.legend()
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.ylim((0.968, 1.005))
    plt.title("Transit Light Curve WASP-52 b")
    plt.grid()
    plt.savefig("assignment2_taskB.png") 
    plt.show()
   

    plt.figure()

    for rp in radii_2:

        batman_params.rp = rp

        m = batman.TransitModel(batman_params, t)
        flux = m.light_curve(batman_params)

        
        label = f"Planet Radius = {rp:.3f} Stellar Radii"
        plt.plot(t, flux, label=label)
            
        
    plt.legend()
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.ylim((0.880, 1.005))
    plt.title("Transit Light Curve WASP-52 b")
    plt.grid()
    plt.savefig("assignment2_taskC.png") 
    plt.show()
    
