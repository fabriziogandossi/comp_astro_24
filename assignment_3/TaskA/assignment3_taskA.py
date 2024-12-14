import matplotlib.pyplot as plt
%matplotlib widget
import numpy as np
import sys
import taurex.log
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Guillot2010, Isothermal
from taurex.contributions import RayleighContribution, AbsorptionContribution
from taurex.binning import FluxBinner,SimpleBinner

#cross-sections
from taurex.cache import OpacityCache,CIACache
OpacityCache().clear_cache()
OpacityCache().set_opacity_path("path_to_xsecs")
CIACache().set_cia_path("path_to_hitran")

# Define the planetary parameters
planet_radius = 1.27  # Jupiter radii
planet_mass = 0.46     #Jupyter masses
star_radius = 0.79    # Solar radii
T_irr = 1200.0        # Irradiation temperature (K)
T_star = 5000.0       # Stellar temperature (K)

guillot = Guillot2010(T_irr=T_irr)   # Create temperature profile

# Create a planet and a star instance
planet = Planet(planet_radius=planet_radius, planet_mass=planet_mass)  
star = BlackbodyStar(temperature=T_star, radius=star_radius)

chemistry = TaurexChemistry()  #defining the chemistry
# Define and randomize gas abundances for H2O, CH4, CO2, and CO
gases = ['H2O', 'CH4', 'CO2', 'CO']

for gas_name in gases:
    abundance = 10**np.random.uniform(-8, -3)  # Randomize abundance within the specified range
    chemistry.addGas(ConstantGas(gas_name, mix_ratio=abundance))
    print(gas_name, abundance)

#Initialize the model
tm = TransmissionModel(planet=planet,
                        chemistry=chemistry, 
                        star=star, 
                        atm_min_pressure=1e-0, 
                        atm_max_pressure=1e6, 
                        nlayers=30),
                        temperature_profile=guillot)

#Adding the contributions
tm.add_contribution(AbsorptionContribution())
tm.add_contribution(RayleighContribution())

tm.build()  #building the model

full_fig = plt.figure()  #plotting the spectrum
plt.plot(np.log10(10000/native_grid),rprs**2) #in micrometers, absorptio
#plt.plot((native_grid)**-1,rprs)
plt.title('Absorption Profile of the Planet WASP-52 b')
plt.xlabel('log($\mu$m)')
#plt.xlim([2.5,5.1])
plt.ylabel('Absorbed relative flux')
plt.show()


binned_fig = plt.figure()
wngrid = np.sort(10000/np.logspace(-0.5,1.7,200))  #Make a logarithmic grid
bn = SimpleBinner(wngrid=wngrid)
bin_wn, bin_rprs,_,_  = bn.bin_model(tm.model(wngrid=wngrid))

error=np.std(bin_rprs)/2 àcalculating the error from the standard deviation

plt.errorbar(np.log10(10000/bin_wn), bin_rprs, error)
#plt.xscale('log')
plt.title('Absorption Profile of the Planet WASP-52 b')
plt.xlabel('log($\mu$m)')
#plt.xlim([2.5,5.1])
plt.ylabel('Absorbed relative flux')
plt.savefig('WASP-52_b_assignment3_taskA_spectrum.png')
plt.show()

# Convert the native grid (wavenumber) to wavelength in microns
wavelength = 10000 / native_grid  # Convert from cm^-1 to microns

# Calculate (Rp/Rs)^2 and sqrt((Rp/Rs)^2)
rprs_squared = bin_rprs
rprs_sqrt = [error]*len(bin_rprs)

# Combine the data into a single array
spectrum_data = np.column_stack((10000/bin_wn , rprs_squared, rprs_sqrt))

#Saving the file so the gas abundances for the specific running are written in the header
# Extract the active gases and their abundances
gases = tm.chemistry.activeGases
abundances = tm.chemistry.activeGasMixProfile[:, 0]  # Assuming column 0 contains the relevant abundances

# Format the gases and abundances into a string
gases_header = ", ".join([f"{gas}: {abundance:.2e}" for gas, abundance in zip(gases, abundances)])

# Create the header
header = f"Wavelength(micron) (Rp/Rs)^2 sqrt((Rp/Rs)^2), Gases and abundances: {gases_header}"

# Save the data to a file
planet_name = "WASP-52b"  # Replace with your planet's name
file_name = f"{planet_name}_assignment3_taskA_spectrum_prova.dat"
np.savetxt(file_name, spectrum_data, header=header, fmt="%.6e")

print(f"Spectrum saved to {file_name}")



