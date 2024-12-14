import numpy as np
import matplotlib.pyplot as plt
import yaml
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Guillot2010, Isothermal
from taurex.cache import OpacityCache, CIACache
from taurex.binning import FluxBinner,SimpleBinner
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.optimizer.nestle import NestleOptimizer
from taurex.contributions import AbsorptionContribution, RayleighContribution, CIAContribution



class ForwardModel:
    def __init__(self, yaml_file):  #giving the .yaml file in input
        self.params = self.load_yaml(yaml_file)
        self.opacity_path = self.params["paths"]["opacity_path"]
        self.cia_path = self.params["paths"]["cia_path"]
        self.planet_params = self.params["planet"]
        self.star_params = self.params["star"]
        self.temperature_params = self.params["temperature"]
        self.chemistry_params = self.params["chemistry"]["gases"]
        self.output_file = self.params["output"]["file_name"]

    def load_yaml(self, yaml_file):
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file)

    def setup_paths(self):
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(self.opacity_path)
        CIACache().set_cia_path(self.cia_path)

    def create_model(self):
        self.setup_paths()

        # Define planet and star
        planet = Planet(
            planet_radius=self.planet_params["radius"],
            planet_mass=self.planet_params["mass"],
        )
        star = BlackbodyStar(
            temperature=self.star_params["temperature"],
            radius=self.star_params["radius"],
        )

        # Determine the temperature profile based on YAML configuration
        profile_type = self.temperature_params.get("profile", "guillot")  # Default to "guillot" if not specified

        if profile_type == "isothermal":
            # Create an isothermal profile (assuming you have a class or method for that)
            temperature_profile = Isothermal(T=self.temperature_params["T_irr"])
        elif profile_type == "guillot":
            temperature_profile = Guillot2010(T_irr=self.temperature_params["T_irr"])
        else:
            raise ValueError(f"Unknown temperature profile type: {profile_type}")

        # Set up chemistry with the elements and abundances from the .yaml file
        chemistry = TaurexChemistry()
        for gas_name, abundance in self.chemistry_params.items():
            try:
                abundance = float(abundance)  # Ensure mix ratio is a float
                chemistry.addGas(ConstantGas(gas_name, mix_ratio=abundance))
            except ValueError:
                raise ValueError(
                    f"Invalid abundance for {gas_name}: {abundance}. It must be a number."
                )

        # Create the transmission model
        model = TransmissionModel(
            planet=planet,
            temperature_profile=temperature_profile,  # Use the selected temperature profile
            chemistry=chemistry,
            star=star,
            atm_min_pressure=1e-2,
            atm_max_pressure=1e6,
            nlayers=30,
        )

        model.add_contribution(AbsorptionContribution())
        model.add_contribution(RayleighContribution())
        model.build()
        model.model()

        return model



    def save_spectrum(self, model):
        #res = model.model()
        #native_grid, rprs, tau, _ = res

        wngrid = np.sort(10000/np.logspace(-0.4,1.1,200))
        bn = SimpleBinner(wngrid=wngrid)
        bin_wn, bin_rprs,_,_  = bn.bin_model(model.model(wngrid=wngrid))

        error=np.std(bin_rprs)/2
        

        # Calculate (Rp/Rs)^2 and sqrt((Rp/Rs)^2)
        rprs_squared = bin_rprs
        rprs_sqrt = np.full_like(rprs_squared, error)

        
        # Combine the data into a single array
        spectrum_data = np.column_stack((10000/bin_wn , rprs_squared, rprs_sqrt))

        # Save the spectrum
        
        np.savetxt(
            self.output_file,
            spectrum_data,
            header="Wavelength(micron) (Rp/Rs)^2 sqrt((Rp/Rs)^2)",
            fmt="%.6e",
        )
        print(f"Spectrum saved to {self.output_file}")

    def run(self):
        model = self.create_model()
        self.save_spectrum(model)



class Retrieval:
    def __init__(self, config_path):
        """
        Initialize the Retrieval using a configuration file.

        Parameters:
        config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.planet = None
        self.star = None
        self.temperature_profile = None
        self.chemistry = None
        self.model = None
        self.optimizer = None
        self.obs = None

    def setup_environment(self):
        """
        Set up the opacity and CIA paths from the configuration.
        """
        opacity_path = self.config['paths']['opacity']
        cia_path = self.config['paths']['cia']

        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(opacity_path)
        CIACache().set_cia_path(cia_path)

    def setup_star_and_planet(self):
        """
        Initialize the star and planet instances.
        """
        star_params = self.config['star']
        self.star = BlackbodyStar(
            temperature=star_params['temperature'],
            radius=star_params['radius']
        )

        self.planet = Planet()

    def setup_temperature_profile(self):
        """
        Initialize the temperature profile based on the configuration.
        """
        temp_profile = self.config['atmosphere']['temperature_profile']
        temp_type = temp_profile['type']
        
        if temp_type == 'Guillot2010':
            self.temperature_profile = Guillot2010(T_irr=self.temperature_params.get('T_irr', 1500.0))  # Provide default if not specified
        elif temp_type == 'Isothermal':
            T = temp_profile.get('T', 1000)  # Replace with the default temperature if necessary
            self.temperature_profile = Isothermal(T)  # Initialize with temperature only
            
            # If `Isothermal` requires layers to be set, find the appropriate way to do that.
            # Check if there is a method to set layers or if it has a public attribute.
            self.temperature_profile.nlayers = self.config['atmosphere'].get('nlayers', 30)  # Set nlayers if allowed
            
        else:
            raise ValueError(f"Unsupported temperature profile type: {temp_type}")



    def setup_chemistry(self):
        # Retrieve gas names and abundances from the configuration
        chemistry = TaurexChemistry() 
        gas_name = self.config['chemistry']['gases']['names']
        gas_abundance = self.config['chemistry']['gases']['abundances']
        
        for i in range(len(gas_name)):
            chemistry.addGas(ConstantGas(gas_name[i], mix_ratio = gas_abundance[i]))


    def setup_model(self):
        """
        Build the transmission model with the provided components.
        """
        atm_params = self.config['atmosphere']

        self.model = TransmissionModel(
            planet=self.planet,
            chemistry=self.chemistry,
            star=self.star,
            atm_min_pressure=float(atm_params['pressure_min']),
            atm_max_pressure=float(atm_params['pressure_max']),
            nlayers=atm_params['nlayers'],
            temperature_profile=self.temperature_profile
        )

        # Add contributions
        self.model.add_contribution(RayleighContribution())
        self.model.add_contribution(AbsorptionContribution())
        self.model.build()
        self.model.model()

    def load_observed_spectrum(self):
        """
        Load the observed spectrum from the configuration.
        """
        spectrum_path = self.config['paths']['observed_spectrum']
        self.obs = ObservedSpectrum(spectrum_path)

    def setup_optimizer(self):
        """
        Initialize the optimizer and enable fitting parameters.
        """
        self.optimizer = NestleOptimizer(
            num_live_points=self.config['optimizer']['num_live_points'],
            method=self.config['optimizer']['method'],
            tol=self.config['optimizer']['tol']
        )

        self.optimizer.set_model(self.model)
        self.optimizer.set_observed(self.obs)

        for i in range(len(self.config['fit_parameters']['names'])):
            parameter=self.config['fit_parameters']['names'][i]
            boundary=self.config['fit_parameters']['boundaries'][i]
            self.optimizer.enable_fit(parameter)
            self.optimizer.set_boundary(parameter,boundary)

    def run(self):
        """
        Execute the retrieval process.
        """
        self.setup_environment()
        self.setup_star_and_planet()
        self.setup_temperature_profile()
        self.setup_chemistry()
        self.setup_model()
        self.load_observed_spectrum()
        self.setup_optimizer()
        solution = self.optimizer.fit()
        self.plot_observed_spectrum()
        self.plot_results(solution)

    def plot_observed_spectrum(self):
        """
        Plot the observed spectrum.
        """
        plt.figure()
        plt.errorbar(self.obs.wavelengthGrid, self.obs.spectrum, self.obs.errorBar, label='Observed Spectrum')
        plt.title('Observed Spectrum')
        plt.xscale('log')
        plt.xlabel('log($\mu$m)')
        plt.ylabel('Absorbed relative flux')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_results(self, solution):
        """
        Plot the best-fit model and residuals.
        """
        obin = self.obs.create_binner()
        for solution, optimized_map, _, _ in self.optimizer.get_solution():
            self.optimizer.update_model(optimized_map)
            model_spectrum = obin.bin_model(self.model.model(self.obs.wavenumberGrid))[1]
            residuals = self.obs.spectrum - model_spectrum

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.errorbar(self.obs.wavelengthGrid, self.obs.spectrum, yerr=self.obs.errorBar, label='Observed')
            ax1.plot(self.obs.wavelengthGrid, model_spectrum, label='Best Fit', color='r')
            ax1.set_xscale('log')
            ax1.set_title('Best-Fit Spectrum')
            ax1.legend()
            ax1.grid(True)

            ax2.errorbar(self.obs.wavelengthGrid, residuals, yerr=self.obs.errorBar, fmt='o')
            ax2.set_xscale('log')
            ax2.set_xlabel('log($\mu$m)')
            ax2.set_ylabel('Residuals')
            ax2.grid(True)

            plt.tight_layout()
            plt.show()


