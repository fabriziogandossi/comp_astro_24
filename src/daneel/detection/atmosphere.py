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

        # Define planet, star, and temperature profile
        planet = Planet(
            planet_radius=self.planet_params["radius"],
            planet_mass=self.planet_params["mass"],
        )
        star = BlackbodyStar(
            temperature=self.star_params["temperature"],
            radius=self.star_params["radius"],
        )
        guillot = Guillot2010(T_irr=self.temperature_params["T_irr"])

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
            temperature_profile=guillot,
            chemistry=chemistry,
            star=star,
            atm_min_pressure=1e-2,
            atm_max_pressure=1e6,
            nlayers=30,
        )

        model.add_contribution(AbsorptionContribution())
        model.add_contribution(RayleighContribution())
        model.build()

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
    def __init__(self, yaml_file):
        self.params = self.load_yaml(yaml_file)
        self.spectrum_file = self.params["paths"]["observed_spectrum"]
        self.output_file = self.params["output"]["retrieved_file"]
        self.planet_params = self.params["planet"]
        self.star_params = self.params["star"]
        self.temperature_params = self.params["temperature"]
        self.chemistry_params = self.params["chemistry"]
        self.fit_params = self.params["retrieval"]["fit_params"]

    def load_yaml(self, yaml_file):
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file)

    def setup_paths(self):
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(self.params["paths"]["opacity_path"])
        CIACache().set_cia_path(self.params["paths"]["cia_path"])

    def create_model(self):
        self.setup_paths()

        # Define planet, star, and temperature profile
        planet = Planet(
            planet_radius=self.planet_params["radius"],
            planet_mass=self.planet_params["mass"],
        )
        star = BlackbodyStar(
            temperature=self.star_params["temperature"],
            radius=self.star_params["radius"],
        )

        # Create temperature profile
        if self.temperature_params["model"] == "Guillot2010":
            temperature_profile = Guillot2010(
                T_irr=self.temperature_params["T_irr"]
            )
        else:
            temperature_profile = Isothermal(
                T=self.temperature_params.get("T", 1500.0)
            )

        # Set up chemistry
        chemistry = TaurexChemistry()
        for gas_name, abundance in self.chemistry_params["gases"].items():
            try:
                abundance = float(abundance)
                chemistry.addGas(ConstantGas(gas_name, mix_ratio=abundance))
            except ValueError:
                raise ValueError(
                    f"Invalid abundance for {gas_name}: {abundance}. It must be a number."
                )

        # Create the transmission model
        model = TransmissionModel(
            planet=planet,
            temperature_profile=temperature_profile,
            chemistry=chemistry,
            star=star,
            atm_min_pressure=1e-0,
            atm_max_pressure=1e6,
            nlayers=30,
        )

        model.add_contribution(AbsorptionContribution())
        model.add_contribution(RayleighContribution())
        model.add_contribution(CIAContribution(cia_pairs=["H2-H2", "H2-He"]))
        model.build()

        return model

    def run(self):
        # Load and configure the observed spectrum
        obs = ObservedSpectrum(self.spectrum_file)
        obin = obs.create_binner()

        # Create the model
        model = self.create_model()

        # Set up the optimizer
        opt = NestleOptimizer(num_live_points=50)
        opt.set_model(model)
        opt.set_observed(obs)

        # Enable fitting for specified parameters
        for param, bounds in self.fit_params.items():
            opt.enable_fit(param)
            opt.set_boundary(param, bounds)

        # Run optimization
        solution = opt.fit()

        # Save and plot the retrieved spectrum
        for solution, optimized_map, optimized_value, values in opt.get_solution():
            opt.update_model(optimized_map)

            # Save the retrieved spectrum
            wavelength, rprs, rprs_err = obs.wavelengthGrid, obs.spectrum, obs.errorBar
            np.savetxt(
                self.output_file,
                np.column_stack([wavelength, rprs, rprs_err]),
                header="Wavelength(micron) (Rp/Rs)^2 sqrt((Rp/Rs)^2)",
                fmt="%.6e",
            )
            print(f"Retrieved spectrum saved to {self.output_file}")

            # Plot the results
            plt.figure()
            plt.errorbar(wavelength, rprs, yerr=rprs_err, label="Observed")
            plt.plot(
                wavelength,
                obin.bin_model(model.model(obs.wavenumberGrid))[1],
                label="Retrieved",
            )
            plt.legend()
            plt.show()