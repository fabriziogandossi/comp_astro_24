import os
import yaml
import numpy as np

class Parameters:
    def __init__(self, input_file):
        self.params = {}  # Initialize params to an empty dictionary by default

        # Load parameters from the YAML file
        if os.path.exists(input_file) and os.path.isfile(input_file):
            with open(input_file) as in_f:
                self.params = yaml.load(in_f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError(f"The file {input_file} does not exist or is not a valid file.")
        
        # Replace "None" string with actual None value
        for par in list(self.params.keys()):
            if self.params[par] == "None":
                self.params[par] = None

    def get(self, param):
        # Safely return the value for the given key, or None if it doesn't exist
        return self.params.get(param, None)

