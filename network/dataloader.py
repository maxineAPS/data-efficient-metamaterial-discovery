import json
import torch
from torch.utils.data import Dataset
import os
import sys

class StressStrainDataset(Dataset):
    """
    Custom PyTorch Dataset for loading stress-strain data from JSON files.
    This version handles a nested directory structure and includes validation
    to ensure the data directory exists and contains data.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = []

        # --- Validation Step ---
        # Check if the data directory actually exists.
        if not os.path.isdir(self.data_dir):
            print(f"--- FATAL ERROR ---")
            print(f"The data directory specified in your args.py file does not exist.")
            print(f"Path not found: '{os.path.abspath(self.data_dir)}'")
            print(f"Please make sure the 'data_dir' path in args.py is correct.")
            sys.exit(1)

        # Walk through the directory structure to find all json files
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    self.file_list.append(os.path.join(root, file))
        
        # Check if any files were found
        if not self.file_list:
            print(f"--- WARNING ---")
            print(f"The data directory '{self.data_dir}' exists, but no .json files were found inside.")
            print(f"Please check that your data files are in the correct location.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        try:
            strain_data = data['strain']
            stress_data = data['stress']
            params_data = data['params']
        except KeyError as e:
            print(f"Missing key in {file_path}: {e}")
            return None

        strain_values = torch.tensor(strain_data, dtype=torch.float32) 
        stress_values = torch.tensor(stress_data, dtype=torch.float32)
        params_values = torch.tensor(params_data, dtype=torch.float32)
        
        return strain_values, params_values, stress_values