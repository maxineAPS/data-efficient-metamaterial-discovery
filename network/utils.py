import torch
from model import TwoHeadedMLP
from args import args
import os

def load_models(model_path, num_models):
    """
    Loads an ensemble of trained models.
    Checks if the model files exist before attempting to load.
    """
    models = []
    print(f"Loading {num_models} models from {model_path}...")
    for i in range(num_models):
        # Using the final model, not the 'best_model' for consistency
        m_path = os.path.join(model_path, f'model_{i}.pt')
        
        if not os.path.exists(m_path):
            print(f"--- ERROR: Model file not found at {m_path} ---")
            print("Please run train.py to train the ensemble first.")
            return [] # Return empty list if any model is missing

        model = TwoHeadedMLP(
            strain_input_size=args['sim_strain_size'], 
            params_input_size=args['params_size'], 
            output_size=args['sim_stress_size'], 
            hidden_size=args['hidden_size'],
            dropout=args['dropout_rate']
        )
        
        # Load the state dict, ensuring it's mapped to the correct device (e.g., CPU)
        # if the models were trained on a different device (e.g., GPU).
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(m_path, map_location=device))
        model.to(device) # Move model to the appropriate device
        model.eval()
        models.append(model)
        
    print(f"Successfully loaded {len(models)} models.")
    return models


def predict_with_uncertainty(models, strain_inputs, params_inputs):
    """
    Makes predictions with an ensemble of models to quantify uncertainty.
    """
    device = next(models[0].parameters()).device
    strain_inputs = strain_inputs.to(device)
    params_inputs = params_inputs.to(device)

    with torch.no_grad():
        # Initialize a list to store outputs from each model
        all_outputs_list = []

        for model in models:
            output = model(strain_inputs.float(), params_inputs.float())
            all_outputs_list.append(output.unsqueeze(0))

        # Concatenate the list of tensors into a single tensor
        all_outputs = torch.cat(all_outputs_list, dim=0)

        mean_outputs = all_outputs.mean(dim=0)
        std_outputs = all_outputs.std(dim=0)
        
    return mean_outputs, std_outputs

def get_objective_with_uncertainty(models, strain_inputs, params_inputs):
    """
    Calculates a scalar objective (area under the stress-strain curve) with uncertainty.
    The stress-strain curve is split into loading (first 60 points) and unloading.
    """
    device = next(models[0].parameters()).device
    strain_inputs = strain_inputs.to(device)
    params_inputs = params_inputs.to(device)
    
    with torch.no_grad():
        # If a single strain curve is provided for multiple parameter sets, repeat it.
        if strain_inputs.shape[0] != params_inputs.shape[0]:
            strain_inputs = strain_inputs.repeat(params_inputs.shape[0], 1)

        all_areas = torch.zeros(len(models), params_inputs.shape[0], device=device)

        for i, model in enumerate(models):
            stresses = model(strain_inputs.float(), params_inputs.float())
            
            # The first half of the curve is loading
            loading = stresses[:, :args['sim_stress_size'] // 2]
            # The second half is unloading, which we flip for correct integration
            unloading = torch.flip(stresses[:, args['sim_stress_size'] // 2:], dims=[1])
            
            # The objective is the area between the loading and unloading curves (hysteresis loop)
            # We assume the strain values for integration are uniform.
            # The integral of (loading - unloading) d(strain)
            area = torch.trapz(loading - unloading, dx=(max(strain_inputs)/60), dim=1)
            all_areas[i] = area

        mean_area = all_areas.mean(dim=0)
        std_area = all_areas.std(dim=0)
        
    return mean_area, std_area