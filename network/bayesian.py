import torch
import numpy as np
import json
import logging
from dataloader import StressStrainDataset
from args import args
from utils import load_models, get_objective_with_uncertainty

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_parameter_points(n_samples):
    """
    Generates random parameter points using a Dirichlet distribution.
    """
    logging.info(f"Generating {n_samples} candidate parameter points...")
    parameter_points = []
    param_ranges = [(0, 1)] * args['params_size']

    for _ in range(n_samples):
        params = np.random.dirichlet([1] * len(param_ranges))
        scaled_params = [np.round(p * (high - low) + low, 3) for (low, high), p in zip(param_ranges, params)]
        last_param = 1.0 - sum(scaled_params[:-1])
        scaled_params[-1] = np.round(last_param * (param_ranges[-1][1] - param_ranges[-1][0]) + param_ranges[-1][0], 3)
        parameter_points.append(scaled_params)

    logging.info("Candidate point generation complete.")
    return torch.tensor(parameter_points, dtype=torch.float32)


def penalized_ucb(means, stds, sampled_params, candidate_params, kappa=0):
    """
    Calculates the Upper Confidence Bound (UCB) with a penalization term.
    """
    ucb_scores = kappa * stds + means

    candidate_points_expanded = candidate_params.unsqueeze(1)
    sampled_points_expanded = sampled_params.unsqueeze(0)
    distances = torch.norm(candidate_points_expanded - sampled_points_expanded, dim=2)
    d1 = distances.min(dim=1).values

    radii = torch.full_like(d1, args['penalization_radius'])
    within_radius = d1 < radii
    penalization_factors = (radii - d1) / radii
    penalization_factors[~within_radius] = 0

    ucb_scores = ucb_scores * (1 - penalization_factors)
    return ucb_scores.unsqueeze(1)


def select_next_point(ensemble_models, candidate_params, strain_inputs, sampled_params=torch.tensor([]), kappa=0):
    """
    Selects the next best point to sample based on the penalized UCB score.
    """
    means, stds = get_objective_with_uncertainty(ensemble_models, strain_inputs, candidate_params)
    ucb_scores = penalized_ucb(means, stds, sampled_params, candidate_params, kappa)
    selected_index = torch.argmax(ucb_scores).item()

    selected_param = candidate_params[selected_index:selected_index + 1]
    logging.info(f"Selected point with mean objective: {means[selected_index]:.4f}, Params: {selected_param.numpy().tolist()}")

    candidate_params = torch.cat([candidate_params[:selected_index], candidate_params[selected_index + 1:]])
    return selected_param, candidate_params


def select_next_batch(ensemble_models, candidate_params, strain_inputs, sampled_params=torch.tensor([]), batch_size=3, kappa=0):
    """
    Selects a batch of points to sample next.
    """
    selected_params = torch.empty(0, candidate_params.shape[1])
    for i in range(batch_size):
        logging.info(f"--- Selecting point {i+1}/{batch_size} for the batch ---")
        selected_param, candidate_params = select_next_point(ensemble_models, candidate_params, strain_inputs, sampled_params, kappa)
        selected_params = torch.cat((selected_params, selected_param), dim=0)
        sampled_params = torch.cat((sampled_params, selected_param), dim=0)
    return selected_params


def get_seen_params():
    """
    Loads the parameters of the training data.
    """
    logging.info("Loading previously seen parameters from the dataset...")
    seen_dataset = StressStrainDataset(args['data_dir'])
    seen_params = [point[1].detach().numpy() for point in seen_dataset]
    logging.info(f"Loaded {len(seen_params)} seen parameter sets.")
    return torch.tensor(seen_params)


if __name__ == "__main__":
    logging.info("Starting Bayesian Optimization process.")

    models = load_models(args['model_save_path'], args['num_models'])
    logging.info(f"Loaded {len(models)} models for the ensemble.")

    dataset = StressStrainDataset(args['data_dir'])
    strain_values = dataset[0][0]
    logging.info("Loaded strain values for objective calculation.")

    results = []

    candidate_params = sample_parameter_points(args['num_candidate_samples'])
    seen_params = get_seen_params()

    selected_params = select_next_batch(models, candidate_params, strain_values, sampled_params=seen_params, batch_size=args['bayesian_batch_size'], kappa=args['kappa'])

    result_list = selected_params.detach().numpy().tolist()
    results.extend(result_list)

    output_file = 'bayesian_selected_points.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"--- Bayesian Optimization complete. Results saved to {output_file} ---")