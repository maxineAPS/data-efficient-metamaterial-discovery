args = {
    'sim_strain_size': 120,
    'params_size': 8,
    'sim_stress_size': 120,
    'hidden_size': 256,
    'num_layers': 5,
    'dropout_rate': 0.3,
    'batch_size': 30,
    'num_epochs': 2000,
    'learning_rate': 0.001,
    'num_models': 30,
    'test_split_ratio': 0.2, #Note that in the paper, all available data is used for training each batch.
    'data_dir': 'data',
    'model_save_path': 'models',
    'bayesian_batch_size': 40,
    'num_candidate_samples': 1000000,
    'penalization_radius': 0.2,
    'kappa': 1.0 
}
