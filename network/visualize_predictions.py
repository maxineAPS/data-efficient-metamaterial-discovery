import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from dataloader import StressStrainDataset
from args import args
from utils import load_models, predict_with_uncertainty

def visualize_predictions_with_uncertainty(models, dataloader, num_plots=1):
    """
    Visualizes model predictions with uncertainty against ground truth.
    """
    criterion = torch.nn.MSELoss()
    
    for i, (strain_inputs, params_inputs, targets) in enumerate(dataloader):
        if i >= num_plots:
            break

        mean_outputs, std_outputs = predict_with_uncertainty(models, strain_inputs, params_inputs)
        
        loss = criterion(mean_outputs, targets)
        print(f"Test Batch {i+1} Loss: {loss.item():.4f}")

        strain_inputs_np = strain_inputs.numpy()
        targets_np = targets.numpy()
        mean_outputs_np = mean_outputs.detach().numpy()
        std_outputs_np = std_outputs.detach().numpy()

        num_subplots = min(strain_inputs.shape[0], 4) # Show up to 4 plots per batch
        num_rows = (num_subplots + 1) // 2
        plt.figure(figsize=(12, 4 * num_rows))

        for j in range(num_subplots):
            ax = plt.subplot(num_rows, 2, j + 1)
            ax.plot(strain_inputs_np[j], targets_np[j], label='Ground Truth', color='blue')
            ax.plot(strain_inputs_np[j], mean_outputs_np[j], label='Mean Prediction', color='red', linestyle='--')
            
            ax.fill_between(
                strain_inputs_np[j], 
                mean_outputs_np[j] - 2 * std_outputs_np[j], 
                mean_outputs_np[j] + 2 * std_outputs_np[j], 
                color='red',
                alpha=0.2, 
                label='Uncertainty (2 std. dev.)'
            )

            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress')
            
            params_str = ', '.join([f'{param:.2f}' for param in params_inputs[j]])
            uncertainty_score = np.trapz(std_outputs_np[j], x=strain_inputs_np[j])
            ax.set_title(f'Params: [{params_str}]\nUncertainty Score: {uncertainty_score:.4f}', fontsize=8)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(h_pad=3.0)
        plt.show()

if __name__ == "__main__":
    models = load_models(args['model_save_path'], args['num_models'])

    # Load the full dataset and create a test split to visualize
    print("Loading data for visualization from:", args['data_dir'])
    full_dataset = StressStrainDataset(args['data_dir'])
    test_size = int(len(full_dataset) * args['test_split_ratio'])
    train_size = len(full_dataset) - test_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)

    visualize_predictions_with_uncertainty(models, dataloader, num_plots=1)