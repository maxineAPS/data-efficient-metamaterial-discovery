import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
from dataloader import StressStrainDataset
from model import TwoHeadedMLP
from args import args
import os

def train_single_model(model, train_dataloader, test_dataloader, num_epochs, model_save_path, model_index, device):
    """
    Trains a single model with early stopping based on validation loss.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=model.weight_decay)

    early_stopping_patience = 100
    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for strain_inputs, params_inputs, targets in train_dataloader:
            strain_inputs, params_inputs, targets = strain_inputs.to(device), params_inputs.to(device), targets.to(device)
            
            outputs = model(strain_inputs.float(), params_inputs.float())
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for strain_inputs, params_inputs, targets in test_dataloader:
                strain_inputs, params_inputs, targets = strain_inputs.to(device), params_inputs.to(device), targets.to(device)
                outputs = model(strain_inputs.float(), params_inputs.float())
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_dataloader)
        
        print(f'Model {model_index + 1}/{args["num_models"]}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_path = f'{model_save_path}/best_model_{model_index}.pt'
            torch.save(model.state_dict(), best_model_path)
            print(f'Validation loss improved. Saving best model to {best_model_path}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    final_model_path = f'{model_save_path}/model_{model_index}.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Finished training model {model_index + 1}. Final model saved at {final_model_path}')

def train_ensemble(args):
    """
    Loads all data, splits it into training and testing sets, and trains an ensemble of models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    full_dataset = StressStrainDataset(args['data_dir'])
    
    if len(full_dataset) == 0:
        print("No data found. Exiting.")
        return

    test_size = int(len(full_dataset) * args['test_split_ratio'])
    train_size = len(full_dataset) - test_size
    
    # Ensure the split is valid
    if train_size <= 0 or test_size <= 0:
        print(f"Dataset is too small to create a train/test split with ratio {args['test_split_ratio']}. Exiting.")
        return

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    for i in range(args['num_models']):
        print(f'\n--- Training model {i+1}/{args["num_models"]} ---')

        model = TwoHeadedMLP(
            strain_input_size=args['sim_strain_size'], 
            params_input_size=args['params_size'], 
            output_size=args['sim_stress_size'], 
            hidden_size=args['hidden_size'],
            # The 'num_layers' arg is not used in the hardcoded model architecture
            # num_layers=args['num_layers'], 
            dropout=args['dropout_rate']
        ).to(device)
        
        train_single_model(model, train_dataloader, test_dataloader, args['num_epochs'], args['model_save_path'], i, device)

if __name__ == "__main__":
    start_time = time.time()
    train_ensemble(args)
    elapsed_time = time.time() - start_time
    print(f"\nTotal training time: {elapsed_time:.2f} seconds")