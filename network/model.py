import torch
import torch.nn as nn

class TwoHeadedMLP(nn.Module):
    """
    A two-headed MLP model for predicting stress values from strain and material parameters.
    The architecture is based on Figure 7 from the paper.
    
    Args:
        strain_input_size (int): Size of the strain input vector.
        params_input_size (int): Size of the material parameters input vector.
        output_size (int): Size of the output vector (predicted stress).
        hidden_size (int): Size of the hidden layers.
        dropout (float): Dropout rate.
        weight_decay (float, optional): Weight decay (L2 regularization) factor. Default is 1e-4.
    """
    def __init__(self, strain_input_size, params_input_size, output_size, hidden_size=256, dropout=0.3, weight_decay=1e-4):
        super(TwoHeadedMLP, self).__init__()
        
        # Note: The 'num_layers' argument from args.py is not used here,
        # as the architecture is hardcoded to match the paper's diagram for reproducibility.

        # Strain head: processes the strain values
        self.strain_head = nn.Sequential(
            nn.Linear(strain_input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        
        # Params head: processes the 8 TPMS parameters
        self.params_head = nn.Sequential(
            nn.Linear(params_input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout)
        )
        
        # Combined layers: merge the outputs of the two heads and produce the final prediction
        # The combined input size is hidden_size * 2
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2), # Corresponds to 512 in paper
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, output_size)
        )
        
        self.weight_decay = weight_decay

    def forward(self, strain_input, params_input):
        """
        Forward pass of the model.
        
        Args:
            strain_input (Tensor): Input tensor for strain values.
            params_input (Tensor): Input tensor for material parameters.
        
        Returns:
            Tensor: Predicted stress values.
        """
        strain_output = self.strain_head(strain_input)
        params_output = self.params_head(params_input)
        
        # Concatenate the outputs from both heads along the feature dimension
        combined_output = torch.cat((strain_output, params_output), dim=1)
        
        output = self.combined_layers(combined_output)
        return output
