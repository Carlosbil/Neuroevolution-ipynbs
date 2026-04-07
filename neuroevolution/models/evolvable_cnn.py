"""
Evolvable CNN architecture for 1D audio processing.
"""

import torch
import torch.nn as nn
import random
from neuroevolution.config import ACTIVATION_FUNCTIONS


class EvolvableCNN(nn.Module):
    """
    Evolvable CNN class for 1D audio processing.
    Uses Conv1D layers for audio/sequential data.
    """
    
    def __init__(self, genome: dict, config: dict):
        super(EvolvableCNN, self).__init__()
        self.genome = genome
        self.config = config
        
        # Validate and fix genome structure before building
        self._validate_genome()
        
        # Build convolutional layers (1D for audio)
        self.conv_layers = self._build_conv_layers()
        
        # Calculate output size after convolutions
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Build fully connected layers
        self.fc_layers = self._build_fc_layers()
    
    def _validate_genome(self):
        """Validates and fixes genome structure to ensure consistency."""
        # Ensure conv-related lists match num_conv_layers
        num_conv = self.genome['num_conv_layers']
        
        if len(self.genome['filters']) != num_conv:
            # Fix filters list
            self.genome['filters'] = self.genome['filters'][:num_conv]
            while len(self.genome['filters']) < num_conv:
                self.genome['filters'].append(
                    random.randint(self.config['min_filters'], self.config['max_filters'])
                )
        
        if len(self.genome['kernel_sizes']) != num_conv:
            # Fix kernel_sizes list
            self.genome['kernel_sizes'] = self.genome['kernel_sizes'][:num_conv]
            while len(self.genome['kernel_sizes']) < num_conv:
                self.genome['kernel_sizes'].append(
                    random.choice(self.config['kernel_size_options'])
                )
        
        # Ensure fc-related lists match num_fc_layers
        num_fc = self.genome['num_fc_layers']
        
        if len(self.genome['fc_nodes']) != num_fc:
            # Fix fc_nodes list
            self.genome['fc_nodes'] = self.genome['fc_nodes'][:num_fc]
            while len(self.genome['fc_nodes']) < num_fc:
                self.genome['fc_nodes'].append(
                    random.randint(self.config['min_fc_nodes'], self.config['max_fc_nodes'])
                )
        
    def _build_conv_layers(self) -> nn.ModuleList:
        """Builds 1D convolutional layers according to genome."""
        layers = nn.ModuleList()
        
        in_channels = self.config['num_channels']
        normalization_type = self.genome.get('normalization_type', 'batch')

        for i in range(self.genome['num_conv_layers']):
            # Safe access with validation
            if i >= len(self.genome['filters']) or i >= len(self.genome['kernel_sizes']):
                raise IndexError(
                    f"Genome list mismatch: i={i}, num_conv_layers={self.genome['num_conv_layers']}, "
                    f"filters_len={len(self.genome['filters'])}, kernel_sizes_len={len(self.genome['kernel_sizes'])}"
                )
            
            out_channels = self.genome['filters'][i]
            kernel_size = self.genome['kernel_sizes'][i]
            
            # Ensure kernel size is odd and reasonable for 1D
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            padding = kernel_size // 2
            
            # 1D Convolutional layer
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            layers.append(conv)
            
            # Normalization layer (Layer Normalization or Batch Normalization)
            if normalization_type == 'layer':
                layers.append(nn.LayerNorm(out_channels))
            else:
                # Batch normalization (default)
                layers.append(nn.BatchNorm1d(out_channels))
            
            # Activation function
            activation_name = self.genome['activations'][i % len(self.genome['activations'])]
            activation_func = ACTIVATION_FUNCTIONS[activation_name]()
            layers.append(activation_func)
            
            # Max pooling (1D) - reduce sequence length
            pool_size = 2 if i < self.genome['num_conv_layers'] - 1 else 2
            layers.append(nn.MaxPool1d(pool_size, pool_size))
            
            # Optional dropout after pooling
            if i < self.genome['num_conv_layers'] - 1:
                layers.append(nn.Dropout(0.1))
            
            in_channels = out_channels
            
        return layers
    
    def _calculate_conv_output_size(self) -> int:
        """
        Calculates output size after convolutional layers.
        Raises ValueError if the architecture produces invalid dimensions.
        """
        # Create dummy tensor to calculate size
        dummy_input = torch.zeros(1, self.config['num_channels'], 
                                 self.config['sequence_length'])
        
        # Pass through convolutional layers with validation
        x = dummy_input
        normalization_type = self.genome.get('normalization_type', 'batch')
        
        try:
            # Set model to eval mode to avoid BatchNorm training issues with batch_size=1
            self.eval()
            
            for layer in self.conv_layers:
                # Check dimensions before BatchNorm layers
                if isinstance(layer, nn.BatchNorm1d) and normalization_type == 'batch':
                    # Check if spatial dimension is too small for BatchNorm
                    if x.shape[2] <= 1:  # spatial dimension
                        raise ValueError(
                            f"Invalid architecture: spatial dimension too small ({x.shape[2]}) "
                            f"for BatchNorm1d. This genome produces architectures that are too deep. "
                            f"Genome: num_conv_layers={self.genome['num_conv_layers']}, "
                            f"sequence_length={self.config['sequence_length']}"
                        )
                
                x = layer(x)
                
                # Additional check after each layer
                if x.shape[2] < 1:
                    raise ValueError(
                        f"Invalid architecture: sequence length became zero or negative. "
                        f"Current shape: {x.shape}, "
                        f"Genome: num_conv_layers={self.genome['num_conv_layers']}"
                    )
            
            # Back to training mode
            self.train()
            
        except ValueError as e:
            # Re-raise our custom validation errors
            raise e
        except Exception as e:
            # Catch any other errors during size calculation
            raise ValueError(
                f"Error calculating conv output size: {str(e)}. "
                f"Genome may produce invalid architecture. "
                f"num_conv_layers={self.genome['num_conv_layers']}, "
                f"sequence_length={self.config['sequence_length']}"
            )
        
        # Flatten and get size
        flattened_size = x.view(-1).shape[0]
        
        # Ensure we have a reasonable output size
        if flattened_size < 1:
            raise ValueError(
                f"Invalid architecture: flattened size is {flattened_size}. "
                f"The architecture is too aggressive in dimension reduction."
            )
        
        return flattened_size
    
    def _build_fc_layers(self) -> nn.ModuleList:
        """Builds fully connected layers."""
        layers = nn.ModuleList()
        
        input_size = self.conv_output_size
        normalization_type = self.genome.get('normalization_type', 'batch')

        for i in range(self.genome['num_fc_layers']):
            # Safe access with validation
            if i >= len(self.genome['fc_nodes']):
                raise IndexError(
                    f"Genome list mismatch: i={i}, num_fc_layers={self.genome['num_fc_layers']}, "
                    f"fc_nodes_len={len(self.genome['fc_nodes'])}"
                )
            
            output_size = self.genome['fc_nodes'][i]
            
            # Linear layer
            layers.append(nn.Linear(input_size, output_size))
            
            # Normalization layer (Layer Normalization or Batch Normalization)
            if normalization_type == 'layer':
                layers.append(nn.LayerNorm(output_size))
            else:
                # Batch normalization for FC layers (default)
                layers.append(nn.BatchNorm1d(output_size))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout if not last layer
            if i < self.genome['num_fc_layers'] - 1:
                layers.append(nn.Dropout(self.genome['dropout_rate']))
            
            input_size = output_size
        
        # Final classification layer
        layers.append(nn.Linear(input_size, self.config['num_classes']))
        
        return layers
    
    def forward(self, x):
        """Forward pass of the network."""
        # Ensure input is in correct format for Conv1d
        # Expected: (batch, channels, sequence_length)
        if len(x.shape) == 2:  # (batch, sequence)
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        return x
    
    def get_architecture_summary(self) -> str:
        """Returns an architecture summary."""
        summary = []
        summary.append(f"Conv1D Layers: {self.genome['num_conv_layers']}")
        summary.append(f"Filters: {self.genome['filters']}")
        summary.append(f"Kernel Sizes: {self.genome['kernel_sizes']}")
        summary.append(f"FC Layers: {self.genome['num_fc_layers']}")
        summary.append(f"FC Nodes: {self.genome['fc_nodes']}")
        summary.append(f"Activations: {self.genome['activations']}")
        summary.append(f"Normalization: {self.genome.get('normalization_type', 'batch')}")
        summary.append(f"Dropout: {self.genome['dropout_rate']:.3f}")
        summary.append(f"Optimizer: {self.genome['optimizer']}")
        summary.append(f"Learning Rate: {self.genome['learning_rate']:.4f}")
        return " | ".join(summary)
