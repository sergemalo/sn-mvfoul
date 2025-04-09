import torch
import torch.nn as nn
import os

class ChannelReducer(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 activation: str = 'leakyrelu',
                 data_range: tuple = (0, 255),
                 initial_channels: int = None):
        """
        Channel reducer for multiview video format
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel. Default: 1
            stride (int): Stride of the convolution. Default: 1
            padding (int): Zero-padding added to both sides of the input. Default: 0
            dilation (int): Spacing between kernel elements. Default: 1
            groups (int): Number of blocked connections from input to output channels. Default: 1
            bias (bool): If True, adds a learnable bias to the output. Default: True
            padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
            activation (str): Type of activation function to use. Options: 'relu', 'leakyrelu', 'sigmoid', 'tanh'. Default: 'leakyrelu'
            data_range (tuple): Range of the data. Default: (0, 255)
            initial_channels (int): Number of input channels to use for initialization. If None, all channels are used. Default: None
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_channels = initial_channels if initial_channels is not None else in_channels
        
        # Create a configurable convolution to reduce channels
        self.conv1 = nn.Conv2d(
            in_channels, 
            hidden_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

        self.conv2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            print(f"WARNING: Unknown activation '{activation}'. Using no activation.")

        self.output_activation = nn.Sigmoid()
        self.data_range = data_range
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initialize all weights in the model using Xavier initialization.
        This will override any previous initialization.
        """
        with torch.no_grad():
            # Initialize conv1 weights
            nn.init.xavier_normal_(self.conv1.weight)
            if self.conv1.bias is not None:
                nn.init.zeros_(self.conv1.bias)
            
            # Initialize conv2 weights
            nn.init.xavier_normal_(self.conv2.weight)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
            
            # Set weights for inactive channels to zero.
            if self.initial_channels < self.in_channels:
                self.conv1.weight[:, self.initial_channels:, :, :] = 0.0

    def forward(self, x):
        """
        Forward pass for multiview video format
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, views, channels, frames, height, width)
        
        Returns:
            torch.Tensor: Output tensor with reduced channels (batch_size, views, out_channels, frames, height, width)
        """
        original_shape = x.shape

        # ------------------------------------------------------------
        # Check if we have the expected 6D tensor
        if len(original_shape) != 6:
            raise ValueError(f"Expected 6D tensor with shape (batch, views, channels, frames, height, width), but got shape {original_shape}")
        
        batch_size, num_views, channels, frames, height, width = original_shape
        
        if channels != self.in_channels:
            print(f"WARNING: Expected {self.in_channels} channels but got {channels}")
            if channels < self.in_channels:
                print(f"ERROR: Input has fewer channels ({channels}) than expected ({self.in_channels})")
                raise ValueError(f"Input has fewer channels ({channels}) than expected ({self.in_channels})")
        # ------------------------------------------------------------
        
        # Reshape to process all views and frames at once using batched operations
        # Reshape to (batch_size * num_views * frames, channels, height, width)
        reshaped = x.reshape(batch_size * num_views * frames, channels, height, width)
        
        # Apply channel reduction
        reduced = self.conv1(reshaped)
        reduced = self.activation(reduced)
        reduced = self.conv2(reduced)
        reduced = self.output_activation(reduced)
        # Values are between 0 and 1, we need to scale it to 0 and 255
        scale = self.data_range[1] - self.data_range[0]
        offset = self.data_range[0]
        reduced = reduced.mul(scale).add(offset)
        
        # Reshape back to original dimensions but with reduced channels
        result = reduced.reshape(batch_size, num_views, frames, self.out_channels, height, width)
        
        # Permute to match expected output format (batch_size, views, out_channels, frames, height, width)
        result = result.permute(0, 1, 3, 2, 4, 5)
        
        return result 
    
    def get_channel_importance(self):
        """
        Analyze the importance of each input channel based on weight magnitude.
        
        Returns:
            dict: Dictionary containing:
                - 'absolute_importance': Tensor of shape (in_channels,) showing absolute importance
                - 'relative_importance': Tensor of shape (in_channels,) showing relative importance (sums to 1)
                - 'per_output_channel': Tensor of shape (out_channels, in_channels) showing importance per output channel
        """
        return self._get_weights_magnitude()

    def _get_weights_magnitude(self):
        """
        Analyze the importance of each input channel based on weight magnitude.
        
        Returns:
            dict: Dictionary containing:
                - 'absolute_importance': Tensor of shape (in_channels,) showing absolute importance
                - 'relative_importance': Tensor of shape (in_channels,) showing relative importance (sums to 1)
                - 'per_output_channel': Tensor of shape (out_channels, in_channels) showing importance per output channel
        """
        # Get the weights of the convolution
        weights = self.conv1.weight.clone().detach()
        
        # Calculate absolute importance for each input channel based on weight magnitude
        # Sum across output channels and spatial dimensions
        absolute_importance = torch.sum(torch.abs(weights), dim=(0, 2, 3))
        
        # Calculate relative importance (normalized to sum to 1)
        # Handle case where all weights might be zero
        sum_importance = torch.sum(absolute_importance)
        if sum_importance > 0:
            relative_importance = absolute_importance / sum_importance
        else:
            relative_importance = torch.zeros_like(absolute_importance)
        
        # Calculate importance per output channel
        # Sum across spatial dimensions only
        per_output_channel = torch.sum(torch.abs(weights), dim=(2, 3))
        
        return {
            'absolute_importance': absolute_importance,
            'relative_importance': relative_importance,
            'per_output_channel': per_output_channel
        }

    def _get_gradients_magnitude(self):
        """
        Analyze the importance of each input channel in the output based on gradient magnitude.
        This method should be called after a backward pass has occurred.
        
        Returns:
            dict: Dictionary containing:
                - 'absolute_importance': Tensor of shape (in_channels,) showing absolute importance
                - 'relative_importance': Tensor of shape (in_channels,) showing relative importance (sums to 1)
                - 'per_output_channel': Tensor of shape (out_channels, in_channels) showing importance per output channel
                - 'no_grad_available': Boolean indicating if gradients were available
        """
        # Check if gradients are available
        if self.conv1.weight.grad is None:
            print("WARNING: No gradients available. Run backward pass before calling this method.")
            # Return zeros with appropriate shapes
            absolute_importance = torch.zeros(self.in_channels)
            relative_importance = torch.zeros(self.in_channels)
            per_output_channel = torch.zeros(self.out_channels, self.in_channels)
            return {
                'absolute_importance': absolute_importance,
                'relative_importance': relative_importance,
                'per_output_channel': per_output_channel,
                'no_grad_available': True
            }
        
        # Get the gradient of the convolution weights
        grad = self.conv1.weight.grad.clone().detach()
        
        # Calculate absolute importance for each input channel based on gradient magnitude
        # Sum across output channels and spatial dimensions
        absolute_importance = torch.sum(torch.abs(grad), dim=(0, 2, 3))
        
        # Calculate relative importance (normalized to sum to 1)
        # Handle case where all gradients might be zero
        sum_importance = torch.sum(absolute_importance)
        if sum_importance > 0:
            relative_importance = absolute_importance / sum_importance
        else:
            relative_importance = torch.zeros_like(absolute_importance)
        
        # Calculate importance per output channel
        # Sum across spatial dimensions only
        per_output_channel = torch.sum(torch.abs(grad), dim=(2, 3))
        
        return {
            'absolute_importance': absolute_importance,
            'relative_importance': relative_importance,
            'per_output_channel': per_output_channel,
            'no_grad_available': False
        }

    def save_model(self, path, include_metadata=True):
        """
        Save the model to the specified path.
        
        Args:
            path (str): Path where the model will be saved
            include_metadata (bool): Whether to include metadata about the model configuration
        
        Returns:
            str: Path where the model was saved
        """
        # Create a dictionary with the model state
        save_dict = {
            'state_dict': self.state_dict(),
        }
        
        # Add metadata if requested
        if include_metadata:
            metadata = {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'initial_channels': self.initial_channels,
                'data_range': self.data_range,
                'channel_importance': self.get_channel_importance(),
            }
            save_dict['metadata'] = metadata
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
        
        return path
    
    @classmethod
    def load_model(cls, path, map_location=None):
        """
        Load a model from the specified path.
        
        Args:
            path (str): Path to the saved model
            map_location: Optional torch.device to map model to a specific device
        
        Returns:
            ChannelReducer: Loaded model
        """
        # Load the saved dictionary
        saved_dict = torch.load(path, map_location=map_location)
        
        # Check if we have metadata
        if 'metadata' in saved_dict:
            metadata = saved_dict['metadata']
            # Create a new model with the saved configuration
            model = cls(
                in_channels=metadata['in_channels'],
                out_channels=metadata['out_channels'],
                hidden_channels=32,  # Default value
                initial_channels=metadata.get('initial_channels', None),
                data_range=metadata.get('data_range', (0, 255))
            )
            print(f"Loaded model with {metadata['in_channels']} input channels and {metadata['out_channels']} output channels")
        else:
            # If no metadata, we need the configuration to create the model
            print("WARNING: No metadata found in saved model. Using default configuration.")
            model = cls(
                in_channels=4,  # Default value
                out_channels=3,  # Default value
                hidden_channels=32  # Default value
            )
        
        # Load state dictionary
        model.load_state_dict(saved_dict['state_dict'])
        
        return model

# Example usage
if __name__ == "__main__":
    # Example parameters
    in_channels = 4
    out_channels = 3
    batch_size = 2
    views = 2
    frames = 17
    height = 224
    width = 224
    
    # Basic usage with default parameters
    model_default = ChannelReducer(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32
    )
    
    # Advanced usage with custom convolution parameters and activation
    model_custom = ChannelReducer(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        kernel_size=3,       # Using 3x3 convolution instead of 1x1
        padding=1,           # Add padding to maintain spatial dimensions
        stride=1,
        bias=False,          # Disable bias
        activation='relu'    # Add ReLU activation
    )
    
    # Example with active input channels - only using the first 3 channels of a 4-channel input
    model_active_channels = ChannelReducer(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        initial_channels=3  # Only initialize weights for first 3 channels
    )
    
    # Example input - Multiview video format (batch, views, channels, frames, height, width)
    x = torch.randn(batch_size, views, in_channels, frames, height, width)
    
    # Forward pass with default model
    output_default = model_default(x)
    
    # Forward pass with custom model
    output_custom = model_custom(x)
    
    # Forward pass with active channels model
    output_active = model_active_channels(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (default model): {output_default.shape}")
    print(f"Output shape (custom model): {output_custom.shape}")
    print(f"Output shape (active channels model): {output_active.shape}")
    print(f"Output range (default model): {output_default.min().item()}, {output_default.max().item()}")
    print(f"Output range (custom model): {output_custom.min().item()}, {output_custom.max().item()}")
    print(f"Output range (active channels model): {output_active.min().item()}, {output_active.max().item()}")
    
    # Verify that inactive channel weights are zero in the active channels model
    with torch.no_grad():
        inactive_weights = model_active_channels.conv1.weight[:, 3, :, :]
        active_weights = model_active_channels.conv1.weight[:, :3, :, :]
        print(f"\nInactive channel weights sum: {inactive_weights.abs().sum().item()}")
        print(f"Active channel weights sum: {active_weights.abs().sum().item()}")
        print(f"Active channel weights: {model_active_channels._get_weights_magnitude()['relative_importance']}")
    
    # Analyze channel importance
    importance = model_default.get_channel_importance()
    print("\nChannel Importance Analysis:")
    print(f"Absolute importance: {importance['absolute_importance']}")
    print(f"Relative importance: {importance['relative_importance']}")
    print(f"Per output channel importance:\n{importance['per_output_channel']}")
    
    # Example of saving and loading the model
    save_path = os.path.join("models", "channel_reducer.pt")
    model_default.save_model(save_path)
    
    # Load the model
    loaded_model = ChannelReducer.load_model(save_path)
    
    # Verify the loaded model works
    output_loaded = loaded_model(x)
    print(f"\nLoaded model output shape: {output_loaded.shape}")
    print(f"Loaded model output range: {output_loaded.min().item()}, {output_loaded.max().item()}") 