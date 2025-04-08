import torch
import torch.nn as nn

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
                 data_range: tuple = (0, 255)):
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
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
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
        reduced = reduced * (self.data_range[1] - self.data_range[0]) + self.data_range[0]
        reduced = reduced.int()
        
        # Reshape back to original dimensions but with reduced channels
        result = reduced.reshape(batch_size, num_views, frames, self.out_channels, height, width)
        
        # Permute to match expected output format (batch_size, views, out_channels, frames, height, width)
        result = result.permute(0, 1, 3, 2, 4, 5)
        
        return result 

    def get_channel_importance(self):
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
            relative_importance = torch.ones(self.in_channels) / self.in_channels
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
            relative_importance = torch.ones_like(absolute_importance) / len(absolute_importance)
        
        # Calculate importance per output channel
        # Sum across spatial dimensions only
        per_output_channel = torch.sum(torch.abs(grad), dim=(2, 3))
        
        return {
            'absolute_importance': absolute_importance,
            'relative_importance': relative_importance,
            'per_output_channel': per_output_channel,
            'no_grad_available': False
        }

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
    
    # Example input - Multiview video format (batch, views, channels, frames, height, width)
    x = torch.randn(batch_size, views, in_channels, frames, height, width)
    
    # Forward pass with default model
    output_default = model_default(x)
    
    # Forward pass with custom model
    output_custom = model_custom(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (default model): {output_default.shape}")
    print(f"Output shape (custom model): {output_custom.shape}")
    print(f"Output range (default model): {output_default.min().item()}, {output_default.max().item()}")
    print(f"Output range (custom model): {output_custom.min().item()}, {output_custom.max().item()}")
    
    # Analyze channel importance
    importance = model_default.get_channel_importance()
    print("\nChannel Importance Analysis:")
    print(f"Absolute importance: {importance['absolute_importance']}")
    print(f"Relative importance: {importance['relative_importance']}")
    print(f"Per output channel importance:\n{importance['per_output_channel']}") 