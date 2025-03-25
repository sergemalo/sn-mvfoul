import torch
import torch.nn as nn

class ChannelReducer(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
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
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create a configurable convolution to reduce channels
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
    
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
        reduced = self.conv(reshaped)
        
        # Reshape back to original dimensions but with reduced channels
        result = reduced.reshape(batch_size, num_views, frames, self.out_channels, height, width)
        
        # Permute to match expected output format (batch_size, views, out_channels, frames, height, width)
        result = result.permute(0, 1, 3, 2, 4, 5)
        
        return result 

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
        out_channels=out_channels
    )
    
    # Advanced usage with custom convolution parameters
    model_custom = ChannelReducer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,       # Using 3x3 convolution instead of 1x1
        padding=1,           # Add padding to maintain spatial dimensions
        stride=1,
        bias=False           # Disable bias
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