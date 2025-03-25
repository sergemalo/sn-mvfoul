import torch
import torch.nn as nn

class ChannelReducer(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int):
        """
        Channel reducer for multiview video format
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create a 1x1 convolution to reduce channels
        # This is much more efficient than an MLP for this task
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        print(f"--> Created channel reducer from {in_channels} to {out_channels} channels")
    
    def forward(self, x):
        """
        Forward pass for multiview video format
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, views, channels, frames, height, width)
        
        Returns:
            torch.Tensor: Output tensor with reduced channels (batch_size, views, out_channels, frames, height, width)
        """
        original_shape = x.shape
        print(f"Channel reducer input shape: {original_shape}")
        
        # Check if we have the expected 6D tensor
        if len(original_shape) != 6:
            raise ValueError(f"Expected 6D tensor with shape (batch, views, channels, frames, height, width), but got shape {original_shape}")
        
        batch_size, num_views, channels, frames, height, width = original_shape
        
        if channels != self.in_channels:
            print(f"WARNING: Expected {self.in_channels} channels but got {channels}")
            if channels < self.in_channels:
                print(f"ERROR: Input has fewer channels ({channels}) than expected ({self.in_channels})")
                raise ValueError(f"Input has fewer channels ({channels}) than expected ({self.in_channels})")
        
        # Reshape to process all views and frames at once using batched operations
        # Reshape to (batch_size * num_views * frames, channels, height, width)
        reshaped = x.reshape(batch_size * num_views * frames, channels, height, width)
        
        # Apply channel reduction
        reduced = self.conv(reshaped)
        
        # Reshape back to original dimensions but with reduced channels
        result = reduced.reshape(batch_size, num_views, frames, self.out_channels, height, width)
        
        # Permute to match expected output format (batch_size, views, out_channels, frames, height, width)
        result = result.permute(0, 1, 3, 2, 4, 5)
        
        print(f"Channel reducer output shape: {result.shape}")
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
    
    # Create model
    model = ChannelReducer(
        in_channels=in_channels,
        out_channels=out_channels
    )
    
    # Example input - Multiview video format (batch, views, channels, frames, height, width)
    x = torch.randn(batch_size, views, in_channels, frames, height, width)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 