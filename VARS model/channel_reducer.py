import torch
import torch.nn as nn

class ChannelReducerMLP(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 height: int = None,
                 width: int = None,
                 hidden_layers: list[int] = [128, 64],
                 process_full_video: bool = False):
        """
        MLP model to reduce number of channels in video/images
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            height (int, optional): Height of input images/video. If None, will be inferred at first forward pass.
            width (int, optional): Width of input images/video. If None, will be inferred at first forward pass.
            hidden_layers (list[int]): List of hidden layer sizes
            process_full_video (bool): If True, process entire video at once. If False, process frame by frame
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.process_full_video = process_full_video
        self.hidden_layers = hidden_layers
        
        # MLP will be built on first forward pass if height and width are None
        self.mlp = None
        self.initialized = False
        
        if height is not None and width is not None:
            self._build_mlp(height * width * in_channels)
            self.initialized = True
        
    def _build_mlp(self, input_size):
        """Build the MLP with the specified input size"""
        print(f"Building MLP with input size: {input_size}")
        
        # Build MLP layers
        layers = []
        
        # Add hidden layers
        current_size = input_size
        for hidden_size in self.hidden_layers:
            print(f"Adding layer: {current_size} -> {hidden_size}")
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        
        # Add output layer
        output_size = self.height * self.width * self.out_channels
        print(f"Adding output layer: {current_size} -> {output_size}")
        layers.append(nn.Linear(current_size, output_size))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, frames, channels, height, width)
                            or (batch_size, channels, height, width) for single frame
        
        Returns:
            torch.Tensor: Output tensor with reduced channels
        """
        original_shape = x.shape
        batch_size = original_shape[0]
        
        # Initialize dimensions if not done yet
        if not self.initialized:
            if len(original_shape) == 5:  # Video input
                self.height = original_shape[3]
                self.width = original_shape[4]
            else:  # Image input
                self.height = original_shape[2]
                self.width = original_shape[3]
                
            print(f"Initialized dimensions: height={self.height}, width={self.width}")
            self._build_mlp(self.height * self.width * self.in_channels)
            self.initialized = True
        
        print(f"Input shape: {original_shape}")
        
        if len(original_shape) == 5:  # Video input
            if self.process_full_video:
                print("Processing entire video at once")
                # Process entire video at once
                # Reshape to (batch_size, frames*channels*height*width)
                x_flat = x.reshape(batch_size, -1)
                print(f"Flattened shape: {x_flat.shape}")
                frames = original_shape[1]
                
                # Check if the matrix sizes match
                expected_size = frames * self.in_channels * self.height * self.width
                actual_size = x_flat.shape[1]
                if expected_size != actual_size:
                    print(f"WARNING: Expected flattened size {expected_size} but got {actual_size}")
                    
                x_out = self.mlp(x_flat)
                print(f"MLP output shape: {x_out.shape}")
                
                # Reshape back to (batch_size, frames, out_channels, height, width)
                return x_out.reshape(batch_size, frames, self.out_channels, self.height, self.width)
            else:
                print("Processing frame by frame")
                # Process frame by frame
                frames = original_shape[1]
                output_frames = []
                
                for frame_idx in range(frames):
                    frame = x[:, frame_idx]  # (batch_size, channels, height, width)
                    print(f"Frame {frame_idx} shape: {frame.shape}")
                    
                    # Check if channels match what we expect
                    if frame.shape[1] != self.in_channels:
                        print(f"WARNING: Expected {self.in_channels} channels but got {frame.shape[1]}")
                    
                    # Reshape to (batch_size, channels*height*width)
                    frame_flat = frame.reshape(batch_size, -1)
                    print(f"Flattened frame shape: {frame_flat.shape}")
                    
                    # Check if the matrix sizes match
                    expected_size = self.in_channels * self.height * self.width
                    actual_size = frame_flat.shape[1]
                    if expected_size != actual_size:
                        print(f"WARNING: Expected flattened frame size {expected_size} but got {actual_size}")
                    
                    # Process through MLP
                    processed_frame = self.mlp(frame_flat)
                    print(f"Processed frame shape: {processed_frame.shape}")
                    
                    # Reshape to (batch_size, out_channels, height, width)
                    processed_frame = processed_frame.reshape(batch_size, self.out_channels, self.height, self.width)
                    output_frames.append(processed_frame)
                
                # Stack frames back together
                stacked = torch.stack(output_frames, dim=1)
                print(f"Final output shape: {stacked.shape}")
                return stacked
        
        else:  # Single frame input
            print("Processing single frame")
            # Reshape to (batch_size, channels*height*width)
            x_flat = x.reshape(batch_size, -1)
            print(f"Flattened shape: {x_flat.shape}")
            
            # Check if the matrix sizes match
            expected_size = self.in_channels * self.height * self.width
            actual_size = x_flat.shape[1]
            if expected_size != actual_size:
                print(f"WARNING: Expected flattened size {expected_size} but got {actual_size}")
            
            x_out = self.mlp(x_flat)
            print(f"MLP output shape: {x_out.shape}")
            
            # Reshape back to (batch_size, out_channels, height, width)
            return x_out.reshape(batch_size, self.out_channels, self.height, self.width)


# Example usage and training code
def train_channel_reducer(model, train_loader, num_epochs, learning_rate=1e-3, device='cuda'):
    """
    Training function for the channel reducer model
    
    Args:
        model (ChannelReducerMLP): The model to train
        train_loader (DataLoader): DataLoader for training data
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimization
        device (str): Device to train on ('cuda' or 'cpu')
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # You might want to use a different loss depending on your needs
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Assuming batch contains both input and target tensors
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Example of how to create and use the model
if __name__ == "__main__":
    # Example parameters
    in_channels = 4
    out_channels = 3
    height = 224
    width = 398
    hidden_layers = [512, 256, 128]
    batch_size = 2
    frames = 50
    
    # Create model
    model = ChannelReducerMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        hidden_layers=hidden_layers,
        process_full_video=False  # Process frame by frame
    )
    
    # Example input
    x = torch.randn(batch_size, frames, in_channels, height, width)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 