import torch
import torch.nn as nn

class ChannelReducerMLP(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 height: int,
                 width: int,
                 hidden_layers: list[int] = [128, 64],
                 process_full_video: bool = False):
        """
        MLP model to reduce number of channels in video/images
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            height (int): Height of input images/video
            width (int): Width of input images/video
            hidden_layers (list[int]): List of hidden layer sizes
            process_full_video (bool): If True, process entire video at once. If False, process frame by frame
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.process_full_video = process_full_video
        
        # Build MLP layers
        layers = []
        input_size = height * width * in_channels
        
        # Add hidden layers
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        
        # Add output layer
        output_size = height * width * out_channels
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
        
        if len(original_shape) == 5:  # Video input
            if self.process_full_video:
                # Process entire video at once
                # Reshape to (batch_size, frames*channels*height*width)
                x = x.reshape(batch_size, -1)
                x = self.mlp(x)
                # Reshape back to (batch_size, frames, out_channels, height, width)
                return x.reshape(batch_size, original_shape[1], self.out_channels, self.height, self.width)
            else:
                # Process frame by frame
                frames = original_shape[1]
                output_frames = []
                
                for frame_idx in range(frames):
                    frame = x[:, frame_idx]  # (batch_size, channels, height, width)
                    # Reshape to (batch_size, channels*height*width)
                    frame = frame.reshape(batch_size, -1)
                    # Process through MLP
                    processed_frame = self.mlp(frame)
                    # Reshape to (batch_size, out_channels, height, width)
                    processed_frame = processed_frame.reshape(batch_size, self.out_channels, self.height, self.width)
                    output_frames.append(processed_frame)
                
                # Stack frames back together
                return torch.stack(output_frames, dim=1)
        
        else:  # Single frame input
            # Reshape to (batch_size, channels*height*width)
            x = x.reshape(batch_size, -1)
            x = self.mlp(x)
            # Reshape back to (batch_size, out_channels, height, width)
            return x.reshape(batch_size, self.out_channels, self.height, self.width)


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