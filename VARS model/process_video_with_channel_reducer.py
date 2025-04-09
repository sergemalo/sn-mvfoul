import torch
import os
from pathlib import Path
from torchvision.io.video import read_video
from channel_reducer import ChannelReducer
import numpy as np
import cv2

# Global variables
CHANNEL_REDUCER_PATH = "models/channel_reducer.pt"
VIDEO_PATH = "models/action_12/clip_1.mp4"  # Replace with your video path
DEPTH_PATH = "models/action_12/depth/clip_1.mp4"  # Replace with your depth video path
OUTPUT_PATH = "models/reduced_video.mp4"  # Replace with desired output path
START_FRAME = 0
END_FRAME = 100
FPS = 25

def load_channel_reducer(model_path: str) -> ChannelReducer:
    """
    Load the channel reducer model from the specified path.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        ChannelReducer: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = ChannelReducer.load_model(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set to evaluation mode
    return model

def load_and_process_video(video_path: str, depth_path: str) -> torch.Tensor:
    """
    Load and process video and depth data into a tensor suitable for the channel reducer.
    
    Args:
        video_path: Path to the video file
        depth_path: Path to the depth video file
        
    Returns:
        torch.Tensor: Processed tensor with shape (1, views, channels, frames, height, width)
    """
    # Load videos
    video, _, _ = read_video(video_path, output_format="THWC", pts_unit="sec")
    depth_video, _, _ = read_video(depth_path, output_format="THWC", pts_unit="sec")

    # Ensure video is loaded correctly
    if video is None:
        raise ValueError(f"Video is not loaded correctly. Path: {video_path}")
    
    # Ensure depth video is loaded correctly
    if depth_video is None:
        raise ValueError(f"Depth video is not loaded correctly. Path: {depth_path}")
    
    # Convert to float and normalize to [0, 1]
    video = video.float()
    depth_video = depth_video.float()

    # Process frames
    final_frames = process_video_frames(video)
    depth_final_frames = process_video_frames(depth_video)

    # Print final_frames shape
    print(f"Final frames shape before cat: {final_frames.shape}")
    print(f"Depth final frames shape before cat: {depth_final_frames.shape}")
    
    # Add depth channel
    final_frames = torch.cat((final_frames, depth_final_frames[0:1, :, :, :]), 0)
    # final_frames = torch.cat((final_frames, depth_final_frames[:, 0:1, :, :]), 1)
    
    # Reshape for channel reducer (1, views, channels, frames, height, width)
    final_frames = final_frames.unsqueeze(0).unsqueeze(0)  # Add batch and view dimensions


    # Print final_frames shape
    print(f"Final frames shape before permute: {final_frames.shape}")

    final_frames = final_frames.permute(0, 1, 2, 3, 4, 5)  # Ensure correct order

    # Print final_frames shape
    print(f"Final frames shape after permute: {final_frames.shape}")

    
    return final_frames


def process_video_frames(video: torch.Tensor) -> torch.Tensor:
    """Process video frames"""
    # Extract frames between start and end
    frames = video[START_FRAME:END_FRAME, :, :, :]
    
    # Downsample frames
    final_frames = None
    for j in range(len(frames)):
        frame = frames[j, :, :, :].unsqueeze(0)
        final_frames = frame if final_frames is None else torch.cat((final_frames, frame), 0)
    
    # In case of empty final_frames, return None
    if final_frames is None:
        return None

    # Apply transformations
    final_frames = final_frames.permute(0, 3, 1, 2)
    
    return final_frames.permute(1, 0, 2, 3)

def save_tensor_as_video(tensor: torch.Tensor, output_path: str, sort_channels: bool = False):
    """
    Save the output tensor as a video file.
    
    Args:
        tensor: Output tensor from channel reducer
        output_path: Path to save the video
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensor to numpy array
    video_data = tensor.squeeze(0).squeeze(0)  # Remove batch and view dimensions
    video_data = video_data.permute(1, 2, 3, 0)  # (frames, height, width, channels)
    video_data = video_data.cpu().numpy() 

    # Print video_data shape
    print(f"Video data shape: {video_data.shape}")

    if sort_channels:
        # Sort channels to retrieve the RGB order
        video_data = video_data[:, :, :, [2, 1, 0]]
    
    # Get video dimensions
    height, width = video_data.shape[1:3]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    # Write frames
    for frame in video_data:
        # Convert from float to uint8
        # frame = (frame * 255).astype(np.uint8)
        frame = frame.astype(np.uint8)
        out.write(frame)
    
    out.release()

def main():
    """
    Main function to process video through channel reducer.
    """
    try:
        # Load model
        print("Loading channel reducer model...")
        model = load_channel_reducer(CHANNEL_REDUCER_PATH)
        
        # Load and process video
        print("Loading and processing video...")
        input_tensor = load_and_process_video(VIDEO_PATH, DEPTH_PATH)
        
        # Process through channel reducer
        print("Processing through channel reducer...")
        with torch.no_grad():
            output_tensor = model(input_tensor)
            # --------------------------------------------------------------------------
            # TESTING
            # output_tensor = input_tensor
            # print(f"Output tensor shape: {output_tensor.shape}")
            # Output tensor shape: torch.Size([1, 1, 4, 100, 224, 398])
            # Drop last channel
            # output_tensor = output_tensor[:, :, :-1, :, :, :]
            # print(f"Output tensor shape after dropping last channel: {output_tensor.shape}")
            # --------------------------------------------------------------------------
        
        # Save output video
        print("Saving output video...")
        save_tensor_as_video(output_tensor, OUTPUT_PATH, sort_channels=True)
        
        print(f"Processing complete. Output saved to {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 