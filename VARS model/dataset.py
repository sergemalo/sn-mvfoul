from torch.utils.data import Dataset
import torch
import random
import os
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
from typing import Optional, Tuple, List, Union
from pathlib import Path


class MultiViewDataset(Dataset):
    """Dataset class for handling multi-view soccer action videos.
    
    This dataset loads and processes multiple views of soccer actions, supporting
    both training and challenge splits. It handles video frame extraction, 
    transformations, and label management.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        start: int,
        end: int,
        fps: int,
        split: str,
        num_views: int,
        transform: Optional[callable] = None,
        transform_model: Optional[callable] = None,
        depth_path: Union[str, Path] = None
    ):
        """Initialize the dataset.
        
        Args:
            path: Path to the dataset
            start: Starting frame index
            end: Ending frame index
            fps: Frames per second
            split: Dataset split ('Train', 'Val', 'Test', or 'Chall')
            num_views: Number of views per action
            transform: Optional transform to apply to frames
            transform_model: Model-specific transform to apply
            depth_path: Path to the depth video dataset
        """
        self.path = path
        self.depth_path = depth_path
        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views
        
        # Calculate frame sampling factor
        self.factor = (end - start) / (((end - start) / 25) * fps)
        
        # Load dataset annotations and clips
        if split != 'Chall':
            self._load_annotations(path, num_views)
        else:
            self._load_challenge_data(path, num_views)
            
        self.length = len(self.clips)
        print(f"--> Dataset {split}: Total actions: {self.length}; "
              f"Number of views per action: {num_views}")

    def _load_annotations(self, path: Union[str, Path], num_views: int) -> None:
        """Load annotations and compute class distributions."""
        # Load labels and clips
        (self.labels_offence_severity, self.labels_action, 
         self.distribution_offence_severity, self.distribution_action,
         not_taking, self.number_of_actions) = label2vectormerge(path, self.split, num_views)
        self.clips = clips2vectormerge(path, self.split, num_views, not_taking)
        
        # Compute class distributions and weights
        self.distribution_offence_severity = torch.div(self.distribution_offence_severity, 
                                                     len(self.labels_offence_severity))
        self.distribution_action = torch.div(self.distribution_action, 
                                           len(self.labels_action))
        
        self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
        self.weights_action = torch.div(1, self.distribution_action)

    def _load_challenge_data(self, path: Union[str, Path], num_views: int) -> None:
        """Load challenge data without annotations."""
        self.clips = clips2vectormerge(path, self.split, num_views, [])

    def _process_video_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Process video frames: extract, downsample, and transform."""
        # Extract frames between start and end
        frames = video[self.start:self.end, :, :, :]
        
        # Downsample frames
        final_frames = None
        for j in range(len(frames)):
            if j % self.factor < 1:
                frame = frames[j, :, :, :].unsqueeze(0)
                final_frames = frame if final_frames is None else torch.cat((final_frames, frame), 0)
        
        # In case of empty final_frames, return None
        if final_frames is None:
            return None

        # Apply transformations
        final_frames = final_frames.permute(0, 3, 1, 2)
        if self.transform:
            final_frames = self.transform(final_frames)
        final_frames = self.transform_model(final_frames)
        
        return final_frames.permute(1, 0, 2, 3)
    
    def _get_depth_video_path(self, video_path: str) -> str:
        """Replace self.path with self.depth_path in video_path."""
        depth_path = video_path.replace(self.path, self.depth_path)
        print(f"Original video path: {video_path}")
        print(f"Depth video path: {depth_path}")
        return depth_path

    def _pick_view(self, num_views: int, previous_views: List[int]) -> int:
        """Select view index based on split type."""
        if self.split == 'Train':
            return random.choice([i for i in range(num_views) if i not in previous_views])
        else:
            return len(previous_views)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[str, int]]:
        """Get a dataset item.
        
        Returns:
            Tuple containing:
            - Offense severity label (tensor of size 4)
            - Action type label (tensor of size 8)
            - Video tensor (V, C, N, H, W)
            - Action ID
        """
        previous_views = []
        videos = None
        
        # Process each view
        for _ in range(len(self.clips[index])):
            # Limit the number of views to 2 during training
            if self.split == 'Train' and len(previous_views) == 2:
                break
                
            # Select view index
            index_view = self._pick_view(len(self.clips[index]), previous_views)
            previous_views.append(index_view)
            
            # Load and process video
            video, _, _ = read_video(self.clips[index][index_view], 
                                   output_format="THWC", 
                                   pts_unit="sec")
            final_frames = self._process_video_frames(video)

            # In case of empty final_frames, throw an error
            if final_frames is None:
                raise ValueError(f"Failed to load frames for index {index}. Available frames count: {len(video)}. Path: {self.clips[index][index_view]}")

            # --------------------------------------------------------------------------
            # Load and add depth channel
            if self.depth_path is not None:
                print(f"Adding depth channel from {self.depth_path}")
                depth_video_path = self._get_depth_video_path(self.clips[index][index_view])
                # Check if depth video path exists
                if not os.path.exists(depth_video_path):
                    print(f"Error: Depth video path '{depth_video_path}' does not exist.")
                    exit(1)

                depth_video, _, _ = read_video(depth_video_path, 
                                    output_format="THWC", 
                                    pts_unit="sec")
                depth_final_frames = self._process_video_frames(depth_video)

                # In case of empty depth_final_frames, throw an error
                if depth_final_frames is None:
                    raise ValueError(f"Failed to load depth frames for index {index}. Available frames count: {len(depth_video)}. Path: {depth_video_path}")

                # Add the first channel of depth_final_frames as a fourth channel to final_frames
                print(f"Depth final frames shape: {depth_final_frames.shape}")
                print(f"Final frames shape before concatenation: {final_frames.shape}")
                final_frames = torch.cat((final_frames, depth_final_frames[:, 0:1, :, :]), 1)
                print(f"Final frames shape after concatenation: {final_frames.shape}")
            # --------------------------------------------------------------------------
            
            # Combine views
            if videos is None:
                videos = final_frames.unsqueeze(0)
            else:
                videos = torch.cat((videos, final_frames.unsqueeze(0)), 0)
        
        # Final tensor organization
        if self.num_views not in [1, 5]:
            videos = videos.squeeze()
        videos = videos.permute(0, 2, 1, 3, 4)

        print(f"Final videos shape: {videos.shape}")
        
        # Return appropriate labels based on split
        if self.split != 'Chall':
            return (self.labels_offence_severity[index][0],
                   self.labels_action[index][0],
                   videos,
                   self.number_of_actions[index])
        return -1, -1, videos, str(index)

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        return self.length

    def get_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get class distributions for offense severity and action types."""
        return self.distribution_offence_severity, self.distribution_action

    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get class weights for offense severity and action types."""
        return self.weights_offence_severity, self.weights_action 

if __name__ == "__main__":

    # Define dataset parameters
    dataset_path = "../mvfouls-sub2-lr/videos"
    depth_path = "../mvfouls-sub2-lr/depth"
    start_frame = 25
    end_frame = 75
    fps = 25
    split = "Train"
    num_views = 3
    
    # Define a simple transform (identity transform)
    def transform_model(x):
        return x
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        exit(1)
    
    print(f"Loading dataset from {dataset_path}...")
    
    # Create dataset
    dataset = MultiViewDataset(
        path=dataset_path,
        depth_path=depth_path,
        start=start_frame,
        end=end_frame,
        fps=fps,
        split=split,
        num_views=num_views,
        transform=None,
        transform_model=transform_model
    )
    
    print(f"Dataset loaded successfully with {len(dataset)} items.")
    print(f"Class distributions:")
    offense_dist, action_dist = dataset.get_distribution()
    print(f"  Offense severity: {offense_dist}")
    print(f"  Action type: {action_dist}")
    
    # Loop through items and display info
    for i in range(min(5, len(dataset))):
        try:
            offense_label, action_label, videos, action_id = dataset[i]
            
            print("--------------------------------")
            print(f"\nItem {i}:")
            print(f"  Action ID: {action_id}")
            print(f"  Offense severity label shape: {offense_label.shape}, values: {offense_label}")
            print(f"  Action type label shape: {action_label.shape}, values: {action_label}")
            print(f"  Videos tensor shape: {videos.shape}")
            
            # Display info about each view
            num_views_actual = videos.shape[0]
            for v in range(num_views_actual):
                print(f"  View {v} shape: {videos[v].shape}")
                
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}") 