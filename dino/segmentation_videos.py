#!/usr/bin/env python3

import logging
from pathlib import Path
from segmentation_client import SegmentorClient
import sys
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('segmentation_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    source_dir = Path("../dataset/source")
    segmentation_dir = Path("../dataset/segmentation")
    
    if not source_dir.exists():
        logger.error(f"Source directory {source_dir} does not exist!")
        sys.exit(1)
    
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories checked/created successfully")
    return source_dir, segmentation_dir

def get_video_files(source_dir: Path) -> list:
    """Recursively get all video files from source directory.
    
    Args:
        source_dir: Path to source directory
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4'} # {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(source_dir.rglob(f"*{ext}"))

    # Sort by path name
    video_files.sort()
    
    logger.info(f"Found {len(video_files)} video files to process")
    return video_files

def get_output_path(video_path: Path, source_dir: Path, segmentation_dir: Path) -> Path:
    """Generate output path maintaining directory structure.
    
    Args:
        video_path: Path to source video
        source_dir: Base source directory
        segmentation_dir: Base segmentation directory
        
    Returns:
        Path where segmentation video should be saved
    """
    relative_path = video_path.relative_to(source_dir)
    return segmentation_dir / relative_path

def main():
    """Main function to process videos."""
    logger.info("Starting segmentation video processing")
    
    # Initialize client
    client = SegmentorClient()
    
    # Check CUDA status
    try:
        cuda_status = client.get_cuda_status()
        logger.info(f"CUDA Status: {cuda_status}")
    except Exception as e:
        logger.warning(f"Could not get CUDA status: {e}")
    
    # Setup directories
    source_dir, segmentation_dir = setup_directories()
    
    # Get video files
    video_files = get_video_files(source_dir)
    
    # Process videos
    for video_path in tqdm(video_files, desc="Processing videos"):
        output_path = get_output_path(video_path, source_dir, segmentation_dir)
        
        # Skip if output already exists
        if output_path.exists():
            # logger.debug(f"Skipping existing video: {output_path}")
            continue
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # logger.info(f"Processing video: {video_path}")
        try:
            client.get_segmentation_video(
                video_path=video_path,
                output_path=output_path,
                # scale_factor=1.0,
                codec='mp4v',
                classes_only='person'
            )
            # logger.info(f"Successfully processed: {video_path} -> {output_path}")
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    main() 