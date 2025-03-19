#!/bin/bash

# Check if both arguments (directory path and frame count) have been provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <directory_path> <frame_threshold>"
    exit 1
fi

# Target directory and frame threshold
TARGET_DIR="$1"
FRAME_THRESHOLD="$2"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

# Check if frame threshold is a valid number
if ! [[ "$FRAME_THRESHOLD" =~ ^[0-9]+$ ]]; then
    echo "Error: Frame threshold must be a positive number"
    exit 1
fi

# Create output file for results
OUTPUT_FILE="videos_less_${FRAME_THRESHOLD}_frames.txt"
echo "List of videos with less than ${FRAME_THRESHOLD} frames:" > "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"

# Function to process each video file
process_video() {
    local video="$1"
    # Display video name
    echo "Processing video: $video"

    # Get number of frames using ffprobe
    frames=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$video")
    
    # Check if $frames is a valid number
    if ! [[ "$frames" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not get frame count for $video"
        return 1
    fi

    # Check if number of frames is less than threshold
    if [ "$frames" -lt "$FRAME_THRESHOLD" ]; then
        echo "File: $video - Frames: $frames" >> "$OUTPUT_FILE"
    fi
}

# Find all video files and process them
export -f process_video
export FRAME_THRESHOLD
export OUTPUT_FILE
find "$TARGET_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -exec bash -c 'process_video "$0"' {} \;

echo "Analysis complete. Results are in $OUTPUT_FILE"
