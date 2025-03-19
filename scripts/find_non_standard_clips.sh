#!/bin/bash

# Check if directory path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Target directory
TARGET_DIR="$1"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

# Create output file for results
OUTPUT_FILE="non_standard_clips.txt"
echo "List of videos that don't match the pattern 'clip_[number].[extension]': " > "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"

# Find all video files and check their names
find "$TARGET_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | while read -r video; do
    # Get just the filename without the path
    filename=$(basename "$video")
    
    # Check if the filename matches the pattern clip_[number].[extension]
    if ! [[ $filename =~ ^clip_[0-9]+\.[^.]+$ ]]; then
        echo "Found non-standard clip: $video" | tee -a "$OUTPUT_FILE"
    fi
done

echo "Analysis complete. Results are in $OUTPUT_FILE" 