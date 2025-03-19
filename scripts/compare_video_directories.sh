#!/bin/bash

# Check if both source and target directory arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_directory> <target_directory>"
    exit 1
fi

# Source and target directories
SOURCE_DIR="$1"
TARGET_DIR="$2"

# Check if directories exist
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR does not exist"
    exit 1
fi

# Create output file for results
OUTPUT_FILE="missing_videos.txt"
echo "Videos present in $SOURCE_DIR but missing from $TARGET_DIR:" > "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"

# First, use rsync to do a dry run and get the list of differences
echo "Analyzing differences between directories..."
rsync -avn --include="*/" --include="*.mp4" --include="*.avi" --include="*.mov" --include="*.mkv" --exclude="*" "$SOURCE_DIR/" "$TARGET_DIR/" > rsync_output.tmp

# Process each video file in source directory
find "$SOURCE_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | while read -r source_video; do
    # Get relative path by removing source directory prefix
    rel_path=${source_video#"$SOURCE_DIR/"}
    target_video="$TARGET_DIR/$rel_path"
    
    # Check if the file exists in target directory
    if [ ! -f "$target_video" ]; then
        echo "Missing: $rel_path" | tee -a "$OUTPUT_FILE"
    fi
done

# Clean up temporary file
rm -f rsync_output.tmp

# Print summary
echo "----------------------------------------"
echo "Analysis complete. Results are in $OUTPUT_FILE"
echo "To sync missing files, you can run:"
echo "rsync -av --include=\"*/\" --include=\"*.mp4\" --include=\"*.avi\" --include=\"*.mov\" --include=\"*.mkv\" --exclude=\"*\" \"$SOURCE_DIR/\" \"$TARGET_DIR/\"" 