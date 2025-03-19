# Video Processing Scripts

This directory contains utility scripts for video file management and analysis.

## Scripts Overview

### 1. `check_video_frames.sh`
Finds videos that have fewer frames than a specified threshold.

```bash
./check_video_frames.sh /path/to/videos 100
```

Sample output (`videos_less_100_frames.txt`):
```
List of videos with less than 100 frames:
----------------------------------------
File: videos/short_clip.mp4 - Frames: 45
File: videos/test/incomplete.mp4 - Frames: 72
```

### 2. `find_non_standard_clips.sh`
Finds video files that don't follow the naming convention `clip_[number].[extension]`.

```bash
./find_non_standard_clips.sh /path/to/videos
```

Sample output (`non_standard_clips.txt`):
```
List of videos that don't match the pattern 'clip_[number].[extension]':
----------------------------------------
Found non-standard clip: /path/to/videos/random_name.mp4
Found non-standard clip: /path/to/videos/test/wrong-format.mp4
```

### 3. `compare_video_directories.sh`
Compares two directories and finds videos that are missing in the target directory.

```bash
./compare_video_directories.sh /path/to/source /path/to/target
```

Sample output (`missing_videos.txt`):
```
Videos present in /path/to/source but missing from /path/to/target:
----------------------------------------
Missing: subfolder1/video1.mp4
Missing: subfolder2/clip_001.mp4

To sync missing files, you can run:
rsync -av --include="*/" --include="*.mp4" --include="*.avi" --include="*.mov" --include="*.mkv" --exclude="*" "/path/to/source/" "/path/to/target/"
```

## Requirements

- `ffmpeg` (for frame counting)
- `rsync` (for directory comparison)
- `bash` shell

## Supported Video Formats

All scripts support the following video formats:
- `.mp4`
- `.avi`
- `.mov`
- `.mkv` 