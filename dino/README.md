# DINOv2 Depth Estimation & Segmentation

In order to process the videos, you need to download the dataset first.

Run the following script to download the dataset (from the roor directory):

```bash
python download-dataset.py
```

This will download the dataset and extract it in the `dataset` directory.
Then, you have to unzip the to `source` folders.

At this point, the `dataset` directory should look like this:

```
dataset
└── depth
    ├── challenge
    ├── test
    ├── train
    └── valid
```
## Depth Estimation

To estimate the depth of the videos, you must first run the docker container.

From the dino directory, run the following command:

```bash
docker compose up -d depth-estimation
```

This will start the docker container and the depth estimation service.

Once the container is running, you can estimate the depth of the videos by running the following command:

```bash
python depth_videos.py
```

This will process all the videos in the `dataset/source` directory and save the depth maps in the `dataset/depth` directory, using the same subfolder structure.

Once the depth estimation is finished, you can stop the docker container by running the following command:

```bash
docker compose down depth-estimation
```

## Segmentation

To segment the videos, you must first run the docker container.

From the dino directory, run the following command: 

```bash
docker compose up -d segmentation
``` 

This will start the docker container and the segmentation service.

Once the container is running, you can segment the videos by running the following command:

```bash
python segmentation_videos.py
``` 

This will process all the videos in the `dataset/source` directory and save the segmentation masks in the `dataset/segmentation` directory, using the same subfolder structure.

Once the segmentation is finished, you can stop the docker container by running the following command:

```bash
docker compose down segmentation
``` 



