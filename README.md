# Multi-View BEV Perception

## What the Model Does
This repository contains a deep learning model that takes six synchronized camera feeds from a vehicle (front, back, and four corners) and projects them into a single, top-down Bird's-Eye View (BEV) map. 

Instead of relying on LiDAR or depth sensors, the network learns to estimate 3D space directly from 2D images. The final output is a 200x200 binary grid that classifies the area around the car into two categories: drivable space and obstacles.

# Ground Truth Generation (LiDAR to BEV)

To train the model to predict BEV maps from cameras, we must first establish accurate target labels. We generate this ground truth using the vehicle's LiDAR data and the nuScenes dataset API.

![Scene 0](nuscenes_scene_0.gif)

* LiDAR Projection: It extracts the 3D point cloud from the top LiDAR sensor and filters the points to a 100m x 100m bounding box centered on the ego vehicle (50 meters in all directions).

* Occupancy Grid Creation: The filtered 3D points are flattened and mapped onto a high-resolution 1000x1000 2D binary matrix at a resolution of 0.1 meters per pixel. This matrix serves as the definitive top-down map of physical obstacles and occupied space.

* Camera Calibration Extraction: The script iterates through the six camera feeds to extract camera intrinsic matrices, sensor rotations, sensor translations, and the vehicle's ego pose. This spatial metadata is essential to properly correlate the 2D camera pixels with the 3D physical coordinates represented in the LiDAR occupancy grid.

## How We Trained It
We built a custom neural network architecture and trained it on a highly constrained dataset of 404 multi-view frames. 

**1. The Architecture**
* **Feature Extraction:** We used a pre-trained ResNet-18 as the backbone. We froze the early layers to keep its basic edge-detection abilities and only trained the deeper layers to understand our specific environment.
* **Spatial Projection:** We used an adaptive pooling method to collapse the vertical data from the camera images and stretch it across a flat horizontal plane.

**2. The Custom Loss Function**
Our biggest hurdle was the "Empty Space Problem." Because 95% of a BEV map is just empty road, a standard loss function (like BCE) takes the easy way out and just guesses "empty space" everywhere. To fix this, we built a custom loss function:
* **Focal Loss (20%):** Penalized the model heavily for missing actual vehicles.
* **Dice Loss (60%):** Forced the model to focus on the actual shapes and overlap of the cars rather than just counting pixels.
* **Boundary Loss (20%):** Acted as a filter to sharpen the blurry edges of the predicted bounding boxes.

## Model Metrics
By switching to our custom loss function and moving the training to a T4 GPU, we saw a massive jump in the model's ability to understand geometry. 

* **Best Validation Loss:** 0.3803
* **Peak Intersection over Union (IoU):** ~21.6%
* **Performance Leap:** We achieved a >100% relative improvement in structural accuracy compared to our initial baseline, which had flatlined at 9% IoU. 

Getting a 21.6% IoU on a dataset of only 404 frames proves that the network successfully learned how to cross-reference multiple cameras to map physical boundaries, rather than just memorizing the training data.

## Visualizations

**1. Inference Output (6 Cameras to BEV Projection)**

<img width="1589" height="675" alt="inference" src="https://github.com/user-attachments/assets/650cf0e9-fd01-4ab5-892d-953d70b96edd" />




**2. Training Metrics (Loss and IoU)**

<img width="1389" height="590" alt="graphs" src="https://github.com/user-attachments/assets/b41bcab6-6db6-4a76-9194-355c1712fea1" />



note:-
all the graphs and the inference can be obtained by the .ipynb file itself 



THE LINK FOR THE DATASET:-
https://drive.google.com/drive/folders/1m5XFVvy8lJi723Azd9BWxLHL051jHZH7?usp=sharing

THE LINK FOR THE BEST MODEL:-
https://drive.google.com/file/d/1GpTjvbweM8gln-qtOpDeDGkXZ6NdYJcj/view?usp=sharing
