### Scene 0
![Scene 0](nuscenes_scene_0.gif)


# Multi-View BEV Perception

## What the Model Does
This repository contains a deep learning model that takes six synchronized camera feeds from a vehicle (front, back, and four corners) and projects them into a single, top-down Bird's-Eye View (BEV) map. 

Instead of relying on LiDAR or depth sensors, the network learns to estimate 3D space directly from 2D images. The final output is a 200x200 binary grid that classifies the area around the car into two categories: drivable space and obstacles.

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
*(Note: Upload your generated graphs and inference images to the repository and link them below)*

**1. Inference Output (6 Cameras to BEV Projection)**
![Inference Graphic](presentation_graphic.png)

**2. Training Metrics (Loss and IoU)**
![Training Metrics](training_metrics_graph.png)

## How to Run Inference
To test the model on a synchronized frame, place the `bev_latest.pth` weights in the `checkpoints/` directory and run:

```bash
python inference.py
