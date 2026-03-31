<img width="1589" height="675" alt="inference" src="https://github.com/user-attachments/assets/9c1648fe-0eba-4c80-9317-743f018a00ca" /># Multi-View BEV Perception

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
* **Feature Extraction:** We used a pre-trained ResNet-34 as the backbone. We froze the early layers to keep its basic edge-detection abilities and only trained the deeper layers to understand our specific environment.
* **Spatial Projection:** We used an adaptive pooling method to collapse the vertical data from the camera images and stretch it across a flat horizontal plane.

**2. The Custom Loss Function**
Our biggest hurdle was the "Empty Space Problem." Because 95% of a BEV map is just empty road, a standard loss function (like BCE) takes the easy way out and just guesses "empty space" everywhere. To fix this, we built a custom loss function:
* **Focal Loss (20%):** Penalized the model heavily for missing actual vehicles.
* **Dice Loss (60%):** Forced the model to focus on the actual shapes and overlap of the cars rather than just counting pixels.
* **Boundary Loss (20%):** Acted as a filter to sharpen the blurry edges of the predicted bounding boxes.
## Model Metrics
During our optimization phase, we trained two different backbone architectures to find the best balance between spatial accuracy and model stability on our highly constrained 404-frame dataset.

**1. ResNet-18 (Our Final Choice)**
* **Best Validation Loss:** 0.3803
* **Peak Intersection over Union (IoU):** ~21.6%

**2. ResNet-34 (Experimental Upgrade)**
* **Best Validation Loss:** 0.4126
* **Peak Intersection over Union (IoU):** ~24.09%

**The Performance Leap:** Both models achieved a >100% relative improvement in structural accuracy compared to our initial baseline (which used standard BCE loss and flatlined at just 9% IoU). 

### Why We Deployed ResNet-18
While upgrading to the deeper ResNet-34 gave us a slight bump in our peak IoU, we noticed its validation loss actually got worse (0.4126 compared to 0.3803). Because our dataset is strictly limited to 404 frames, the larger 21-million parameter ResNet-34 had too much capacity and began to overfit the training data. 

We deliberately selected the ResNet-18 as our final architecture. Its healthier, lower validation loss proves it is actually generalizing the geometry better, rather than just memorizing the dataset. Additionally, a lighter model is much more efficient for real-time deployment on vehicle edge hardware.

Getting a 21.6% IoU on a dataset of only 404 frames proves that the network successfully learned how to cross-reference multiple cameras to map physical boundaries, rather than just memorizing the training data.

## Visualizations

**1. Inference Output (6 Cameras to BEV Projection)**

RESNET-34's output
<img width="1589" height="675" alt="presentation_graphic" src="https://github.com/user-attachments/assets/370d5d89-265d-4d7d-b4d5-bdbdfa68beaa" />

RESNET-18's output
<img width="1589" height="675" alt="inference" src="https://github.com/user-attachments/assets/1dd05f1c-f1e6-4db4-843c-9a245823bbaf" />



**2. Training Metrics (Loss and IoU)**

RESNET-34's output
<img width="1390" height="590" alt="training_metrics_graph" src="https://github.com/user-attachments/assets/6943c6cd-398f-4cec-9ee8-8407a6690043" />


RESNET-18's output
<img width="1389" height="590" alt="graphs" src="https://github.com/user-attachments/assets/968f0624-c580-4597-8c70-ae7528d863e8" />


note:-
all the graphs and the inference can be obtained by the .ipynb file itself 



THE LINK FOR THE DATASET:-
https://drive.google.com/drive/folders/1m5XFVvy8lJi723Azd9BWxLHL051jHZH7?usp=sharing

THE LINK FOR THE RESNET-34 MODEL:-
https://drive.google.com/file/d/1ypmFQu3jE3gfjR-oU3ucs4MRlsYOW7ku/view?usp=drive_link

THE LINK FOR THE RESNET-18 MODEL:-
https://drive.google.com/file/d/1GpTjvbweM8gln-qtOpDeDGkXZ6NdYJcj/view?usp=sharing


## Reproducing the Training in Google Colab

If you want to train the model from scratch or run the full pipeline using the provided Google Colab notebook (`mahe_mobility.ipynb`), you must structure your Google Drive correctly so the script can find the images and save the model weights.

### Google Drive Directory Structure
First, download the full `master_dataset` from the provided link. Then, create a folder named `mahe_mobility` in the root of your Google Drive and upload the dataset there. 

Your Google Drive must look exactly like this:

```text
My Drive/
└── mahe_mobility/
    └── master_dataset/
        ├── CAM_FRONT/
        ├── CAM_FRONT_LEFT/
        ├── CAM_FRONT_RIGHT/
        ├── CAM_BACK/
        ├── CAM_BACK_LEFT/
        └── CAM_BACK_RIGHT/

# These paths are defined at the top of the notebook
DATASET_PATH = "/content/drive/MyDrive/mahe_mobility/master_dataset"
CHECKPOINT_DIR = "/content/drive/MyDrive/mahe_mobility/models"

bev_latest.pth (The most recent epoch weights)
bev_best.pth (The weights with the highest validation accuracy)
training_log.csv (The epoch-by-epoch loss and IoU metrics)
presentation_graphic.png (The final spatial projection output)
