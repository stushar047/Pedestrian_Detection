# Pedestrian_Detection

## Project Overview

This project presents a computer vision and machine learning pipeline for **pedestrian detection**. The workflow begins with raw input images or video frames and processes them through preprocessing and detection models to identify pedestrians. The final goal is to accurately localize pedestrians using bounding boxes for applications such as autonomous driving, surveillance, and robotics.

<img src="pedestraian.jpg" alt="Alt text" style="height: 200; width: auto;">

The repository is organized around two main stages:

1. **Image preprocessing and pedestrian detection**
2. **Post-processing, visualization, and evaluation**

### Important Code Files

#### `main.py`
This is the main execution script. It loads input images or video, runs the pedestrian detection pipeline, and visualizes or saves the detection results. It acts as the entry point of the system.

Key responsibilities:
- Load input images or video streams
- Run the detection pipeline
- Display or save output with bounding boxes

#### `detector.py`
This file contains the core pedestrian detection logic. It applies the detection model or algorithm to identify pedestrians and generate bounding boxes along with confidence scores.

Key responsibilities:
- Perform model inference
- Detect pedestrians in input frames
- Generate bounding boxes and scores

#### `utils.py`
This file contains helper functions for preprocessing, visualization, and post-processing.

Key responsibilities:
- Image preprocessing and resizing
- Drawing bounding boxes
- Filtering detections
- Handling post-processing operations

#### `model/`
This directory contains trained models or configuration files used for pedestrian detection.

#### `data/`
This directory contains input images or videos used for testing and evaluation.

### Overall Pipeline

The complete project workflow can be summarized as:

**Input image / video frame**  
→ **Image preprocessing**  
→ **Feature extraction / model inference**  
→ **Pedestrian detection**  
→ **Bounding box generation**  
→ **Post-processing (filtering / suppression)**  
→ **Final detection visualization**  

### Repository Outputs

The repository generates:
- Images or video frames with detected pedestrians
- Bounding boxes around pedestrians
- Confidence scores for each detection

## System Setup

Run the following lines to set up the system 

* Create a new environment
  * conda create --name pedestrian
* Activate environment
  * conda activate pedestrian
* Install all libraries
  * pip install opencv-python
  * pip install numpy
  * pip install matplotlib
  * pip install scikit-learn
  * pip install pillow
  * install git [installation instruction for windows](https://github.com/git-guides/install-git)
* Change the directory where you want to download all the files
  * cd Directory (Example: cd C:/Users/tushar/Documents) 
* Download all the files
  * git clone https://github.com/stushar047/Pedestrian_Detection.git  

## Run the code

* For running the code, always make sure that you are in the correct environment and directory 
  * conda activate pedestrian
  * cd Pedestrian_Detection
* Run the code
  * python main.py --input <path_to_image_or_video>
  * Example: python main.py --input data/sample.jpg

## Collect all the files required

There will be two types of outputs:

1. Image / Video outputs<br>
Output files include images or video frames with detected pedestrians and bounding boxes.

2. Detection results<br>
Bounding boxes and confidence scores for detected pedestrians.

## Check the results

* Run the code and verify:
  * Bounding boxes correctly detect pedestrians
  * False detections are minimal
  * Detection consistency across frames (for video)

## Paper References
- MS Thesis References - https://digital.library.txst.edu/items/28c90d7f-cc88-4358-a654-9b46173ffac6
