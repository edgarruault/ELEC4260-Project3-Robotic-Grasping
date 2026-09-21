# ELEC4260 Project 3 --- Data-Driven Grasp Detection for Robotic Manipulation

This repository contains my work for **ELEC 4260 Project 3**, an
end-to-end robotic grasping project combining neural grasp detection,
RGB-D perception, simulation, camera--robot calibration, and deployment
on a physical Airbot arm.

## Project Overview

The system implements the following perception-to-action pipeline:

1.  Capture RGB-D observations.
2.  Predict a 2D grasp using a neural network.
3.  Convert the 2D grasp into a 3D target using depth and camera
    intrinsics.
4.  Transform the target from the camera frame to the robot base frame.
5.  Generate pre-grasp, grasp, post-grasp, and lift poses.
6.  Execute the grasp in simulation or on the physical robot.

The grasp detector was trained on the **Cornell Grasp Dataset**. Several
input modalities and model configurations were evaluated before
deploying the selected RGB-D model.

## Neural Grasp Detection

The baseline GraspCNN uses an encoder-decoder architecture with feature
dimensions:

`32 → 64 → 128 → 64 → 32`

The network has **447,172 trainable parameters**.

The final deployed model uses **4-channel RGB-D input**. RGB channels
are normalized and zero-centered, while depth is mean-centered and
clipped. The same preprocessing is used during training and inference.

The detector predicts a grasp quality map, angle map, and gripper width.
During inference, border suppression and candidate filtering are applied
to reduce spurious detections near image boundaries.

The selected checkpoint was:

`epoch_90_iou_1.00`

It was chosen because the baseline RGB-D model reached repeated **1.00
validation IoU** while remaining smaller and faster than the larger
architectures tested.

## Simulation

The neural detector was integrated into the **DISCOVERSE** simulation
pipeline while retaining the original circle detector as a geometric
baseline.

The simulation pipeline performs:

-   RGB-D image acquisition
-   neural grasp prediction
-   2D-to-3D grasp conversion
-   pre-grasp and post-grasp generation
-   inverse-kinematics-based robot motion
-   grasp execution and lift
-   automated evaluation

The final benchmark evaluated **5 objects with 10 trials per object**.

  Object          Successful grasps   Success rate
  ------------- ------------------- --------------
  Banana                     6 / 10            60%
  Lemon                      7 / 10            70%
  Mug                        9 / 10            90%
  Screwdriver                8 / 10            80%
  Sugar box                  9 / 10            90%
  **Overall**           **39 / 50**        **78%**

The main simulation failures were caused by inverse-kinematics
exceptions at extreme poses and grasps where the object was reached but
not lifted above the success threshold.

## Real-World Deployment

The trained RGB-D model was deployed on a physical **Airbot arm** using
an **Intel RealSense depth camera**.

The real-world pipeline includes:

-   RealSense RGB-D streaming
-   camera intrinsic parameters
-   hand-eye / camera-to-robot calibration
-   neural RGB-D grasp inference
-   depth-based 2D-to-3D projection
-   camera-to-robot coordinate transformation
-   gripper-width control
-   physical grasp execution

Five representative object configurations were evaluated with three
trials each.

  Object                 Successful grasps
  ------------------- --------------------
  Flat mint box                      0 / 3
  Vertical mint box                  1 / 3
  Tape roll                          2 / 3
  Pen                                0 / 3
  Ball                               1 / 3
  **Overall**           **4 / 15 (26.7%)**

The hardware experiments exposed a significant sim-to-real gap. The main
limitations were calibration and pose bias, grasp-angle errors, border
artifacts in the predicted quality map, and sensitivity to grasp height.
Narrow objects such as the pen were particularly sensitive to small
translation and orientation errors.

## Repository Structure

``` text
.
├── README.md
├── grasp_object.py
├── grasp_utils.py
├── GraspDetector.py
├── GraspDetectorNN.py
├── record_grasp.py
├── test_detector_circle.py
├── test_pose.py
│
├── calibration/
│   ├── calibration.py
│   ├── label_points.py
│   ├── record_robot_points.py
│   ├── test_realsense.py
│   ├── calibration.json
│   ├── points.json
│   └── robot_poses.json
│
├── graspcnn/
│   ├── train_model.py
│   ├── eval_model.py
│   ├── models/
│   └── utils/
│
├── figures/
│
├── assets/
│   ├── 2D3DGrasp.png
│   └── pre-pose.png
│
├── videos/
│   └── ...
│
└── report/
    └── ELEC4260_Project3_Report.pdf
```

Large training outputs, intermediate checkpoints, datasets, TensorBoard
logs, and the complete DISCOVERSE framework are intentionally excluded
from this repository.

## Main Files

-   `grasp_object.py` --- real-world grasp execution pipeline.
-   `GraspDetector.py` --- base grasp detector and
    geometric/circle-based detector.
-   `GraspDetectorNN.py` --- neural RGB-D grasp detector and 2D-to-3D
    grasp prediction.
-   `grasp_utils.py` --- transformation, calibration, and gripper
    utilities.
-   `graspcnn/train_model.py` --- GraspCNN training and model
    experiments.
-   `calibration/` --- camera--robot calibration utilities and
    calibration data.
-   `videos/` --- representative simulation and physical-robot grasping
    examples.
-   `report/ELEC4260_Project3_Report.pdf` --- complete project report
    and experimental analysis.

## External Dependencies

The simulation experiments use the **DISCOVERSE** robotics simulation
framework. The full framework is not included here because it is an
external dependency and contains large simulator assets.

The project also relies on libraries including:

-   PyTorch
-   NumPy
-   OpenCV
-   Intel RealSense SDK / `pyrealsense2`
-   SciPy
-   Open3D

The Cornell Grasp Dataset is also not included in this repository.

## Results

The project demonstrated a complete learning-based grasping pipeline
from neural perception to physical robot execution. RGB-D input
performed substantially better than depth-only input during development.
The final system achieved **78% grasp success in simulation** and
**26.7% on the physical robot**, highlighting the remaining calibration,
localization, and grasp-orientation challenges involved in sim-to-real
deployment.

## Course

**ELEC 4260 --- Project 3: Data-Driven Grasp Detection for Robotic
Manipulation**
