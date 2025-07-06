# SmartEye: Real-Time Traffic Sign Detection for Autonomous Vehicles

This project focuses on developing a robust and efficient real-time traffic sign detection and classification system using the YOLO (You Only Look Once) object detection framework. This system is designed to be a critical component for autonomous vehicles, enabling them to accurately perceive and react to traffic signs in diverse driving conditions.

## Problem Statement

Autonomous vehicles require reliable real-time perception to operate safely and legally. Traffic signs are a fundamental source of information, conveying crucial instructions like speed limits, stop signals, and warnings. Detecting these signs accurately under challenging conditions (varying light, weather, occlusions, degraded signs) and with low latency is paramount. Failures in this task can lead to hazardous situations. This project aims to address these challenges by creating a high-performance detection system.

## Objectives

The primary goals of this project are:

1.  **Model Development:** Train a YOLO-based object detection model specifically for traffic sign detection and classification.
2.  **Performance:** Achieve high detection accuracy, targeting mAP@0.5 ≥ 90%, and precision/recall ≥ 90% for critical sign types.
3.  **Real-Time Operation:** Ensure inference latency ≤ 50 milliseconds per frame on target hardware.
4.  **Robustness:** Evaluate and demonstrate the model's performance across various challenging environmental conditions.
5.  **Scalability:** Establish a reproducible pipeline for training, validation, and testing to facilitate future extensions.

## Methodology

The project follows these key steps:

1.  **Dataset Preparation:** Utilize a labeled dataset of traffic signs (details below), performing necessary preprocessing, augmentation, and train/validation/test splitting.
2.  **Model Selection:** Employ the YOLOv8 architecture, known for its balance of speed and accuracy, making it suitable for real-time applications.
3.  **Training:** Fine-tune the YOLOv8 model on the prepared dataset, incorporating augmentations to improve robustness.
4.  **Validation:** Monitor performance on a validation set to tune hyperparameters and prevent overfitting.
5.  **Testing:** Evaluate the final trained model on an independent test set using standard object detection metrics (mAP, precision, recall, latency).
6.  **Analysis:** Analyze performance metrics and failure cases to identify areas for potential improvement.

## Dataset

The project uses the **Car Detection** dataset from Kaggle (https://www.kaggle.com/datasets/pkdarabi/cardetection).

Key features of the dataset:

*   Contains images and annotations of various traffic signs.
*   Includes RGB images captured under diverse driving conditions.
*   Annotations are in YOLO format (bounding boxes and class labels).
*   The dataset is structured with predefined train, validation, and test splits.

*Note: The original dataset included a 'car' class which was filtered out during preprocessing to focus specifically on traffic signs for this project's scope.*

## Evaluation Plan

The success of the project is measured against the following criteria:

| Metric            | Success Criteria           | Failure Criteria                 |
| :---------------- | :------------------------- | :------------------------------- |
| mAP@0.5           | ≥ 90%                      | < 80%                            |
| Precision         | ≥ 90%                      | < 85%                            |
| Recall            | ≥ 90%                      | < 85%                            |
| Latency per frame | ≤ 50 ms                    | > 100 ms                         |
| Robustness        | Works in challenging conditions | Significant degradation in performance |

## Deployment Plan

A plan for deploying the trained model includes:

*   **Target Hardware:** Focus on edge devices like NVIDIA Jetson platforms or other automotive-grade AI accelerators.
*   **Integration:** Integrate the model's output with vehicle control systems, potentially via frameworks like ROS.
*   **Benchmarking:** Measure inference latency rigorously on the target hardware.
*   **Field Testing:** Validate performance with real or simulated driving data.
*   **Monitoring:** Implement systems for performance monitoring and model updates.

## Results

The trained YOLOv8m model achieved the following performance on the test set:

*   **mAP@0.5:** 0.969
*   **mAP@0.5:0.95:** 0.843
*   **Precision:** 0.961
*   **Recall:** 0.951

These metrics indicate strong performance, meeting or exceeding the defined success criteria for this project. The model demonstrates high accuracy in detecting and classifying traffic signs.

Visualizations of the training results, including confusion matrix, F1-confidence, Precision-Recall, Precision-Confidence, and Recall-Confidence curves, are available [here](/content/drive/MyDrive/YOLO_Training_Results/car_detection_run6/).

## Demo

A demo video showcasing the application of the trained model on an `.mp4` video file is available [here](LINK_TO_YOUR_DEMO_VIDEO).

## Getting Started

To replicate this project locally or run the code:

1.  Clone this repository.
2.  Install necessary dependencies (see `requirements.txt` - *Note: You might need to create this file if you haven't already, listing libraries like `ultralytics`, `kagglehub`, `opencv-python`, `PyYAML`, etc.*).
3.  Download the dataset using `kagglehub` as shown in the notebook.
4.  Follow the steps in the provided Colab notebook to preprocess data, train the model, and evaluate results.
