# Non-Invasive Blood Glucometer

A complete end-to-end Biomedical Engineering graduation project for
non-invasive blood glucose estimation using Photoplethysmography
(PPG), Machine Learning, Deep Learning, ESP32, and the MAX30102
sensor.

> **Status:** Biomedical Engineering Department - Helwan Engineering - Capital University - Graduation Project (2026)

------------------------------------------------------------------------

# Overview

Diabetes management relies heavily on frequent blood glucose
measurements, yet conventional glucometers require finger-prick blood
samples. This project explores a non-invasive alternative by estimating
blood glucose from PPG signals acquired using a MAX30102 optical sensor.

The project integrates hardware, embedded systems, signal processing,
machine learning, deep learning, and a browser-based application into
one complete workflow.

------------------------------------------------------------------------

# Demo

🎥 **Project Demonstration**

Google Drive Video:

> https://drive.google.com/file/d/16MKuDp4G-feqg6tA89Y82ZEd0qpIlKbb/view?usp=sharing

------------------------------------------------------------------------

# Objectives

-   Build a complete non-invasive glucose estimation device.
-   Acquire dual-channel PPG signals (Red & IR).
-   Preprocess noisy physiological signals.
-   Investigate multiple ML and DL pipelines.
-   Estimate glucose concentration in real time.
-   Display predictions through a modern web application.

------------------------------------------------------------------------

# Project Workflow

``` text
Finger
   │
   ▼
MAX30102 Sensor
   │
PPG Signal Acquisition
   │
ESP32 Firmware
   │
Bluetooth / Wi-Fi
   │
Web Application
   │
Signal Cleaning
   │
Segmentation
   │
Feature Extraction
   │
Machine Learning / Deep Learning
   │
Blood Glucose Prediction
```

------------------------------------------------------------------------

# Repository Structure

``` text
Firmware/
    ESP32 firmware

ML.Models/
    ML/
        Traditional machine learning pipelines

    DL/
        Deep learning pipelines

Docs/
    Graduation report
    Graduation presentation

Web/
    Browser application

LICENSE
README.md
```

------------------------------------------------------------------------

# Hardware

-   ESP32 Development Board
-   MAX30102 Optical Sensor
-   USB Cable
-   Computer
-   Finger placement on sensor

------------------------------------------------------------------------

# Software

## Firmware

-   Arduino IDE
-   ESP32 Arduino Core

## Machine Learning

-   Python
-   NumPy
-   Pandas
-   SciPy
-   Scikit-Learn
-   TensorFlow / Keras
-   Joblib

## Web

-   HTML5
-   CSS3
-   JavaScript
-   Web Bluetooth API

------------------------------------------------------------------------

# Signal Acquisition

The MAX30102 emits Red and Infrared light into the fingertip. The
reflected light intensity varies with changes in blood volume during the
cardiac cycle, producing PPG waveforms.

Both Red and IR signals are transmitted to the ESP32 for further
processing.

------------------------------------------------------------------------

# Signal Processing

The repository investigates several preprocessing pipelines.

Typical processing stages include:

-   Noise removal
-   Butterworth filtering
-   Baseline correction
-   Signal normalization
-   Motion artifact reduction
-   Segmentation
-   Feature extraction
-   Model inference

------------------------------------------------------------------------

# Machine Learning Pipelines

The project contains several traditional machine learning pipelines
exploring different preprocessing strategies and regression models.

Models investigated include:

-   Linear Regression
-   Support Vector Regression
-   K-Nearest Neighbors
-   Random Forest
-   XGBoost

Saved trained models are included in the repository.

------------------------------------------------------------------------

# Deep Learning Pipelines

The repository also includes deep learning approaches including:

-   1D CNN
-   CNN-BiLSTM Hybrid
-   One-Dimensional ResNet
-   Two-Head ResNet

Pre-trained models (.h5) are provided.

------------------------------------------------------------------------

# Web Application

The browser application provides:

-   Device connection
-   Live communication with ESP32
-   Signal visualization
-   Prediction interface
-   Responsive design
-   Progressive Web App support

------------------------------------------------------------------------

# Documentation

The Docs directory contains:

-   Complete graduation report
-   Final presentation

These documents describe the project background, diabetes fundamentals,
literature review, hardware selection, preprocessing pipelines, machine
learning methodology, deep learning experiments, implementation, and
evaluation.

------------------------------------------------------------------------

# Installation

## Firmware

1.  Install Arduino IDE.
2.  Install ESP32 board support.
3.  Install required libraries.
4.  Open the Firmware project.
5.  Upload to ESP32.

## Web

Open the Web App using Chrome or Microsoft Edge.

> https://ironturbo.github.io/GlucoSense/

## Machine Learning

Open the notebooks in the ML.Models directory and install the required
Python dependencies.

------------------------------------------------------------------------

# Future Improvements

-   Larger clinical dataset
-   Additional physiological features
-   Embedded inference directly on ESP32
-   Mobile application
-   Cloud database
-   User accounts
-   Continuous monitoring
-   Clinical validation

------------------------------------------------------------------------

# Disclaimer

This project was developed for research and educational purposes as a
Biomedical Engineering graduation project.

It is **not** an FDA-approved or clinically certified medical device and
should not be used for diagnosis or treatment decisions.

------------------------------------------------------------------------

# License

This project is released under the MIT License.

------------------------------------------------------------------------

# Acknowledgements

We thank our supervisors, faculty members, assistant engineers,
volunteers, and everyone who contributed to this project.

------------------------------------------------------------------------

# Authors

Abdelrahman Mohamed Abdelmoneim

Biomedical Engineering Department

Helwan Engineering -- Capital University

2026
