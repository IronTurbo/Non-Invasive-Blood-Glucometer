# Non-Invasive Blood Glucometer

A complete end-to-end non-invasive blood glucose estimation system that
combines **PPG signal acquisition**, **machine learning**, **deep
learning**, and a **real-time web application** using the **ESP32** and
**MAX30102** sensor.

This project was developed as a Biomedical Engineering graduation
project to investigate the feasibility of estimating blood glucose
levels from photoplethysmography (PPG) signals without drawing blood.

------------------------------------------------------------------------

## Demo

🎥 **Project Demonstration**

Watch the complete project in action:

> **Google Drive Demo:** *(Insert your Google Drive video link here)*

------------------------------------------------------------------------

## Project Overview

The system acquires infrared (IR) and red PPG signals from a MAX30102
sensor connected to an ESP32. The acquired signals are transmitted
wirelessly to a web application where signal preprocessing and machine
learning models estimate the user's blood glucose level in real time.

The repository contains: - ESP32 firmware - Web application - Machine
learning models - Deep learning models - Project documentation - Final
presentation

------------------------------------------------------------------------

## Features

-   Non-invasive blood glucose estimation
-   Real-time PPG acquisition
-   ESP32 + MAX30102 implementation
-   Wi-Fi communication
-   Browser-based interface
-   Signal preprocessing pipeline
-   Multiple machine learning models
-   Deep learning models
-   Progressive Web App (PWA)

------------------------------------------------------------------------

## System Architecture

``` text
Finger
  ↓
MAX30102 Sensor
  ↓
ESP32
  ↓ Wi-Fi
Web Application
  ↓
Signal Processing
  ↓
Machine Learning Model
  ↓
Estimated Blood Glucose Level
```

------------------------------------------------------------------------

## Hardware

-   ESP32 Development Board
-   MAX30102 Sensor
-   USB Cable
-   Computer

## Software Stack

### Firmware

-   Arduino IDE
-   ESP32 Arduino Framework

### Machine Learning

-   Python
-   NumPy
-   Pandas
-   SciPy
-   Scikit-learn
-   TensorFlow / Keras
-   Joblib

### Web Application

-   HTML
-   CSS
-   JavaScript
-   Web Bluetooth API

------------------------------------------------------------------------

## Repository Structure

``` text
Firmware/
Web/
ML.Models/
Docs/
LICENSE
README.md
```

------------------------------------------------------------------------

## Getting Started

1.  Clone the repository.
2.  Upload the firmware to the ESP32 using Arduino IDE.
3.  Open the web application.
4.  Connect to the ESP32.
5.  Start a measurement.

------------------------------------------------------------------------

## Documentation

The `Docs` folder contains the final graduation report and presentation.

------------------------------------------------------------------------

## Future Work

-   Larger dataset
-   Embedded inference on ESP32
-   Mobile application
-   Improved accuracy
-   Cloud synchronization

------------------------------------------------------------------------

## License

MIT License.

------------------------------------------------------------------------

## Disclaimer

This project is intended for educational and research purposes only. It
is **not** a certified medical device and should not be used for medical
diagnosis or treatment.
