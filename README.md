# Semister Project Report

## Project Title: ** Human Action Detection Using Pose Estimation and Machine Learning**

---

## Abstract

SmartPose.AI is a machine learning-powered system that detects human actions in real-time using webcam input. By leveraging pose estimation techniques (MediaPipe) and a Random Forest classifier, the system classifies physical human actions like sitting, standing, clapping, drinking, etc. The frontend is developed using Streamlit, offering a user-friendly web interface. This project aims to provide an offline and real-time human action recognition solution for smart surveillance, health monitoring, and fitness apps.

---

## Introduction

With the rise of intelligent systems, recognizing human actions in real time has become a growing need in multiple fields such as surveillance, physical therapy, and smart home environments. However, many real-time systems rely heavily on high-end GPUs or cloud services. SmartPose.AI offers an alternative: a lightweight, efficient system that runs locally with standard hardware.

---

## Motivation

The idea was inspired by the growing use of AI in physical fitness apps and the need for automated monitoring in settings where manual observation is not feasible. This project simulates a simple AI fitness or monitoring coach capable of recognizing posture or movements without wearables or extra sensors.

---

## Novelty and Contributions

* Developed a custom pipeline using MediaPipe for pose detection
* Designed a Random Forest classifier tailored to human keypoint features
* Integrated real-time video input and batch image testing
* Built a professional Streamlit dashboard for end-user interaction
* Achieved high prediction accuracy for selected daily actions

---

## Project Directory Structure

```
SmartPose.AI/
|
├── app/                      # Streamlit interface scripts
├── data/                     # Dataset (images, CSV, JSON)
├── model/                    # Trained model files
├── team/                     # Documentation, bios, profiles
|
├── extract_train_data.py     # Pose + action label extractor
├── train_action_classifier.py# Classifier training pipeline
├── streamlit_ui.py           # Full dashboard GUI (Streamlit)
├── main.py                   # Local launcher
├── requirements.txt          # Environment setup file
├── README.md                 # Project overview
```

**📂 Folder Screenshot:**
![Project Folder Structure](attachment:/mnt/data/78dd46c8-b657-411b-b6f9-a618b1db64b7.png)

---

## Deployment Links

* 🚀 **Google Colab:** [Launch SmartPose on Colab](https://colab.research.google.com/drive/your_colab_link_here)
* 🎥 **Demo Video:** [Watch Demo on YouTube](https://youtu.be/your_video_link_here)

> *(Replace the above URLs with your actual working links)*

---

## Key Features

* Detects actions like standing, sitting, drinking, clapping, etc.
* Supports real-time webcam input
* Clean, minimal UI using Streamlit
* Supports both single frame and batch image prediction
* Exportable prediction logs

---

## Future Work

* Upgrade model using deep learning (e.g., CNN or LSTM)
* Extend support to video files and streaming platforms
* Expand dataset with more labeled human actions
* Add user login and history tracking
* Multilingual UI (Urdu, English)

---

## Conclusion

SmartPose.AI successfully demonstrates that real-time human action detection is possible using lightweight tools like MediaPipe and Random Forest. Its modular design and web-based GUI make it accessible and extensible for future innovation. The project bridges the gap between theoretical research and practical application.

---

## Acknowledgements

* Instructor: \[Sir's Name Here]
* Institution: BUETK - Department of Computer Science
* Semester: 6th Semester (BSCS)
* Contributors: Sana Ullah

---

## Appendix

* See Lab Work (Lab 1 to Lab 14) for step-by-step modules of the full system.
* Screenshots and additional images available on request.
* Source code provided in attached `.zip` file or GitHub link.

---

*End of Report*
