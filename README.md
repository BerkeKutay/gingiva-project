# Gingiva Periodontal Detection System

A deep learning–based system for detecting periodontal conditions from intraoral images.  
The project focuses on medical image analysis, model training, and scalable deployment practices.

## Problem Statement

Periodontal diseases are among the most common oral health problems worldwide. 
Early detection is critical for preventing severe complications. 

This project aims to develop a computer vision model capable of detecting gingival and periodontal conditions from intraoral images using deep learning techniques.

## Dataset

The model was trained using the following publicly available dataset:

Mendeley Data:
https://data.mendeley.com/datasets/3253gj88rr/1

The dataset consists of annotated intraoral images for periodontal analysis.

## Model Architecture

- Model: YOLO-based object detection
- Framework: PyTorch
- Task: Periodontal region detection
- Training environment: Jupyter Notebook


## Training Results

Below are the training and validation metrics obtained during model training:

![Training Results](runs/detect/a/dis_eti_clean/results.png)



## Model Weights

The trained model weights are hosted on Hugging Face:

https://huggingface.co/Kutay0/gingiva-periodontal-detection

The model is automatically downloaded when running the application.

## Installation

```bash
git clone https://github.com/BerkeKutay/gingiva-project.git
cd gingiva-project
pip install -r requirements.txt

## Future Improvements

- Model performance optimization
- Deployment on cloud platforms
- Real-time inference support
- Clinical validation with expert dentists

## Author

Developed by Kutay Berke  
AI & Computer Vision Enthusiast
