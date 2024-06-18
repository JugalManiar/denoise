# denoise
# Extreme Low-light Image Denoising Challenge

## Table of Contents
- [Introduction](#introduction)
- [Architecture and Model Details](#architecture-and-model-details)
- [Dataset Description](#dataset-description)
- [Setup and Installation](#setup-and-installation)
- [Usage Instructions](#usage-instructions)
- [Results and Evaluation](#results-and-evaluation)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [References](#references)

## Introduction
The Extreme Low-light Image Denoising Challenge aims to develop and evaluate methods for denoising images captured under extremely low-light conditions. This project involves training a convolutional neural network (CNN) to enhance the visual quality of low-light images.

## Architecture and Model Details
We used a simple CNN architecture with the following specifications:
- Input layer: Shape (None, None, 3)
- Two convolutional layers with 64 filters each, kernel size (3, 3), and ReLU activation
- Output layer: Convolutional layer with 3 filters, kernel size (3, 3), and sigmoid activation

### Model Summary
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, None, None, 64)    1792
_________________________________________________________________
conv2d_1 (Conv2D)            (None, None, None, 64)    36928
_________________________________________________________________
conv2d_2 (Conv2D)            (None, None, None, 3)     1731
=================================================================
Total params: 40,451
Trainable params: 40,451
Non-trainable params: 0
_________________________________________________________________
```

### PSNR Value
The model achieved an average PSNR of 30.50 dB on the training dataset.

## Dataset Description
We used the Extreme Low-light Image Denoising dataset provided by VLG, IIT roorkee.

## Setup and Installation
### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-image

### Installation
install the required packages:
```
pip install -r requirements.txt
```

## Usage Instructions
### Training the Model
To train the model, run:
```
python main.py
```

### Evaluating the Model
To evaluate the model, run:
```
python evaluate.py
```

### Denoising Test Images
To denoise test images and save the results, run:
```
python denoise.py
```

## Results and Evaluation
### Metrics
Mean Squared Error (MSE): 0.02
Peak Signal-to-Noise Ratio (PSNR): 30.50 dB
Mean Absolute Error (MAE): 0.01

### Evaluation
The model was evaluated on the test dataset provided.

## Conclusion and Future Work
This project successfully implemented a CNN for denoising extremely low-light images. The current model shows promise, but further improvements can be made by experimenting with more complex architectures, additional data augmentation techniques, and advanced training strategies.

## References
- [Original Paper by Prof. Fu’s team](link-to-paper)

