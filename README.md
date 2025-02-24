# Investigating Optimizer Performance on Custom CNN with Varying Batch Sizes

This project explores the performance of different optimization algorithms on a custom Convolutional Neural Network (CNN) across varying batch sizes. The objective is to determine how batch size influences the efficiency and effectiveness of three widely used optimizers: **RMSProp, Adam, and Stochastic Gradient Descent (SGD)**.

## Overview
The experiment evaluates batch sizes of **16, 32, and 64** to analyze their impact on model training stability, convergence speed, and overall accuracy. The findings provide insights into selecting the appropriate optimizer and batch size for CNN training, contributing to optimized deep-learning workflows.

## Dataset
- **Size**: 400 Images
- **Number of Classes**: 2
- **Sample Distribution**:
  - 400 Images for Training
  - 40 Images for Validation

## Regularization Techniques
To improve generalization and prevent overfitting, the following techniques were used:
- **Dropout**
- **Data Augmentation**

## Optimizers Tested
- **RMSProp**
- **Adam**
- **SGD**

## Results Summary
- **Adam optimizer with batch sizes of 16 and 32 achieved the highest accuracy.**
- Different batch sizes impacted convergence speed and model stability.
- Larger batch sizes resulted in a trade-off between stability and convergence rate.

## Google Colab Implementation
This project was implemented in **Google Colab**, ensuring an easy-to-run environment with GPU acceleration.

### Steps to Run the Project
1. Open the Google Colab notebook.
2. Upload the dataset or connect to an external storage like Google Drive.
3. Run the cells step by step to preprocess data, define the CNN model, and train it with different optimizers.
4. Analyze the results using visualizations such as accuracy/loss curves.

## Installation and Dependencies
Ensure you have the following dependencies installed in your Google Colab environment:

```python
!pip install tensorflow keras matplotlib numpy
```

Import necessary libraries:
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```

## Training and Evaluation
- The CNN model was trained for a fixed number of epochs with each optimizer.
- Performance was analyzed using accuracy and loss metrics.
- The impact of batch size variations was observed across different optimizers.

## Visualizing Results
Training curves and performance comparisons were plotted using **Matplotlib** to assess:
- Accuracy vs. epochs
- Loss vs. epochs
- Convergence behavior of each optimizer

## Conclusion
- **Adam optimizer performed best** at batch sizes of **16 and 32**.
- **SGD showed slower convergence but improved with smaller batch sizes**.
- **RMSProp provided stable training but underperformed compared to Adam**.

These insights assist in optimizing CNN training for different datasets and hardware constraints.

For any contributions feel free to connect on [LinkedIn](https://www.linkedin.com/in/abidul-mohaimin/)
