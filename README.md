# YOLOv3 Aluminum Surface Defect Detection
Using YOLOv3 to detect surface defects of Aluminum.
# Overview
In this project, I use YOLOv3, an object detection algorithm based on CNN (Convolutional Neural Network), to detect surface defects of Aluminum. After estimated the result of the original algorithm, to reach a better performance, I do a research on how the parameters affect the result of the detection and design a set of strategies to reduce the class imbalance. After the optimization, the evaluation criterion mAP (Mean Average Precision) reaches 68.3%, which is 8.6% higher than the original one, having better performance of the detection in different scales and different classes.
# Dataset
The dataset used in this project is provided by Alibaba Cloud TIANCHI. The dataset contains more than 10,000 Aluminum profile pictures including 3005 pictures labeled with bounding box information and class information of the defects. The defects are classified into 10 types, such as non-conductive area (BuDaoDian), dirty spot (ZangDian), bubble (QiPao) and so on. To realize the goal of object detection, I use the 3005 samples to set up the training dataset, validation dataset and the test dataset.
# YOLOv3 Algorithm
YOLOv3 is a single-stage object detection algorithm based on deep learning. As the backbone of YOLOv3, Darknet53 is a CNN (Convolutional Neural Network) consisting of 23 residual blocks, and can output three feature maps at different scales. In the training process, YOLOv3 take a square picture as the input, calculate the feature maps through Darknet53, then calculate the loss function and use the SGD (Stochastic Gradient Descent) to fit the output and label. When making an inference, the backbone can provide raw predicted information about the bounding boxes with dimension priors and location as well as class. Then through postprocessing like setting up a threshold of confidence score and using NMS (Non-Maximum Suppression) to select the best bounding box from the overlapping boxes, the final result of the prediction can be obtained.
# Data Augmentation
Statistics suggest that the quantity of different class of samples are quite imbalanced, which will reduce the accuracy and robustness of the model. To ease the class imbalance, data augmentation is performed to increase the amount of each class of samples to 500. The process includes geometric transformations (random flipping, random scale) and color space augmentations (random brightness, contrast, saturation and hue), making the model less sensitive to the changes of illumination and scale between different pictures.
# Evaluation
## Original Parameters:
```
Size of Input Image: 320 * 320
Learning Rate: Warm-up Strategy, Max Lr = 0.01
Epoch: 30
Batch Size: 16
Mosaic: Yes
Data Augmentation: No
```
## Optimized Parameters:
```
Size of Input Image: 640 * 640
Learning Rate: Warm-up Strategy, Max Lr = 0.01
Epoch: 30
Batch Size: 16
Mosaic: Applied
Data Augmentation: Yes
```
