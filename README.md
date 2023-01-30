# YOLOv3 Aluminum Surface Defect Detection
Using YOLOv3 to detect surface defects of Aluminum.
# Overview
In this project, I use YOLOv3, an object detection algorithm based on CNN (Convolutional Neural Network), to detect surface defects of Aluminum. After estimated the result of the original algorithm, to reach a better performance, I do a research on how the parameters affect the result of the detection and design a set of strategies to reduce the class imbalance. After the optimization, the evaluation criterion mAP (Mean Average Precision) reaches 68.3%, which is 8.6% higher than the original one, having better performance of the detection in different scales and different classes.
# Dataset
The dataset used in this project is provided by Alibaba Cloud TIANCHI. The dataset contains more than 10,000 Aluminum profile pictures including 3005 pictures labeled with bounding box information and class information of the defects. The defects are classified into 10 types, such as non-conductive area (BuDaoDian), dirty spot (ZangDian), bubble (QiPao) and so on. To realize the goal of object detection, I use the 3005 samples to set up the training dataset, validation dataset and the test dataset.<br><br>
### 10 kinds of defects in the dataset:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/01.png" width="800px" /><br>
### Label of a sample:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/05.png" width="800px" /><br>
### Link of the dataset: https://tianchi.aliyun.com/dataset/140666<br>
# YOLOv3 Algorithm
YOLOv3 is a single-stage object detection algorithm based on deep learning. As the backbone of YOLOv3, Darknet53 is a CNN (Convolutional Neural Network) consisting of 23 residual blocks, and can output three feature maps at different scales. In the training process, YOLOv3 take a square picture as the input, calculate the feature maps through Darknet53, then calculate the loss function and use the SGD (Stochastic Gradient Descent) to fit the output and label. When making an inference, the backbone can provide raw predicted information about the bounding boxes with dimension priors and location as well as class. Then through postprocessing like setting up a threshold of confidence score and using NMS (Non-Maximum Suppression) to select the best bounding box from the overlapping boxes, the final result of the prediction can be obtained.<br><br>
### Structure of Darknet53:<br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/08.png" width="800px" /><br>
### Loss Function:<br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/03.png" width="600px" /><br>
# Data Augmentation
Statistics suggest that the quantity of different class of samples are quite imbalanced, which will reduce the accuracy and robustness of the model. To ease the class imbalance, data augmentation is performed to increase the amount of each class of samples to 500. The process includes geometric transformations (random flipping, random scale) and color space augmentations (random brightness, contrast, saturation and hue), making the model less sensitive to the changes of illumination and scale between different pictures.<br><br>
### Data Augmentation Strategy:<br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/18.png" width="800px" /><br>
### Data Augmentation Program:<br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/19.png" width="800px" /><br> 
### Data Augmentation Visualization:<br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/20.png" width="600px" /><br>
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
### Precision Curves and Recall Curves:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/09.png" width="700px" /><br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/10.png" width="350px" />
### Loss Curves:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/11.png" width="700px" /><br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/12.png" width="350px" />
### mAP(AP50):
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/13.png" width="700px" /><br>
### mAP(AP@50:5:95):
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/14.png" width="700px" /><br>
### Detection Visualization:
#### Good:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/16.png" width="700px" /><br>
#### Bad:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/17.png" width="700px" /><br>
## Optimized Parameters:
```
Size of Input Image: 640 * 640
Learning Rate: Warm-up Strategy, Max Lr = 0.01
Epoch: 30
Batch Size: 16
Mosaic: Applied
Data Augmentation: Yes
```
### Precision Curves and Recall Curves:<br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/21.png" width="700px" /><br>
### Loss Curve:<br>
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/22.png" width="400px" /><br>
### mAP(AP50):
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/23.png" width="700px" /><br>
### mAP(AP@50:5:95):
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/24.png" width="700px" /><br>
### Confusion Matrix Comparison:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/25.png" width="800px" /><br>
### F1 Curves Comparison:
<img src="https://github.com/yangyilin52/YOLOv3-Aluminum-Surface-Defect-Detection/blob/main/imgs/26.png" width="800px" /><br>
