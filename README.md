# Custom_Yolo_Object_Detection
### 1. Introduction

These 3-files are used for training and testing Yolo Model.

### 2. Requirements

Python version: 2 or 3

Packages:

 1.8.0 <= tensorflow < 2.0.1 (theoretically any version that supports tf.data is ok)
opencv-python
tqdm

### 3. Running demos

Please run the below command to test:

python new_test.py --input_image "img.jpg" 


### 4. Training

Use the new_train.py file for training on your own classes and data. The trained checkpoints and weights are also saved for trsting and train_fn()   defines the various training and saving scheme and parameters. 

#### 5 Various Arguments 

The new_args.py file have list of various arguments used while training and testing Yolo model.

### 6. Reference

For more information, please check https://github.com/wizyoung/YOLOv3_TensorFlow 
