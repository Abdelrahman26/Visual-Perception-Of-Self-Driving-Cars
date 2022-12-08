# Semantic-Segmentation-For-The-Road

# The Goal
*How to make vechile able to segment all objects of
the scene and classify each pixel to the suitable class*
*****
# Dataset

- *collection of recorded videos of vehicles driven in Germany in
different seasons and different times of the day*
- *it has 2975 training images files and 500 validation image files*
- *The depth of entities in the scene are also included in this dataset* 
*****

# Data Preprocessing 

#### *Each data in the data file is about one combined image, the left half represents the landscape(real image) and the right half represents the ground truth (label) of the landscape*

* *spliting each data file into two parts so the first half has to add to the data
part and the second has to add to the label part*
* *Label Clustering*
> - *because the label image is three channels we have to convert it to one channel
> manner so we can do training as supervised learning because each pixel in the
> landscape image, the corresponding of its in label image will be a number representing
> the number of true class*
> - used **K-means** to solve it
*****

# Dataset Class 
*implemnting a dataset class that will make the operation of storing the data, matched
with deep learning frameworks format*

# Network Architecture
* **Fully Convolutional Neural Networks**
  - The output shape of the network equals the input shape of the
network, we have to use FCNN rather than the normal CNN, because in CNN the
architecture has fully connected layers in the last layersand this idea is not suitable for
this step, where we want the desired output includes **localization**
  - The output has c channels, each pixel in this channel represents the probability of
belonging to this class channel

* **U-Net Architecture** 
  - used a modified version from the U-Net paper
  - The architecture consists of **a contracting** path to capture context and a symmetric
**expanding** path that enables precise localiza-tion.
  - successive layers, where pooling operators are replaced by upsampling
operators.Hence, these layers increase the resolution of the output. In order to localize,
high resolution features from the contracting path are combined with the upsampled
output.A successive convolution layer can then learn to assemble a more precise output
based on this information.

  - In the upsampling part we also have a large number of feature channels, which
allow the network to propagate context information to higher resolution layers.as a
consequence, the expansive path is more or less symmetric to the contracting path,and
yields a u-shaped architecture

  - The Network consists of a contracting path (left side) and an expansive path
(right side). The contracting path follows the typical architecture of a convolutional
network. It consists of the repeated application of two 3x3 convolutions (unpadded
convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling
operation with stride 2 for downsampling. At each downsampling step we double the
number of feature channels. Every step in the expansive path consists of an
upsampling of the feature map followed by a 2x2 convolution (up-convolution) that
halves the number of feature channels, a concatenation with the correspondingly
cropped feature map from the contracting path, and two 3x3 convolutions, each
followed by a ReLU. The cropping is necessary due to the loss of border pixels in every
convolution. At the last layer a 1x1 convolution is used to map each 64-component
feature vector to the desired number of classes. In total the network has 23
convolutional layers.


# Accuracy
## Intersection Over Union
* The Intersection over Union (IoU) metric, is essentially a method to quantify the
percept overlap between the target mask and our prediction output
* The IoU metric measures the number of pixels common between the target and
prediction masks divided by the total number of pixels present across both masks

## Results 
After Training of 240 epochs and using IoU score to evaluate, we
finally have achieved accuracy higher than 98%
