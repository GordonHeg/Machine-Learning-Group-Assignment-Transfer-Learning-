# Machine Learning Group Assignment: Transfer Learning

The project aims to classify road signs images using deep learning convounitional neural network (CNN).By leveraging a transfer learning appoach, we utilized VGG16 model devloped by University of Oxford to imporove performance and training efficient. The goal of project is to accurately identify road signs from the intial 10 categories [0-9] of the dataset. 

The dataset is a collection of 50tsd images which has 43 categories and collected from Kaggle. 



## Dataset
The dataset consist of 50,000 images across 43 categories, sourced from Kaggle's 
For our project we trained our model only for initial 10 categories from GTSRB - German Traffic Sign Dataset.

For this project we trained our model on a suset of data, focusing only on the first 10 categories (labels 0-9). Each image was preprocessed to align with the requirements of the vgg16 model.


## Model Architecture

We adopted a transfer learning approach with vgg16 model as the base network. Key features of our architectures include:

-  Base Model: VGG16, pre-trained on the ImageNet dataset
-  Custom Layers: Added a fully connected dense layer with 512 neurons with ReLU activation function for feature extraction
-  Output Layer: A softmax layer with 10 neurons to classify the 10 categories.
-  Transfer Learning: Intitial convolutional layers of VGG16 were frozen to pre-trained features, and only the added layers were trained

The combination of pre-trained features and custom layers enabled with limit computational resources.
