# Machine Learning Group Assignment: Transfer Learning

The project aims to classify road sign images using a deep-learning convolutional neural network (CNN). By leveraging a transfer learning approach, we utilized the VGG16 model developed by the University of Oxford to improve performance and make training efficient. The goal of the project is to accurately identify road signs from the initial 10 categories [0-9] of the dataset.

## Dataset

The dataset consists of 50,000 images across 43 categories, sourced from Kaggle's. For our project, we trained our model only on the initial 10 categories from the GTSRB—German Traffic Sign Dataset.

For this project, we trained our model on a subset of data, focusing only on the first 10 categories (labels 0-9). Each image was preprocessed to align with the requirements of the VGG16 model.

![Dataset Sample](https://raw.githubusercontent.com/GordonHeg/Machine-Learning-Group-Assignment-Transfer-Learning-/main/images/dataset_sample.jpg)

## Model Architecture

We adopted a transfer learning approach with the VGG16 model as the base network. Key features of our architecture include:

- **Base Model**: VGG16, pre-trained on the ImageNet dataset
- **Custom Layers**: Added a fully connected dense layer with 512 neurons with ReLU activation function for feature extraction
- ***Output Layer***: A softmax layer with 10 neurons to classify the 10 categories
- **Transfer Learning**: Initial convolutional layers of VGG16 were frozen to pre-trained features, and only the added layers were trained

The combination of pre-trained features and custom layers enabled training with limited computational resources.

![Model Architecture Diagram](https://raw.githubusercontent.com/GordonHeg/Machine-Learning-Group-Assignment-Transfer-Learning-/main/images/model_architecture.jpg)
