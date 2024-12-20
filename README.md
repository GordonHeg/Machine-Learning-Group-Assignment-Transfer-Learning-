# Machine Learning Group Assignment: Transfer Learning

The project aims to classify road sign images using a deep-learning convolutional neural network (CNN). By leveraging a transfer learning approach, we utilized the VGG16 model developed by the University of Oxford to improve performance and make training efficient. The goal of the project is to accurately identify road signs from the initial 10 categories [0-9] of the dataset.

## Dataset

The dataset consists more than 50,000 images of road signs across 43 categories, sourced from Kaggle's. For our project, we trained our model only on the initial 10 categories from the GTSRB—German Traffic Sign Dataset.

For this project, we trained our model on a subset of data, focusing only on the first 10 categories (labels 0-9). Each image was preprocessed to align with the requirements of the VGG16 model.

![Figure 1: 43 categories of road sign available in dataset](https://raw.githubusercontent.com/GordonHeg/Machine-Learning-Group-Assignment-Transfer-Learning-/main/images/dataset_sample.jpg)
*Figure 1: Road signs 43 categories available in dataset*


## Model Architecture

We adopted a transfer learning approach with the VGG16 model as the base network. Key features of our architecture include:

- **Base Model**: VGG16, pre-trained on the ImageNet dataset
- **Custom Layers**: Added a fully connected dense layer with 512 neurons with ReLU activation function for feature extraction
- ***Output Layer***: A softmax layer with 10 neurons to classify the 10 categories
- **Transfer Learning**: Initial convolutional layers of VGG16 were frozen to pre-trained features, and only the added layers were trained

The combination of pre-trained features and custom layers enabled training with limited computational resources.

![Figure 2: VGG16 Model Architecture Diagram](https://raw.githubusercontent.com/GordonHeg/Machine-Learning-Group-Assignment-Transfer-Learning-/main/images/model_architecture.jpg)
*Figure 2: VGG16 Model Architecture Diagram*

**Transfer learning** was chosen to efficiently classify road sign images with limited data of 14000 images of 10 categories. By using the pre-trained VGG16 model, we leveraged its ability to extract features like edges and shapes from the ImageNet dataset. 

The transfer learning approach helps in reducing training time, improving accuracy and minimising the risk of overfitting, The Fine-tunning Fully connected dense layer allowed us to adapt the model specifically for the road sign dataset while retaining the pre-trained knowledge ensuring efficient use of data and computation resources.

## Results
- Train AUC: **0.9932**, Val AUC: **0.9944**, Test AUC: **0.9621** indicate strong overall model performance.  
- Model achieved **77% accuracy** with a macro-averaged **AUC of 0.83**.  
- Precision, recall, and F1-scores show varying performance across classes, with some performing poorly.  
- Class **3 and 5** exhibit low F1-scores (**0.61** and **0.68**, respectively), highlighting room for improvement.  
- The model is effective but requires optimization to enhance performance for underperforming classes.  
## Conclusion

- Demonstrated the effectiveness of transfer learning using a pre-trained VGG16 model.
- Achieved high performance with an AUC of 0.96 on the test set, reducing training effort.
- Struggled with Class 0 and Class 3 despite data augmentation due to insufficient image availability and limited new feature learning.
- Transfer learning proves to be a viable approach for building accurate and efficient machine learning models.

