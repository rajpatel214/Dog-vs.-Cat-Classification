# Dog vs Cat Classification using CNN

## Project Overview
This deep learning project implements a Convolutional Neural Network (CNN) to classify images of dogs and cats. The dataset used is the **Dogs vs. Cats dataset** from Kaggle. The model is trained to distinguish between dog and cat images using TensorFlow and Keras.

## Dataset
- The dataset contains labeled images of dogs and cats.
- Downloaded from Kaggle using the Kaggle API.
- The dataset is split into **training** and **validation** sets.
- Images are preprocessed and normalized before feeding into the model.

## Technologies Used
- **Python**
- **TensorFlow/Keras** for deep learning
- **OpenCV** for image processing
- **Matplotlib** for visualizations

## Project Workflow
### 1. Data Acquisition
- Download dataset using Kaggle API.
- Extract and organize images into `train` and `test` directories.

### 2. Data Preprocessing
- Convert images into tensors using `image_dataset_from_directory`.
- Normalize pixel values between 0 and 1.

### 3. Model Architecture
A CNN model is built using the following layers:
- **Convolutional layers** with ReLU activation
- **Batch Normalization** to improve stability
- **MaxPooling layers** to reduce spatial dimensions
- **Flatten layer** to convert 2D feature maps into a 1D vector
- **Fully connected dense layers** with dropout for regularization
- **Output layer** with a sigmoid activation function for binary classification

### 4. Model Compilation and Training
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metrics: Accuracy
- Trained for **10 epochs**

### 5. Model Evaluation
- Plot training vs validation accuracy and loss.
- Test the model on sample images of dogs and cats.

## Results
- Training and validation accuracy curves are plotted to analyze model performance.
- The model is tested on new images to verify classification performance.

## Future Improvements
- Data augmentation to improve generalization.
- Hyperparameter tuning for better accuracy.
- Experimenting with deeper architectures like ResNet or VGG.

## How to Run the Project
1. Clone the repository.
2. Download the dataset from Kaggle.
3. Install required dependencies.
4. Run the training script.
5. Test the model on new images.

## Conclusion
This project demonstrates the implementation of a CNN model for image classification. It serves as a foundation for further improvements in deep learning-based image classification tasks.

