Machine Learning Mini Project Report
Problem Description:
The objective of this mini-project is to develop a machine learning solution for image classification using the EMNIST Digits dataset. The dataset consists of handwritten digit images converted to a 28x28 pixel format, with 10 balanced classes representing digits from 0 to 9.

Dataset:
The EMNIST Digits dataset contains 280,000 handwritten digit images with 10 balanced classes. Each row in the dataset represents an image, with the first column containing the label value (digit) and the remaining 784 columns containing pixel values for the images.

Model Architecture:
For this project, a Convolutional Neural Network (CNN) architecture has been chosen due to its effectiveness in handling image data. The CNN model consists of convolutional layers followed by max-pooling layers to extract features from the images. After flattening the feature maps, dense layers are used for classification.

Data Preparation:

The dataset was loaded into a DataFrame using the pandas library.
Labels and features were separated, with pixel values normalized between 0 and 1 using MinMaxScaler.
Feature data was reshaped to represent images in a 28x28 pixel format with a single channel.
The dataset was split into training and testing sets using train_test_split.
Model Training:

The CNN model was defined using the Sequential API from Keras.
The model was compiled with the Adam optimizer and sparse categorical crossentropy loss function.
Training was performed on the training set for 10 epochs with a batch size of 32.
Model performance was evaluated on the test set using accuracy as the evaluation metric.
Results:

After training the CNN model, it achieved an accuracy of 99.36% on the test set.
The model successfully predicted 47,694 out of 48,000 samples in the test set.
Conclusion:
In conclusion, the developed CNN model demonstrates excellent performance in classifying handwritten digits from the EMNIST Digits dataset, achieving a high accuracy rate. This mini-project provides valuable hands-on experience in developing machine learning solutions for image classification tasks.

Submitted By: Amin Aouina






