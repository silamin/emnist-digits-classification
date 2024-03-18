# emnist-digits-classification
 
This repository contains a Python script for classifying handwritten digits using the EMNIST dataset, a variation of the MNIST dataset with additional classes and variations. It utilizes convolutional neural networks (CNNs) to achieve accurate digit recognition, providing insights into image classification and deep learning techniques.

Dataset
The EMNIST dataset consists of handwritten character digits, containing various splits such as EMNIST ByClass, EMNIST ByMerge, EMNIST Balanced, EMNIST Letters, EMNIST Digits, and EMNIST MNIST. In this project, we focus on the EMNIST Digits split, which includes 280,000 handwritten digits distributed across 10 balanced classes.

About the Model:
The CNN model employed in this project consists of convolutional layers to extract relevant features from the input images. Max-pooling layers are used for spatial downsampling, reducing the computational complexity of the model. The flattened output is passed through fully connected layers for classification. The model is trained using the Adam optimizer and sparse categorical crossentropy loss function.

Results
After training the model, it is evaluated on a separate test set to measure its performance. The accuracy achieved on the test set reflects the model's ability to correctly classify handwritten digits

