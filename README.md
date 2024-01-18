# breast_cancer_prediction-by-using-neural-networks
Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset is loaded using scikit-learn's load_breast_cancer function. The data is then preprocessed, and features are standardized before training the neural network.

Neural Network Architecture

The neural network consists of an input layer, one hidden layer with 20 neurons and ReLU activation, and an output layer with 2 neurons and a sigmoid activation function. The model is compiled using the Adam optimizer and sparse categorical crossentropy loss.

Training and Evaluation

The dataset is split into training and testing sets using the train_test_split function. The data is standardized using the StandardScaler. The model is trained for 10 epochs with a validation split of 10%. After training, the model is evaluated on the test set, and accuracy, as well as a classification report and confusion matrix, are printed.

Dependencies

NumPy
scikit-learn
Matplotlib
Pandas
TensorFlow
