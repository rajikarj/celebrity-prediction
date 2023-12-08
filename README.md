# celebrity-prediction
Data Loading and Preprocessing:

    Loading Images: The code reads images from the specified directory (root_dir). It loops through each subdirectory (representing different celebrities) and reads images from these directories using cv2.imread() after checking that the file format is PNG.
    Image Processing: Each image is converted into an array, resized to a fixed size (128x128 pixels), and appended to the dataset list. Corresponding labels (representing celebrities) are assigned and added to the label list.

Train-Test Split:

    The dataset is divided into training and testing sets using train_test_split() from scikit-learn. Typically, this helps in assessing the model's performance on unseen data.

Model Definition and Training:

    Model Architecture: A Convolutional Neural Network (CNN) is defined using TensorFlow's Keras API.
        The model comprises a series of layers: convolutional layers (Conv2D), max-pooling layers (MaxPooling2D), a flattening layer (Flatten), and dense layers (Dense) with different activation functions.
        The output layer has 5 neurons (assuming 5 classes of celebrities) with a softmax activation function.
    Model Compilation and Training: The model is compiled with an optimizer (adam) and a loss function (sparse_categorical_crossentropy). It is then trained using the training data for a specified number of epochs and a defined batch size.

Model Evaluation:

    The trained model is evaluated on the testing dataset using model.evaluate(). It computes the loss and accuracy metrics on this separate set of data to assess how well the model generalizes to unseen examples.

Model Prediction:

    A function make_prediction() is defined to predict the celebrity from an input image file.
    It preprocesses the image (resizing, normalization), feeds it to the model using model.predict(), retrieves the predicted class, and maps it back to the corresponding celebrity name using the celebrities list.

Prediction Results:

    The make_prediction() function is called for five specific images, and the predicted celebrities for these images are printed.

Accuracy:

    The accuracy of the model, indicating the percentage of correctly predicted labels on the test dataset (86.27%).

Report:

    The code generates a simple report showing the predicted celebrities for specific images along with the overall accuracy.This workflow illustrates a typical process for image classification using a CNN, including data loading, model training, evaluation, prediction, and basic reporting of model accuracy and predictions for specific images. 
