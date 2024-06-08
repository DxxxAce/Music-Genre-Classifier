import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from src.utils.utils import *

DATASET_PATH = "../preprocessing/data.json"  # Path to JSON file storing MFCCs and labels for each processed segment.


def prepare_datasets(test_size, val_size):
    """
    Splits the train, validation and test datasets for the CNN.

    :param test_size: Size of the test dataset.
    :param val_size: Size of the validation dataset.
    :return: Train, validation and test inputs/targets.
    """

    # Load data.
    inputs, targets = load_data(DATASET_PATH)

    # Create the train/test split.
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=test_size)

    # Create the train/validation split.
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(train_inputs,
                                                                            train_targets,
                                                                            test_size=val_size)

    # Reshape the return values.
    train_inputs = train_inputs[..., np.newaxis]
    val_inputs = val_inputs[..., np.newaxis]
    test_inputs = test_inputs[..., np.newaxis]

    return train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets


def build_model(input_shape):
    """
    Builds the architecture of the CNN.

    :param input_shape: Tuple containing the number of rows, columns and channels of an input.
    :return: CNN model, ready to be trained.
    """

    # Create the model.
    model = keras.Sequential()

    # First convolutional layer.
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    # Second convolutional layer.
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    # Third convolutional layer.
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output and feed it into dense layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))

    # Output layer.
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def predict(model, x, y, genres):
    """
    Attempts to classify a given sample from the test set.

    :param model: Trained CNN model.
    :param x: Sample input.
    :param y: Sample target.
    :param genres: List of music genre semantic labels.
    :return: None
    """

    # Make prediction.
    x = x[np.newaxis, ...]
    prediction = model.predict(x)

    # Extract index with max value.
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected label: {}\nPredicted label: {}".format(genres[y], genres[predicted_index[0]]))


if __name__ == "__main__":
    # Create train, validation and test sets.
    train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets = prepare_datasets(0.2, 0.2)

    # Build the CNN.
    input_shape = (train_inputs.shape[1], train_inputs.shape[2], train_inputs.shape[3])
    model = build_model(input_shape)

    # Compile the CNN.
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the CNN.
    history = model.fit(train_inputs,
              train_targets,
              validation_data=(val_inputs, val_targets),
              batch_size=32,
              epochs=80)

    # Evaluate the CNN on the test set.
    test_error, test_accuracy = model.evaluate(test_inputs, test_targets, verbose=1)
    print("\nAccuracy on test set is: {}".format(test_accuracy))

    # Make prediction on a test sample.
    genres = load_genres(DATASET_PATH)
    x, y = test_inputs[100], test_targets[100]
    predict(model, x, y, genres)

    # Plot accuracy and error over the epochs.
    plot_history(history)
