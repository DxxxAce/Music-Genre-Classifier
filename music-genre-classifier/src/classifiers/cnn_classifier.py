import time

import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from src.utils.utils import *

DATASET_PATH = "../preprocessing/features.csv"  # Path to file storing features and labels for each processed segment.
LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]  # Genre labels.


def prepare_datasets(test_size, val_size):
    """
    Splits the train, validation and test datasets for the CNN.

    :param test_size: Size of the test dataset.
    :param val_size: Size of the validation dataset.
    :return: Train, validation and test inputs/targets.
    """

    # Load data.
    inputs, targets = load_features(DATASET_PATH)

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


# noinspection PyShadowingNames
def build_model(input_shape):
    """
    Builds the architecture of the CNN.

    :param input_shape: Tuple containing the number of rows, columns and channels of an input.
    :return: CNN model, ready to be trained.
    """

    # Create the model.
    model = keras.Sequential()

    # First convolutional layer.
    model.add(keras.layers.Conv1D(64, 3, activation="relu", input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.MaxPool1D(2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Second convolutional layer.
    model.add(keras.layers.Conv1D(128, 3, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.MaxPool1D(2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Third convolutional layer.
    model.add(keras.layers.Conv1D(256, 5, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.MaxPool1D(2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Fourth convolutional layer.
    model.add(keras.layers.Conv1D(512, 5, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.MaxPool1D(2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    # Flatten the output and feed it into dense layers.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5))

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
    train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets = prepare_datasets(0.25, 0.2)

    # Build the CNN.
    input_shape = (train_inputs.shape[1], train_inputs.shape[2])
    model = build_model(input_shape)

    # Compile the CNN.
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    start_time = time.time()

    # Train the CNN.
    history = model.fit(train_inputs,
                        train_targets,
                        validation_data=(val_inputs, val_targets),
                        batch_size=32,
                        epochs=100)

    end_time = time.time()

    # Evaluate the CNN on the test set.
    test_error, test_accuracy = model.evaluate(test_inputs, test_targets, verbose=1)
    print("\nAccuracy on test set is: {}".format(test_accuracy))

    # Plot accuracy and error over the epochs.
    plot_history(history)

    # Compute model evaluation metrics.
    compute_metrics(model, test_inputs, test_targets, history, (start_time, end_time), LABELS, model_name="CNN")
