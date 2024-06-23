import time

import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.utils import *

DATASET_PATH = "../preprocessing/features.csv"  # Path to file storing features and labels for each processed segment.
LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]  # Genre labels.


def prepare_datasets(test_size, val_size):
    """
    Splits the train, validation and test datasets for the RNN-LSTM.

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

    return train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets


# noinspection PyShadowingNames
def build_model(input_shape):
    """
    Builds the architecture of the RNN-LSTM.

    :param input_shape: Tuple containing the number of rows and columns of an input.
    :return: RNN-LSTM model, ready to be trained.
    """

    # Create the model.
    model = keras.Sequential()

    # First LSTM layer.
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    # Second LSTM layer.
    model.add(keras.layers.LSTM(64, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    # Third LSTM layer.
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    # Dense layer.
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.5))

    # Flatten the output and feed it into dense layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # Output layer.
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def predict(model, x, y, genres):
    """
    Attempts to classify a given sample from the test set.

    :param model: Trained RNN-LSTM model.
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

    # Build the RNN-LSTM.
    input_shape = (train_inputs.shape[1], 1)
    model = build_model(input_shape)

    # Compile the RNN-LSTM.
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    start_time = time.time()

    # Train the RNN-LSTM.
    history = model.fit(train_inputs,
                        train_targets,
                        validation_data=(val_inputs, val_targets),
                        batch_size=32,
                        epochs=100)

    end_time = time.time()

    # Evaluate the RNN-LSTM on the test set.
    test_error, test_accuracy = model.evaluate(test_inputs, test_targets, verbose=1)
    print("\nAccuracy on test set is: {}".format(test_accuracy))

    # Make prediction on a test sample.
    x, y = test_inputs[50], test_targets[50]
    predict(model, x, y, LABELS)

    # Plot accuracy and error over the epochs.
    plot_history(history)

    # Compute model evaluation metrics.
    compute_metrics(model, test_inputs, test_targets, history, (start_time, end_time), LABELS, model_name="RNN LSTM")
