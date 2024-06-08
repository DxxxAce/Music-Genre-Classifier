import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from src.utils.utils import load_data, plot_history

DATASET_PATH = "../preprocessing/data.json"  # Path to JSON file storing MFCCs and labels for each processed segment.


def build_model(input_shape):
    """
    Builds the architecture of the MLP.

    :param input_shape: Tuple containing the number of rows and columns of an input.
    :return: MLP model, ready to be trained.
    """

    model = keras.Sequential([
        # Input layer.
        keras.layers.Flatten(input_shape=input_shape),

        # First hidden layer.
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Second hidden layer.
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Third hidden layer.
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Output layer.
        keras.layers.Dense(10, activation="softmax"),
    ])

    return model


if __name__ == '__main__':
    # Load data.
    inputs, targets = load_data(DATASET_PATH)

    # Split data into train and test sets.
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)

    # Build neural network architecture.
    input_shape = (inputs.shape[1], inputs.shape[2])
    model = build_model(input_shape)

    # Compile neural network.
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # Train neural network.
    history = model.fit(train_inputs,
                        train_targets,
                        validation_data=(test_inputs, test_targets),
                        epochs=150,
                        batch_size=32)

    # Plot accuracy and error over the epochs.
    plot_history(history)
