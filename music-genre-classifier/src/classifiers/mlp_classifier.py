import time

import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from src.utils.utils import load_features, plot_history, compute_metrics

DATASET_PATH = "../preprocessing/features.csv"  # Path to file storing features and labels for each processed segment.
LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]  # Genre labels.


# noinspection PyShadowingNames
def build_model(input_shape):
    """
    Builds the architecture of the MLP.

    :param input_shape: Tuple containing the number of rows and columns of an input.
    :return: MLP model, ready to be trained.
    """

    model = keras.Sequential([

        # First dense layer.
        keras.layers.Dense(512, activation="relu", input_dim=input_shape,
                           kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.Dropout(0.2),

        # Second dense layer.
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.003)),
        keras.layers.Dropout(0.2),

        # Third dense layer.
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),

        # Fourth dense layer.
        keras.layers.Dense(32, activation="relu",),

        # Output layer.
        keras.layers.Dense(10, activation="softmax"),
    ])

    return model


if __name__ == '__main__':
    # Load data.
    inputs, targets = load_features(DATASET_PATH)

    # Split data into train and test sets.
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)

    # Build neural network architecture.
    input_shape = inputs.shape[1]
    model = build_model(input_shape)

    # Compile neural network.
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    start_time = time.time()

    # Train neural network.
    history = model.fit(train_inputs,
                        train_targets,
                        validation_data=(test_inputs, test_targets),
                        epochs=100,
                        batch_size=32)

    end_time = time.time()

    # Plot accuracy and error over the epochs.
    plot_history(history)

    # Compute model evaluation metrics.
    compute_metrics(model, test_inputs, test_targets, history, (start_time, end_time), LABELS, model_name="MLP")
