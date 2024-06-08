import json

import matplotlib.pyplot as plt
import numpy as np

__name__ = "utils"
__doc__ = "Utility library for loading data and plotting statistics of the models' performance over the epochs."
__all__ = ["load_data", "load_genres", "plot_history"]


def load_data(dataset_path):
    """
    Loads training dataset from JSON file.

    :param dataset_path: Path to JSON file containing data.
    :return: Tuple consisting of the inputs and targets as arrays.
    """

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # Convert lists into numpy arrays.
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def load_genres(dataset_path):
    """
    Loads semantic labels of the music genres from JSON file.

    :param dataset_path: Path to JSON file containing data.
    :return: List of music genres.
    """

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # Save music genre list.
    genres = data["mapping"]

    return genres


def plot_history(history):
    """
    Generates plots showing evolution of the accuracy and error in the train/test data over the epochs.

    :param history: Record of training, as well as validation loss values and metrics values at successive epochs.
    :return: None.
    """

    fig, axis = plt.subplots(2)

    # Create accuracy subplot.
    axis[0].plot(history.history["accuracy"], label="train accuracy")
    axis[0].plot(history.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")

    # Create error subplot.
    axis[1].plot(history.history["loss"], label="train error")
    axis[1].plot(history.history["val_loss"], label="test error")
    axis[1].set_xlabel("Epoch")
    axis[1].set_ylabel("Error")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error eval")

    # Display the subplots.
    plt.show()
