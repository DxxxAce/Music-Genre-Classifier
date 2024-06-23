import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

__name__ = "utils"
__doc__ = "Utility library for loading data, plotting statistics and computing evaluation metrics."
__all__ = ["load_mfcc", "load_features", "load_genres", "plot_history", "plot_confusion_matrix", "compute_metrics"]

from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_mfcc(data_path):
    """
    Loads training dataset from JSON file.

    :param data_path: Path to JSON file containing data.
    :return: Tuple consisting of the inputs and targets as arrays.
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Convert lists into numpy arrays.
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def load_features(data_path):
    """
    Loads features from CSV file.

    :param data_path: Path to CSV file containing data.
    :return: Tuple consisting of the inputs and targets as arrays.
    """

    df = pd.read_csv(data_path)
    df = df.drop('filename', axis=1)

    inputs = df.iloc[:, :-1]
    targets = df.iloc[:, -1]

    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)

    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(targets)

    return inputs, targets


def load_genres(data_path):
    """
    Loads semantic labels of the music genres from JSON file.

    :param data_path: Path to JSON file containing data.
    :return: List of music genres.
    """

    with open(data_path, "r") as fp:
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


def plot_confusion_matrix(cm, class_names, model_name):
    """
    Plots the confusion matrix for the trained neural network model.

    :param cm: The model's precomputed confusion matrix.
    :param class_names: The class names.
    :param model_name: Name of the neural network model.
    :return: None.
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for the {model_name}')
    plt.show()


def compute_metrics(model, test_inputs, test_targets, history, timestamps, labels, model_name="model"):
    """
    Computes evaluation metrics for the trained neural network model.

    :param model: The trained neural network model.
    :param test_inputs: Test set inputs.
    :param test_targets: Test set expected outputs.
    :param history: Record of training, as well as validation loss values and metrics values at successive epochs.
    :param timestamps: Tuple containing the starting and ending times of the training process.
    :param labels: The class names.
    :param model_name: Name of the trained neural network model.
    :return: None.
    """

    # Retrieve training and validation accuracy.
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    # Predict on the test set.
    predicted_targets_probs = model.predict(test_inputs)
    predicted_targets = np.argmax(predicted_targets_probs, axis=1)

    # Calculate test accuracy
    test_accuracy = accuracy_score(test_targets, predicted_targets)

    # Print accuracies
    print()
    print('Training accuracy:', train_accuracy)
    print('Validation accuracy:', val_accuracy)
    print('Test accuracy:', test_accuracy)

    # Compute Precision, Recall, and F1-Score.
    precision = precision_score(test_targets, predicted_targets, average=None)
    recall = recall_score(test_targets, predicted_targets, average=None)
    f1 = f1_score(test_targets, predicted_targets, average=None)

    print()
    print("Precision (per class):", precision)
    print("Recall (per class):", recall)
    print("F1-Score (per class):", f1)

    # Compute aggregated metrics.
    precision_macro = precision_score(test_targets, predicted_targets, average="macro")
    recall_macro = recall_score(test_targets, predicted_targets, average="macro")
    f1_macro = f1_score(test_targets, predicted_targets, average="macro")

    print()
    print("Macro Precision:", precision_macro)
    print("Macro Recall:", recall_macro)
    print("Macro F1-Score:", f1_macro)

    # Generate the confusion matrix.
    cm = confusion_matrix(test_targets, predicted_targets)

    print()
    print("Confusion matrix:\n", cm)

    plot_confusion_matrix(cm, labels, model_name)

    # Calculate the training duration.
    training_duration = timestamps[1] - timestamps[0]

    print()
    print(f"Training Duration: {training_duration} seconds")
