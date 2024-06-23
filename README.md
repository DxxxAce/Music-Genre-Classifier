# MusicGenreClassifier

This repository contains the code and resources for classifying music genres using three different deep learning models: a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN), and a Recurrent Neural Network with Long Short-Term Memory (RNN LSTM).

## Project Overview

This project explores the classification of music genres using deep learning techniques. We implemented and compared the performance of three models: MLP, CNN, and RNN LSTM, on the GTZAN dataset. The goal is to identify which model performs best in classifying music into 10 distinct genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, and Rock.

## Dataset

We used the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), a widely used benchmark dataset for music genre classification. The dataset includes 1000 audio tracks each 30 seconds long. The tracks are categorized into 10 genres, with 100 tracks per genre.

## Model Architectures

### Multilayer Perceptron (MLP)
- A simple feedforward neural network.
- Utilizes dense layers with ReLU activation.
- Regularization techniques like L2 regularization and Dropout are used to prevent overfitting.

### Convolutional Neural Network (CNN)
- Consists of multiple Conv1D layers to capture temporal patterns in audio signals.
- Employs MaxPooling, BatchNormalization, and Dropout layers to improve model performance and generalization.

### Recurrent Neural Network with LSTM (RNN LSTM)
- Utilizes LSTM layers to capture long-term dependencies in sequential audio data.
- Incorporates BatchNormalization and Dropout for stability and regularization.

### Prerequisites

- Python 3.10
- Libraries: NumPy, TensorFlow, Keras, Librosa, Scikit-learn, Matplotlib, Pandas, Seaborn

### Results
The performance of each model on the test set is summarized as follows:

1. MLP:

  * Test Accuracy: 87.19%
  * Macro F1-Score: 87.19%
  * Training Duration: 226.57 seconds

2. CNN:

  * Test Accuracy: 90.19%
  * Macro F1-Score: 90.17%
  * Training Duration: 269.06 seconds

3. RNN LSTM:

  * Test Accuracy: 66.73%
  * Macro F1-Score: 66.55%
  * Training Duration: 371.00 seconds

### Acknowledgments
  * The GTZAN dataset is used with acknowledgment to the creators and contributors.
  * Thanks to the developers of the libraries used in this project: TensorFlow, Keras, Librosa, Scikit-Learn, Matplotlib, Pandas, and Seaborn
