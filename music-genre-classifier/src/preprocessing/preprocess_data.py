import json
import math
import os

import librosa

DATASET_PATH = "../../dataset"  # Path to audio files dataset.
JSON_PATH = "data.json"  # Path to JSON file storing MFCCs and labels for processed segments.

SAMPLE_RATE = 22050                             # Number of samples per second
DURATION = 30                                   # Track length, measured in seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION      # Total number of samples in a track.


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """
    Extracts the specified number of MFCCs from each audio file in given training dataset.

    :param dataset_path: Path to the training dataset directory.
    :param json_path: Path to JSON file where MFCCs will be stored.
    :param n_mfcc: Number of MFCCs to be extracted.
    :param n_fft: Length of the FFT window.
    :param hop_length: Number of samples between successive frames of the audio signal.
    :param num_segments: Number of segments to split track into.
    :return: None.
    """

    # Dictionary for storing the dataset.
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }

    # Total number of samples in one of the track segments.
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    # Expected number of MFCC vectors per segment.
    mfcc_per_segment = math.ceil(samples_per_segment / hop_length)

    # Loop through all genres
    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        # Ensure we are not at root level.
        if dir_path is not dataset_path:
            # Save semantic label.
            dir_components = dir_path.split("\\")
            semantic_label = dir_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {} tracks...\n".format(semantic_label))

            # Process files for genre.
            for f in file_names:
                # Load audio file.
                file_path = os.path.join(dir_path, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Process segments, extracting MFCCs and storing data.
                for s in range(num_segments):
                    # Starting and ending points of the current track segment.
                    sample_start = samples_per_segment * s
                    sample_end = sample_start + samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[sample_start:sample_end],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Store MFCC for segment if it has the expected length.
                    if len(mfcc) == mfcc_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment: {}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
