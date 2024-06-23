import librosa
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale, scale

__name__ = "feature_extraction"
__doc__ = "Utility library for extracting specific audio features from a given signal."
__all__ = ["extract_waveform", "extract_chroma_stft", "extract_rms", "extract_spectral_centroid",
           "extract_spectral_bandwidth", "extract_spectral_rolloff", "extract_zero_crossing_rate",
           "extract_harmony", "extract_tempo", "extract_mfcc"]


TEST_TRACK = "../../dataset/rock/rock.00000.wav"        # Path to sample audio track.
SAMPLE_RATE = 22050                                     # Number of samples per second.


def extract_waveform(signal, sr):
    """
    Plots a waveform visualization of the input signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :return:
    """

    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(signal, sr=sr)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Sample track")
    plt.xlim([0, 30])
    plt.show()


def extract_chroma_stft(signal, sr, visualize=False):
    """
    Extracts the chroma STFT feature of the audio signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :param visualize: Should plot the feature's visual representation.
    :return:
    """

    # Extract the Chroma STFSs.
    chroma_stft = librosa.feature.chroma_stft(y=signal, sr=sr)

    if visualize:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(chroma_stft, x_axis="time", y_axis="chroma", sr=sr)
        plt.colorbar()
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch Class")
        plt.title("Chroma STFT")
        plt.show()

    return chroma_stft


def extract_rms(signal):
    """
    Extracts the RMS feature of the audio signal.

    :param signal: Audio signal.
    :return:
    """

    # Extract the RMSs.
    rms = librosa.feature.rms(y=signal)

    return rms.mean()


def extract_spectral_centroid(signal, sr, visualize=False):
    """
    Extracts the spectral centroid feature of the audio signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :param visualize: Should plot the feature's visual representation.
    :return:
    """

    # Extract the spectral centroids.
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)

    if visualize:
        # Computing the time variable for visualization.
        frames = range(len(spectral_centroids[0]))
        t = librosa.frames_to_time(frames=frames, sr=sr)

        # Plotting the Spectral Centroid along the waveform.
        plt.figure(figsize=(16, 6))
        ax = plt.axes()
        ax.set_facecolor("#202020")  # Set a dark background color.

        # Waveform plot
        librosa.display.waveshow(y=signal, sr=sr, alpha=0.6, color="#1DB954", linewidth=1.5, label="Waveform")

        # Spectral centroids plot.
        plt.plot(t, minmax_scale(spectral_centroids[0]), color="#FFC300", linewidth=2, label="Normalized Spectral Centroids")

        # Enhancing the plot.
        plt.title("Waveform and Normalized Spectral Centroids", fontsize=16, fontweight="bold", color="white")
        plt.xlabel("Time (seconds)", fontsize=14, color="white")
        plt.ylabel("Normalized Amplitude / Frequency", fontsize=14, color="white")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xticks(fontsize=12, color="white")
        plt.yticks(fontsize=12, color="white")

        plt.show()

    return spectral_centroids.mean()


def extract_spectral_bandwidth(signal, sr):
    """
    Extracts the spectral bandwidth feature of the audio signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :return:
    """

    # Extract the spectral bandwidths.
    spectral_bandwidths = librosa.feature.spectral_bandwidth(y=signal, sr=sr)

    return spectral_bandwidths.mean()


def extract_spectral_rolloff(signal, sr, visualize=False):
    """
    Extracts the spectral rolloff feature of the audio signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :param visualize: Should plot the feature's visual representation.
    :return:
    """

    # Extract the Spectral Rolloff.
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)

    if visualize:
        # Computing the time variable for visualization.
        frames = range(len(spectral_rolloff[0]))
        t = librosa.frames_to_time(frames, sr=sr)

        # Plotting the Spectral Rolloff along the waveform.
        plt.figure(figsize=(16, 6))
        ax = plt.axes()
        ax.set_facecolor("#202020")  # Set a dark background color.

        # Waveform plot
        librosa.display.waveshow(y=signal, sr=sr, alpha=0.6, color="#1DB954", linewidth=1.5, label="Waveform")

        # Spectral Rolloff plot.
        plt.plot(t, minmax_scale(spectral_rolloff[0]), color="#FFC300", linewidth=2, label="Normalized Spectral Rolloff")

        # Enhancing the plot.
        plt.title("Waveform and Normalized Spectral Rolloff", fontsize=16, fontweight="bold", color="white")
        plt.xlabel("Time (seconds)", fontsize=14, color="white")
        plt.ylabel("Normalized Amplitude / Frequency", fontsize=14, color="white")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xticks(fontsize=12, color="white")
        plt.yticks(fontsize=12, color="white")

        plt.show()

    return spectral_rolloff.mean()


def extract_zero_crossing_rate(signal):
    """
    Extracts the zero-crossing rate feature of the audio signal.

    :param signal: Audio signal.
    :return:
    """

    # Extract the zero-crossing rate.
    zero_crossing_rates = librosa.feature.zero_crossing_rate(y=signal)

    return zero_crossing_rates.mean()


def extract_harmony(signal, sr, visualize=False):
    """
    Extracts the harmony feature of the audio signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :param visualize: Should plot the feature's visual representation.
    :return:
    """

    # Extract the harmony.
    chroma_cens = librosa.feature.chroma_cens(y=signal, sr=sr)

    if visualize:
        # Computing the time variable for visualization.
        frames = range(len(chroma_cens[0]))
        t = librosa.frames_to_time(frames, sr=sr)

        # Plotting the chroma CENS along the waveform.
        plt.figure(figsize=(16, 6))
        ax = plt.axes()
        ax.set_facecolor("#202020")  # Set a dark background color.

        # Waveform plot
        librosa.display.waveshow(y=signal, sr=sr, alpha=0.6, color="#1DB954", linewidth=1.5, label="Waveform")

        # Spectral centroids plot.
        plt.plot(t, minmax_scale(chroma_cens[0]), color="#FFC300", linewidth=2, label="Normalized Chroma CENS")

        # Enhancing the plot.
        plt.title("Waveform and Normalized Chroma CENS", fontsize=16, fontweight="bold", color="white")
        plt.xlabel("Time (seconds)", fontsize=14, color="white")
        plt.ylabel("Normalized Amplitude / Frequency", fontsize=14, color="white")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xticks(fontsize=12, color="white")
        plt.yticks(fontsize=12, color="white")

        plt.show()

    return chroma_cens.mean()


def extract_tempo(signal, sr):
    """
    Extracts the tempo feature of the audio signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :return:
    """

    # Extract the tempo.
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)

    return tempo.mean()


def extract_mfcc(signal, sr, visualize=False):
    """
    Extracts the MFCCs feature of the audio signal.

    :param signal: Audio signal.
    :param sr: Sample rate.
    :param visualize: Should plot the feature's visual representation.
    :return:
    """

    # Extract the MFCCs.
    mfcc = librosa.feature.mfcc(y=signal, sr=sr)

    if visualize:
        # Apply Feature Scaling
        mfcc = scale(mfcc, axis=1)

        # Plot MFCCs
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis="time", cmap="coolwarm")
        plt.colorbar(format="%+2.0f dB")
        plt.title("MFCC")
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")
        plt.tight_layout()

        plt.show()

    return mfcc.T

