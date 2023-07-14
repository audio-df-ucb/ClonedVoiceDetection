import librosa
import numpy as np
from math import trunc
from scipy import signal
from numpy import diff

def filter_signal(audio, sr, low_pass_filter_cutoff):
    
    # Smooth signal with low pass filter, the parameters for which were tuned locally
    t = np.arange(len(audio)) / sr
    w = low_pass_filter_cutoff / (sr / 2)
    b, a = signal.butter(5, w, "low")
    smoothed_signal = signal.filtfilt(b, a, audio)

    return smoothed_signal

def get_amplitude(audio, window_size, silence_threshold, sr, low_pass_filter_cutoff):
    
    # Generate amplitude features
    abs_audio = abs(audio)
    smoothed_signal = filter_signal(abs_audio, sr, low_pass_filter_cutoff)

    deriv_amplitude = np.mean(diff(smoothed_signal))
    mean_amplitude = np.mean(smoothed_signal)

    return {
        "abs_deriv_amplitude": abs(deriv_amplitude),
        "mean_amplitude": mean_amplitude,
    }


def normalize_audio_amplitudes(paths):
    
    # Normalize amplitudes to be within [-1, 1] according to max absolute value
    normalized_audio = []
    for file in paths:
        sample = librosa.load(file)[0]
        max_abs = np.max(np.abs(sample))
        normalized_sample = sample / max_abs
        normalized_audio.append(normalized_sample)

    return normalized_audio


def truncate_silences(
    normalized_audio,
    window_size,
    silence_threshold,
    sr=None,
    low_pass_filter_cutoff=None,
    counter=0,
):
    # Remove start and end silences from clips 
    start_ids = []
    end_ids = []
    truncated_audio = []

    for audio in normalized_audio:
        truncation_id_start = None
        truncation_id_end = None

        counter += 1
        if counter % 100 == 0:
            print(
                f"Truncating audio {counter}/{len(normalized_audio)} ({round(counter*100/len(normalized_audio))}%)"
            )

        for j in range(len(audio)):
            roll_average = np.mean(np.abs(audio[j : j + window_size]))
            if roll_average > silence_threshold:
                truncation_id_start = j
                break

        for j in reversed(range(len(audio))):
            roll_average = np.mean(np.abs(audio[j - window_size : j]))
            if roll_average > silence_threshold:
                truncation_id_end = j - window_size
                break

        if truncation_id_start is not None and truncation_id_end is not None:
            truncated_audio.append(audio[truncation_id_start:truncation_id_end])
        start_ids.append(truncation_id_start)
        end_ids.append(truncation_id_end)

    return start_ids, end_ids, truncated_audio


def moving_average(x, w):
    #compute moving average
    return np.convolve(x, np.ones(w), "valid") / w


def get_silence(
    audio, window_size, silence_threshold, sr=None, low_pass_filter_cutoff=None
):
    #computes silent and voiced portions of audio
    thresh = max(abs(audio)) * silence_threshold
    moving_avg = moving_average(abs(audio), window_size)
    silent = np.where(abs(moving_avg) < thresh)
    voiced = np.where(abs(moving_avg) >= thresh)

    # Get percentage of silence and voiced
    pct_pause = len(silent[0]) * 100 / (len(silent[0]) + len(voiced[0]))
    pct_voiced = len(voiced[0]) * 100 / (len(silent[0]) + len(voiced[0]))

    if len(voiced[0]) == 0:
        ratio_pause_voiced = None
    else:
        ratio_pause_voiced = len(silent[0]) / len(voiced[0])

    return {
        "pct_pause": pct_pause,
        "pct_voiced": pct_voiced,
        "ratio_pause_voiced": ratio_pause_voiced,
    }


def get_silence_spread(
    audio, window_size, silence_threshold, sr=None, low_pass_filter_cutoff=None
):
    
    thresh = max(abs(audio)) * silence_threshold
    moving_avg = moving_average(abs(audio), window_size) 

    silent_windows = np.where(moving_avg < thresh)
    moving_avg[silent_windows] = 0
    silence_count = 0
    silence_counts = []

    for i in range(len(moving_avg) - 1):
        item = moving_avg[i]
        next_item = moving_avg[i + 1]

        if item != 0 and next_item == 0:
            silence_count = 0

        elif item == 0 and next_item == 0:
            silence_count += 1

        elif item == 0 and next_item != 0:
            silence_counts.append(silence_count)

        else:
            continue

    # Get spreads/means and normalise
    spread_of_silences = np.std(silence_counts) / len(moving_avg)
    mean_of_silences = np.mean(silence_counts) / len(moving_avg)
    n_pauses = len(silence_counts)

    return {
        "spread_of_silences": spread_of_silences,
        "mean_of_silences": mean_of_silences,
        "silence_counts": silence_counts,
        "n_pauses": n_pauses,
    }
