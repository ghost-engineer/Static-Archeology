# -*- coding: utf-8 -*-

"""
A script for pre-filtering long audio recordings with heavy static
noise.
Purpose: To identify files that likely contain music underneath the
static,
optimizing them for further processing.
The algorithm is based on analyzing spectral, tonal, and rhythmic
features
of the audio signal using the librosa library.
"""

import librosa
import numpy as np
import os
import warnings

#
# Threshold Configuration
# These values are a starting point and may require empirical tuning
# on your specific dataset.
THRESHOLDS = {
    # Spectral Flatness
    # A value close to 1.0 is characteristic of white noise.
    # A value close to 0 is characteristic of tonal signals (music).
    'spectral flatness max': 0.05,

    # Onsets Per Second
    # Music usually has a distinct rhythmic structure.
    # This sets the minimum number of "beats" or "note attacks" per
    # second.
    'onsets_per_second_min': 1.0,

    # Voiced Frames Ratio
    # The PYIN algorithm determines if an audio frame has a specific
    # pitch (f0).
    # Noise is atonal, so the ratio of voiced frames will be low.
    'voiced_frames_ratio_min': 0.10,  # 10% of frames should have a
                                      # detectable pitch

    # Chroma Features Standard Deviation
    # A chromagram shows energy distribution across the 12 musical
    # pitches.
    # In music, this distribution changes constantly (high standard
    # deviation).
    # In noise, it's more stable (low standard deviation).
    'chroma std min': 0.30,

    # Decision Score Threshold
    # How many of the above criteria must be met to classify the
    # recording as likely containing music.
    'decision_score_min': 2
}

def analyze_audio_features(file_path: str) -> dict:
    """
    Analyzes a single audio file and extracts a set of numerical
    metrics.
    Args:
        file_path (str): Path to the audio file (MP3, WAV, etc.).
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    metrics = {}

    try:
        #1. Load Audio
        # sr=None preserves the original sampling rate.
        # duration = 60 limits the analysis to the first 60 seconds for
        # speed.
        print(f"\nAnalyzing file: {os.path.basename(file_path)}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=None, duration=60)

        if len(y) == 0:
            print("File is empty or could not be read.")
            return None

        duration = librosa.get_duration(y=y, sr=sr)
        metrics['duration'] = duration

        #2. Spectral Energy Analysis (FFT-based)
        # Get the magnitude spectrogram
        S = np.abs(librosa.stft(y))

        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(S=S)
        metrics['spectral_flatness_mean'] = np.mean(flatness)

        # Chroma Features (Tonal components)
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)

        # Calculate the mean standard deviation across all chroma
        # features.
        # A high value suggests harmonic diversity (music).
        metrics['chroma std mean'] = np.mean(np.std(chroma, axis=1))

        #3. Pitch Detection
        # Use PYIN to estimate the fundamental frequency (f0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=y,
            sr=sr,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=2048  # Increase frame length for better
                               # low-frequency capture
        )
        # Calculate the ratio of frames where a pitch was confidently
        # detected
        metrics['voiced_frames_ratio'] = np.sum(voiced_flag) / \
                                         len(voiced_flag) if len(voiced_flag) > 0 else 0

        #4. Onset Detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        metrics['onset_count'] = len(onsets)
        metrics['onsets_per_second'] = len(onsets) / duration if \
                                        duration > 0 else 0

        return metrics

    except Exception as e:
        print(f"Failed to process file {file_path}. Error: {e}")
        return None

def classify_audio(metrics: dict, thresholds: dict) -> tuple:
    """
    Makes a decision about the presence of music based on metrics and
    thresholds.
    Args:
        metrics (dict): Dictionary of metrics from
                        analyze_audio_features.
        thresholds (dict): Dictionary of threshold values.
    Returns:
        tuple: (A string with the decision, the final score, and check
                details).
    """
    score = 0
    details = []

    # Check #1: Spectral Flatness
    flatness = metrics.get('spectral_flatness_mean', 1.0)
    if flatness < thresholds['spectral flatness max']:
        score += 1
        details.append(f" [V] Spectrum is tonal "
                       f"(flatness={flatness:.3f} < {thresholds['spectral flatness max']})")
    else:
        details.append(f" [X] Spectrum is noise-like "
                       f"(flatness={flatness:.3f} >= {thresholds['spectral flatness max']})")

    # Check #2: Rhythmic Onsets
    onsets_ps = metrics.get('onsets_per_second', 0)
    if onsets_ps > thresholds['onsets_per_second_min']:
        score += 1
        details.append(f" [V] Rhythm detected ({onsets_ps:.2f} "
                       f"onsets/sec > {thresholds['onsets_per_second_min']})")
    else:
        details.append(f" [X] No rhythm detected ({onsets_ps:.2f} "
                       f"onsets/sec <= {thresholds['onsets_per_second_min']})")

    # Check #3: Presence of Pitch (Voice/Instruments)
    voiced_ratio = metrics.get('voiced_frames_ratio', 0)
    if voiced_ratio > thresholds['voiced_frames_ratio_min']:
        score += 1
        details.append(f" [V] Tonal components found "
                       f"({voiced_ratio:.1%} > {thresholds['voiced_frames_ratio_min']:.0%})")
    else:
        details.append(f" [X] Signal is atonal ({voiced_ratio:.1%} <= "
                       f"{thresholds['voiced_frames_ratio_min']:.0%})")

    # Check #4: Harmonic Diversity
    chroma_std = metrics.get('chroma std mean', 0)
    if chroma_std > thresholds['chroma_std_min']:
        score += 1
        details.append(f" [V] Harmonic development found "
                       f"(chroma_std={chroma_std:.3f} > {thresholds['chroma_std_min']})")
    else:
        details.append(f" [X] No harmonic development "
                       f"(chroma_std={chroma_std:.3f} "
                       f"<= {thresholds['chroma_std_min']})")

    # Final Decision
    if score >= thresholds['decision_score_min']:
        decision = ">> Verdict: Likely contains music"
    else:
        decision = ">> Verdict: Likely static only"

    return decision, score, details

def process_audio_files(file_paths: list, thresholds: dict):
    """
    Main function to process a list of audio files.
    Args:
        file_paths (list): A list of file paths.
        thresholds (dict): The configuration of thresholds.
    """
    if not file_paths:
        print("The list of files to analyze is empty.")
        return

    print("-- Starting Audio Filtering Process ---")
    print(f"Using the following thresholds for decision (requires "
          f"{thresholds['decision_score_min']} matches):")
    for key, value in thresholds.items():
        if key != 'decision_score_min':
            print(f"  {key}: {value}")

    for file_path in file_paths:
        metrics = analyze_audio_features(file_path)
        if metrics:
            decision, score, details = classify_audio(metrics,
                                                      thresholds)
            print("  Detailed Analysis:")
            for detail in details:
                print(f"    {detail}")
            print(f"  Final Score: {score} out of {len(details)}")
            print(f"  {decision}")

    print("\n--- Audio Filtering Process Finished ---")

if __name__ == "__main__":
    #
    # MAIN BLOCK TO RUN THE SCRIPT
    #
    #1. PREPARE FILES FOR ANALYSIS
    #
    #
    ##
    # Place your MP3/WAV files in the same folder as this script,
    # or provide the full paths to them.
    # IMPORTANT: To work with MP3 files, you might need to install
    # ffmpeg.
    # Instructions:
    # https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
    #
    # For demonstration, we'll create some test WAV files using
    # the soundfile library. If you don't have it installed, run:
    #
    #
    # pip install soundfile

    try:
        import soundfile as sf

        sr_test = 22050
        duration_test = 10

        # File #1: Pure noise
        noise = np.random.randn(sr_test * duration_test) * 0.8
        sf.write("test_noise_only.wav", noise, sr_test)

        # File #2: Noise + very faint music (sine waves)
        t = np.linspace(0., duration_test, int(sr_test * \
                                              duration_test), endpoint=False)
        tone_melody = (np.sin(2*np.pi*220*t) + \
                      np.sin(2*np.pi*261*t*1.5) + \
                      np.sin(2*np.pi*330*t*0.5))
        music_signal = noise * 0.7 + tone_melody * 0.05 # Music is
                                                        # very quiet
        sf.write("test_music_and_noise.wav", music_signal, sr_test)

        # File #3: More prominent music with noise
        music_signal_stronger = noise * 0.5 + tone_melody * 0.15
        sf.write("test_music_stronger.wav", music_signal_stronger,
                  sr_test)

        audio_files_to_check = [
            "test_noise_only.wav",
            "test_music_and_noise.wav",
            "test_music_stronger.wav",
            "non_existent_file.mp3" # Example of error handling
        ]
        print("Test files 'test*.wav' created successfully.")

    except ImportError:
        print("WARNING: 'soundfile' library not found (pip install "
              "soundfile).")
        print("Test files cannot be created. Please specify the paths "
              "to your files manually.")
        # REPLACE THIS LIST WITH YOUR FILES
        audio_files_to_check = [
            # "C:/path/to/your/file1.mp3",
            # "radio_archive_part2.wav",
        ]
    except Exception as e:
        print(f"An error occurred while creating test files: {e}")
        audio_files_to_check = []

    #2. RUN THE ANALYSIS
    process_audio_files(audio_files_to_check, THRESHOLDS)