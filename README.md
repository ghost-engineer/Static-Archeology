# 🎙️ Static-Archeology

**Audio Pre-Filter for Static Noise**

> This Python script is designed to pre-filter audio recordings (e.g., from radio broadcasts) that suffer from heavy static noise. Its primary goal is to distinguish between files that likely contain faint music and those that are purely static. This allows for prioritizing files for more computationally expensive noise reduction and music extraction tasks.

- **No machine learning or visual analysis** — relies purely on digital signal processing (DSP) techniques provided by the `librosa` library.

---

## ⚙️ How It Works

The script analyzes each audio file based on four key characteristics:

1. **Spectral Flatness**  
   Measures how noise-like a sound is. Pure tones and music have a low flatness value, while white noise has a high value.
2. **Onset Detection**  
   Detects rhythmic "attacks" or beats. Music typically has a regular rhythmic structure, whereas static does not.
3. **Pitch Detection**  
   Identifies frames in the audio that have a discernible musical pitch (fundamental frequency). Music is tonal; static is atonal.
4. **Chroma Features**  
   Analyzes the distribution of energy across the 12 musical notes. Music shows harmonic variation and development, while noise is harmonically flat.

> 📝 Based on a scoring system that combines these metrics, the script provides a verdict:  
> **"Likely contains music"** or **"Likely static only"**.

---

## 🛠️ Prerequisites

- **Python** 3.6+
- Required Python libraries:
  ```bash
  pip install librosa numpy soundfile
  ```
- **FFmpeg** (for MP3 support):  
  `librosa` needs ffmpeg to be installed on your system to load audio formats other than WAV.

  - **Windows:** Download from the official site and add the `bin` directory to your system's PATH.
  - **macOS:**
    ```bash
    brew install ffmpeg
    ```
  - **Linux:**
    ```bash
    sudo apt-get install ffmpeg
    ```

---

## 🚀 Usage

1. Save the script as `audio_filter.py`.
2. Place your audio files (`.wav`, `.mp3`, etc.) in the same directory as the script.
3. *(Optional)* Modify the `audio_files_to_check` list in the `if __name__ == '__main__':` block to include the full paths to your files.
4. Run the script:
   ```bash
   python audio_filter.py
   ```
5. The script will analyze each file and print a detailed report and a final verdict to the console.

> On its first run, the script will attempt to generate three test `.wav` files to demonstrate its functionality.

---

## ⚡ Configuration

The core of the classification logic depends on the **THRESHOLDS** dictionary at the top of the script:

```python
THRESHOLDS = {
    'spectral flatness max': 0.05,
    'onsets_per_second_min': 1.0,
    'voiced frames ratio min': 0.10,
    'chroma std min': 0.30,
    'decision_score_min': 2
}
```

### Parameter Explanations:

- **spectral_flatness_max:** Lower this if your music is very tonal and still being missed.
- **onsets_per_second_min:** Increase for fast-paced music, decrease for ambient or slow music.
- **voiced_frames_ratio_min:** This is a good indicator of any pitched sound. Lower it if the music is extremely faint.
- **chroma_std_min:** Increase if you only want to detect harmonically rich music.
- **decision_score_min:** The number of criteria (out of 4) that must be met. A higher value makes the filter stricter.

> **Recommendation:**  
> Run the script on a small set of files where you already know the content (some with music, some without). Observe the metric values in the output and adjust the thresholds to best separate the two groups.