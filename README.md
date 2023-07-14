# Single and Multi Speaker Cloned Voice Detection: From Perceptual to Learned Features

<!-- Add link to license on github and decide which license, check python version for accuracy -->
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8.0](https://img.shields.io/badge/python-3.8.0-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

This is the repository for the following paper: [Single and Multi Speaker Cloned Voice Detection: From Perceptual to Learned Features](). 

At present, only the code relevant to the execution of the single-speaker and multi-speaker pipelines is provided. Please note that only part of the specific dataset utilized in our experiments is publicly accessible. An analogous dataset with voice clones can be constructed using various voice cloning architectures/providers. Features need to be generated and saved to disk and the relevant data handling code in the pipeline needs to be modified for the pipeline to run on the new data.

# Folder Structure

The repository is structured as follows:

| Folder    | File       | Description                                       |
|-----------|------------|---------------------------------------------------|
|__Experiment Pipeline__|
| `/src/`   |`run_pipeline_ljspeech.py`| Runs the pipeline for single voice (LJSpeech) experiments|
| `/src/`   |`run_pipeline_multivoice.py`| Runs the pipeline for multivoice experiments|
| `/src/packages/`  | `ExperimentPipeline.py`     | Class for running the experiment_pipeline and logging results|
| `/src/packages/`  | `ModelManager.py`            |Class for managing the final classification models |
|__Feature Generation__|
| `/src/packages/`  | `AudioEmbeddingsManager.py`  | Class for managing learned features generated using [NVIDIA TitaNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html)|
| `/src/packages/`  | `SmileFeatureManager.py`     | Class for managing spectral features generated using [openSMILE](https://audeering.github.io/opensmile-python/usage.html)|
| `/src/packages/`  | `SmileFeatureGenerator.py`   | Class for generating spectral features and saving to disk for collections of audio files|
| `/src/packages/`  | `SmileFeatureSelector.py`    | Class for selecting spectral features using `sklearn.feature_selection` |
| `/src/packages/`  | `CadenceModelManager.py`     | Class for managing perceptual features generated using handcrafted technqiues|
| `/src/packages/`  | `CadenceUtils.py`            | Utility functions used by `CadenceModelManager` for generating features |
| `/src/packages/`  | `BayesSearch.py`             | A class that implements Bayesian Hyperparameter Optimization for perceptual model |
| `/src/packages/`  | `SavedFeatureLoader.py`      | Helper function for loading during experiments the generated features saved to disk|
|__Data Loaders__|
| `/src/packages/`  | `LJDataLoader.py`            | Class for loading and handling the LJSpeech data for experiments|
| `/src/packages/`  | `TIMITDataLoader.py`         | Class for loading and handling the TIMIT data for multivoice experiments|
|__Data Generation__|
| `/src/packages/`  | `BaseDeepFakeGenerator.py`   | Base class used for processing data used for voice cloning |
| `/src/packages/`  | `ElevenLabsDeepFakeGenerator.py`| Class used to generate deepfakes using the ElevenLabs API |
| `/src/packages/`  | `AudioManager.py`            | Class for resampling audio files and performing adversarial laundering |
|__Misc__|
| `.`       | `README.md` | Provides an overview for the project|
| `.`       | `conda_requirements.txt` | Dependencies for creating the `conda` environment|
| `.`       | `pip_requirements.txt` | Dependencies installed with `pip`|

# Data

An overview of the real and synthetic datasets used in our single-speaker (top) and multi-speaker (bottom) evaluations. The 91,700 WaveFake samples correspond to 13,100 samples per each of seven different vocoder architectures, hence the larger number of clips and duration.

### Single-speaker

| **Type** | **Name** | **Clips (#)** | **Duration (sec)** |
|:--------:|:--------:|:-------------:|:------------------:|
| Real | LJSpeech | 13,100 | 86,117 |
| Synthetic | WaveFake | 91,700 | 603,081 |
| Synthetic | ElevenLabs | 13,077 | 78,441 |
| Synthetic | Uberduck | 13,094 | 83,322 |

### Multi-speaker

| **Type** | **Name** | **Clips (#)** | **Duration (sec)** |
|:--------:|:--------:|:-------------:|:------------------:|
| Real | TIMIT | 4,620 | 14,192 |
| Synthetic | ElevenLabs | 91,700 | 603,081 |

### Publicly Available Data

1. The LJ Speech 1.1 Dataset -- [Data](https://keithito.com/LJ-Speech-Dataset/)
2.  WaveFake: A Data Set to Facilitate Audio Deepfake Detection -- [Paper](https://arxiv.org/abs/2111.02813), [Data](https://zenodo.org/record/5642694)
3. TIMIT Acoustic-Phonetic Continuous Speech Corpus -- [Data](https://catalog.ldc.upenn.edu/LDC93S1)

### Commercial Voice Cloning Tools

1. ElevenLabs (EL) -- https://beta.elevenlabs.io/
2. UberDuck (UD) -- https://app.uberduck.ai/

# Results

### Single-speaker

Accuracy for a personalized, single-speaker classification of unlaundered audio (top) and audio subject to adversarial laundering in the form of additive noise and transcoding (bottom). Dataset corresponds to ElevenLabs (EL), UberDuck (UD), and WaveFake (WF); Model corresponds to a linear (L) or non-linear (NL) classifier, and for a single-classifier (real v. synthetic) or multi-classifier (real vs. specific synthethis architecture); accuracy (%) is reported for synthetic audio, real audio, and (for the single-classifiers) equal error rate (EER) is also reported.


|          |        | Synthetic Accuracy (%) |     |     | Real Accuracy (%) |     |     | EER (%) |     |     |
|----------|--------|:----------------------:|-----|-----|:-----------------:|-----|-----|:-------:|-----|-----|
| **Dataset**  | **Model**  | **Learned**  | **Spectral** | **Perceptual** | **Learned**  | **Spectral** | **Perceptual** | **Learned**  | **Spectral** | **Perceptual** |
|__Unlaundered__|
|Binary|
| EL       | single (L)  | 100.0 | 99.2 | 77.2 | 100.0 | 99.9 | 72.5 | 0.0 | 0.5 | 24.9 |
|          | single (NL) | 100.0 | 99.9 | 82.2 | 100.0 | 100.0 | 80.4 | 0.0 | 0.1 | 18.6 |
| UD       | single (L) | 99.8 | 98.9 | 51.9 | 99.9 | 98.9 | 54.0 | 0.1 | 1.1 | 47.2 |
|          | single (NL) | 99.7 | 99.2 | 54.4 | 99.9 | 99.0 | 56.5 | 0.2 | 0.9 | 44.5 |
| WF       | single (L) | 96.5 | 78.4 | 57.8 | 97.1 | 82.3 | 45.6 | 3.3 | 19.7 | 48.5 |
|          | single (NL) | 94.5 | 87.6 | 50.3 | 96.7 | 90.2 | 52.7 | 4.4 | 11.2 | 48.6 |
| EL+UD    | single (L) | 99.7 | 94.8 | 63.4 | 99.9 | 97.1 | 60.3 | 0.2 | 4.2 | 37.9 |
|          | single (NL) | 99.7 | 99.2 | 57.3 | 99.9 | 99.6 | 69.0 | 0.2 | 0.8 | 37.6 |
| EL+UD+WF | single (L) | 93.2 | 79.7 | 58.4 | 98.7 | 93.0 | 57.6 | 3.6 | 15.9 | 42.1 |
|          | single (NL) | 91.2 | 90.6 | 53.1 | 99.0 | 94.1 | 64.7 | 4.1 | 7.9 | 41.6 |
|Multiclass|
| EL+UD    | multi (L) | 99.9 | 96.6 | 61.0 | 100.0 | 94.6 | 35.7 | - | - | - |
|          | multi (NL) | 99.7 | 98.3 | 65.6 | 100.0 | 97.2 | 43.2 | - | - | - |
| EL+UD+WF | multi (L) | 98.8 | 80.2 | 45.1 | 97.3 | 64.3 | 22.9 | - | - | - |
|          | multi (NL) | 98.1 | 94.2 | 48.6 | 96.3 | 84.4 | 27.6 | - | - | - |
|__Laundered__|
|Binary|
| EL       | single (L)  | 95.5 | 94.3 | 61.1 | 94.5 | 92.6 | 65.2 | 4.9 | 6.7 | 36.6 |
|          | single (NL)  | 96.0 | 96.2 | 70.4 | 95.4 | 95.6 | 69.6 | 4.1 | 4.1 | 30.1 |
| UD       | single (L) | 95.4 | 81.1 | 61.4 | 91.8 | 84.3 | 44.7 | 6.3 | 17.3 | 46.7 |
|          | single (NL) | 95.4 | 86.8 | 52.9 | 93.3 | 86.1 | 55.9 | 5.5 | 13.6 | 45.6 |
| WF       | single (L) | 87.6 | 60.7 | 59.6 | 85.0 | 70.4 | 42.5 | 13.9 | 34.4 | 49.4 |
|          | single (NL) | 83.6 | 77.1 | 51.4 | 85.6 | 76.7 | 53.9 | 15.3 | 23.1 | 47.3 |
| EL+UD    | single (L) | 95.2 | 79.1 | 54.0 | 91.7 | 78.4 | 59.8 | 6.2 | 21.3 | 43.1 |
|          | single (NL) | 94.8 | 86.1 | 55.2 | 93.3 | 90.0 | 62.4 | 6.0 | 12.0 | 41.4 |
| EL+UD+WF | single (L) | 83.7 | 70.9 | 50.6 | 88.6 | 72.9 | 59.7 | 13.2 | 28.2 | 44.8 |
|          | single (NL) | 83.4 | 79.2 | 53.0 | 90.7 | 85.1 | 60.7 | 12.5 | 17.9 | 43.6 |
|Multiclass|
| EL+UD    | multi (L)  | 94.2 | 85.6 | 50.9 | 91.0 | 77.1 | 29.1 | -   | -   | -   |
|          | multi (NL) | 94.5 | 91.7 | 53.2 | 90.3 | 82.9 | 41.3 | -   | -   | -   |
| EL+UD+WF | multi (L)  | 89.8 | 65.4 | 35.3 | 83.1 | 44.3 | 26.2 | -   | -   | -   |
|          | multi (NL) | 88.8 | 78.8 | 39.8 | 82.1 | 63.0 | 28.6 | -   | -   | -   |

### Multi-speaker

Accuracy for a non-personalized, multi-speaker classification of unlaundered audio. Dataset corresponds to ElevenLabs (EL); Model corresponds to a linear (L) or non-linear (NL) classifier, and for a single-classifier (real v. synthetic) or multi-classifier (real vs. specific synthethis architecture); accuracy (%) is reported for synthetic audio, real audio, and (for the single-classifiers) equal error rate (EER) is also reported.

|          |        | Synthetic Accuracy (%) |     |     | Real Accuracy (%) |     |     | EER (%) |     |     |
|----------|--------|:----------------------:|-----|-----|:-----------------:|-----|-----|:-------:|-----|-----|
| **Dataset**  | **Model**  | **Learned**  | **Spectral** | **Perceptual** | **Learned**  | **Spectral** | **Perceptual** | **Learned**  | **Spectral** | **Perceptual** |
| EL       | single (L)  | 100.0 | 94.2 | 83.8 | 99.9 | 98.3 | 86.9 | 0.0 | 3.0 | 1.3 |
|          | single (NL) | 92.3 | 96.3 | 82.2 | 100.0 | 99.7 | 87.7 | 0.1 | 1.6 | 1.4 |



# Research Group

* Sarah Barrington<sup>1</sup> -- <sbarrington@berkeley.edu>
* Romit Barua<sup>1</sup> -- <romit_barua@berkeley.edu>
* Gautham Koorma<sup>1</sup> -- <gautham.koorma@berkeley.edu>
* Hany Farid<sup>1,2</sup> -- <hfarid@berkeley.edu> 

School of Information<sup>1</sup> and Electrical Engineering and Computer Sciences<sup>1,2</sup> at the University of California, Berkeley

# Citation

Please cite this repository as follows if you use its data or code:

```

```
