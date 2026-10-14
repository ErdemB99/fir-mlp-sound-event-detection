# FIR-MLP: Temporal Neural Networks for Sound Event Detection

**A PyTorch implementation of a Finite Impulse Response Multi-Layer Perceptron for multi-label sound event detection in spatial audio.**

This project investigates whether incorporating learnable temporal FIR connections into a feedforward neural network improves acoustic event recognition compared to conventional frame-wise Multi-Layer Perceptrons (MLPs).

Using the TAU Spatial Sound Events 2019 dataset, the proposed FIR-MLP achieves a preliminary **0.6683 ± 0.0078 Micro-F1** across five cross-validation folds, representing a **25.5% relative improvement** over a standard frame-wise MLP baseline.

The implementation includes a complete audio machine learning pipeline, from spectral feature extraction and temporal neural modeling to class-balanced optimization, cross-validation, and model evaluation.

---

## Key Highlights

- **Custom temporal neural architecture:** FIR-based connections capture temporal dependencies across consecutive audio frames.
- **43,019 trainable parameters:** Compact neural architecture with two learnable temporal filtering stages.
- **Multi-label classification:** Detection of 11 acoustic event classes, including overlapping sound events.
- **End-to-end PyTorch implementation:** Automatic differentiation through temporal FIR connections.
- **Class imbalance handling:** Weighted Binary Cross-Entropy and training-set-based feature normalization.
- **Systematic evaluation:** Five-fold cross-validation, threshold optimization, and comparison against a conventional MLP.
- **25.5% relative Micro-F1 improvement:** Preliminary experimental improvement over the frame-wise baseline.

## 1. Research Motivation

Sound Event Detection (SED) aims to identify the presence and timing of acoustic events within audio recordings.

Conventional frame-wise MLPs process individual feature vectors independently, limiting their ability to capture temporal patterns such as sound onsets, offsets, and evolving acoustic characteristics.

Recurrent neural networks, convolutional architectures, and attention-based models address temporal dependencies through more sophisticated mechanisms.

This project explores an alternative: **Finite Impulse Response neural connections**, which incorporate learnable temporal filters directly into a feedforward architecture.

The central research question is:

> Can temporal FIR connections improve multi-label sound event detection compared to a conventional MLP while maintaining a relatively compact neural architecture?

## 2. Dataset and Feature Engineering

The project uses audio recordings and annotations from the **TAU Spatial Sound Events 2019** dataset.

### Audio preprocessing

| Property | Configuration |
|---|---|
| Audio representation | First-Order Ambisonics (FOA) |
| Audio channels | 4 |
| Sampling rate | 48 kHz |
| Spectral transformation | Short-Time Fourier Transform (STFT) |
| FFT size | 2,048 |
| Hop length | 960 samples (20 ms) |
| Mel-frequency bands | 32 per channel |
| Input feature dimension | 128 |
| Target classes | 11 |

Each audio channel is independently transformed into a log-mel representation. The four resulting feature vectors are concatenated into a 128-dimensional feature vector per time frame.

Event annotations are aligned with the corresponding STFT frame timestamps to generate frame-level multi-hot labels.

The resulting dataset contains approximately **1.18 million audio frames**.

### Class imbalance

Only approximately 7.18% of all class-frame label entries are positive, creating a substantial class imbalance.

To address this, the training pipeline implements:

- Per-feature standardization using training data statistics.
- Class-specific positive weights in Binary Cross-Entropy.
- Positive-weight clipping to limit extreme gradients.
- Micro-averaged Precision, Recall, and F1 evaluation.

Normalization statistics and class weights are computed separately for each training fold to avoid validation-data leakage.

## 3. FIR-MLP Architecture

The network consists of three layers with two learnable temporal FIR filtering stages.

```text
4-Channel FOA Audio
         |
         v
STFT + Log-Mel Feature Extraction
         |
         v
128-Dimensional Feature Vector
         |
         v
Temporal FIR Window (5 Frames)
         |
         v
Fully Connected Layer
640 Inputs -> 64 Hidden Neurons
         |
         v
Sigmoid Activation
         |
         v
Temporal FIR Window (3 Frames)
         |
         v
Fully Connected Layer
192 Inputs -> 11 Output Classes
         |
         v
Sigmoid Probabilities
         |
         v
Multi-Label Sound Event Predictions
```

### Model configuration

| Component | Configuration |
|---|---|
| Input dimension | 128 |
| Hidden neurons | 64 |
| Output classes | 11 |
| Input-to-hidden FIR length | 5 |
| Hidden-to-output FIR length | 3 |
| Hidden activation | Sigmoid |
| Output activation | Sigmoid at inference |
| Loss function | Weighted BCEWithLogitsLoss |
| Optimizer | SGD with Momentum |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Trainable parameters | 43,019 |

The model applies causal temporal windows, meaning predictions at time step \(t\) depend only on the current and previous frames.

PyTorch Autograd computes gradients through both FIR filtering stages, allowing temporal connections to be optimized jointly with the network weights.

## 4. Experimental Results

### FIR-MLP vs. Standard MLP

Two neural architectures were evaluated using the same dataset, feature representation, optimization approach, and five-fold cross-validation procedure.

| Model | FIR Lengths | Parameters | Mean Micro-F1 | Std |
|---|---|---:|---:|---:|
| Standard MLP | 1, 1 | 8,971 | 0.5324 | 0.0087 |
| **FIR-MLP** | **5, 3** | **43,019** | **0.6683** | **0.0078** |

### Performance improvement

The FIR-MLP achieved an absolute Micro-F1 improvement of **0.1359**, corresponding to a relative improvement of approximately **25.5%** over the frame-wise MLP baseline.

The results suggest that temporal context provides useful information for acoustic event recognition.

However, the FIR-MLP also contains more trainable parameters than the baseline. Consequently, the observed improvement cannot yet be attributed exclusively to temporal FIR connections.

### Training behavior

The experiments demonstrated:

- Consistently decreasing training and validation losses.
- Increasing validation Micro-F1 over successive epochs.
- Relatively low variability across five cross-validation folds.
- Improved detection performance after introducing feature normalization and class-balanced optimization.

**Evaluation note:** Reported results are preliminary development metrics. Probability thresholds and model checkpoints were selected using the corresponding validation folds. Therefore, these scores should not be interpreted as independent held-out test performance or official TAU benchmark results.

## 5. Project Structure

```text
FIRMLP-main/
|
|-- code/
|   |-- PFG_main.py
|   |-- PFG_neural_network.py
|   |-- PFG_parameters.py
|   |-- tau2019_to_npy.py
|
|-- data/
|   |-- raw/
|   |   |-- foa_dev/
|   |   |-- mic_dev/
|   |   |-- metadata_dev/
|   |
|   |-- training_par/
|   |-- training_target/
|   |-- eval_par/
|   |-- eval_target/
|
|-- results/
|   |-- checkpoints/
|
|-- experiments/
|   |-- FIR_5_3/
|   |-- MLP_1_1/
|
|-- inputParameters.txt
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## 6. Installation

### Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- SciPy
- SoundFile
- Scikit-learn

### Environment setup

Clone the repository and navigate to the project directory.

```bash
git clone <YOUR_REPOSITORY_URL>
cd FIRMLP-main
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## 7. Data Preparation

Download the TAU Spatial Sound Events 2019 dataset from its official distribution source.

Place the four-channel FOA recordings and corresponding CSV metadata files in:

```text
data/raw/foa_dev/
data/raw/metadata_dev/
```

Generate log-mel features and multi-label targets:

```bash
python code/tau2019_to_npy.py
```

The preprocessing pipeline generates paired NumPy files:

```text
data/training_par/<clip_id>.npy
data/training_target/<clip_id>.npy
```

Expected array dimensions:

- Features: `(n_frames, 128)`
- Labels: `(n_frames, 11)`

The output directories must be empty before generating a new dataset to prevent mixing incompatible preprocessing configurations.

## 8. Training and Cross-Validation

The training pipeline supports three operating modes.

### Mode 1: Five-fold cross-validation

```bash
python code/PFG_main.py 1 --epochs 30 --patience 6
```

The pipeline automatically:

1. Splits audio files into five folds.
2. Computes normalization statistics from training files.
3. Calculates class-specific positive weights.
4. Initializes a new neural network for each fold.
5. Trains using weighted Binary Cross-Entropy.
6. Evaluates on the validation fold.
7. Selects validation thresholds and model checkpoints.
8. Reports the mean and standard deviation of Micro-F1.

### Mode 2: Final training

```bash
python code/PFG_main.py 2 --epochs 30
```

The final model checkpoint is saved to:

```text
results/checkpoints/final_model.pt
```

### Mode 3: Evaluation

```bash
python code/PFG_main.py 3
```

Evaluation requires paired feature and label files in:

```text
data/eval_par/
data/eval_target/
```

The pipeline restores the model configuration, trained weights, and feature-normalization statistics from the saved checkpoint.

An independent evaluation dataset must be prepared separately.

## 9. Reproducing the Baseline Comparison

The model architecture is configured through `inputParameters.txt`.

### Standard MLP

```text
3
128
64
11
1
1
0.01
0.9
```

### Temporal FIR-MLP

```text
3
128
64
11
5
3
0.01
0.9
```

For each configuration, execute:

```bash
python code/PFG_main.py 1 --epochs 30 --patience 6
```

Save the resulting experiment outputs separately, since the training script uses the same checkpoint filenames for both architectures.

## 10. Current Limitations

The current implementation is an experimental research prototype with several limitations:

- **No independent benchmark score:** Current results use validation-based model and threshold selection.
- **Parameter-count difference:** The FIR-MLP has substantially more parameters than the standard MLP baseline.
- **Dataset splitting:** Random file-level cross-validation does not necessarily reproduce the official TAU evaluation protocol.
- **Threshold calibration:** Global probability thresholds are tuned using validation data.
- **No localization:** Spatial audio is used as input, but source direction and position are not predicted.
- **No modern architecture comparison:** CRNN and Transformer-based baselines have not yet been evaluated.

These limitations are important when interpreting the experimental results.

## 11. Future Research

Planned extensions include:

1. **Parameter-matched baseline:** Compare the FIR-MLP against a larger conventional MLP with approximately the same number of parameters.
2. **FIR ablation study:** Investigate temporal filter lengths such as 1/1, 3/3, 5/3, and 10/5.
3. **Independent evaluation:** Establish a held-out test set and separate threshold-calibration procedure.
4. **Architecture benchmarking:** Compare against CNN, CRNN, and attention-based sound event detection models.
5. **Efficiency analysis:** Measure inference latency, computational complexity, and parameter efficiency.
6. **Per-class analysis:** Evaluate class-specific Precision, Recall, and F1, including rare acoustic events.

## 12. Technology Stack

**Deep Learning:** PyTorch, automatic differentiation, SGD with Momentum

**Audio Processing:** SciPy, SoundFile, STFT, log-mel filterbanks

**Machine Learning:** NumPy, Scikit-learn, cross-validation, class-weighted optimization

**Evaluation:** Micro-F1, Precision, Recall, Binary Cross-Entropy, threshold selection

---

## Project Status

**Functional research prototype with preliminary experimental results.**

The current implementation demonstrates a measurable improvement over a conventional frame-wise MLP under the evaluated cross-validation setup. Further controlled experiments and independent testing are required before drawing stronger conclusions about architectural superiority or generalization performance.