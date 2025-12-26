# Pix2Pix: Satellite to Map Translation

This project implements a **Pix2Pix Conditional GAN (cGAN)** to translate satellite imagery into map routes. It is built from scratch using PyTorch and trained on the standard Maps dataset available on Kaggle.

## 🎯 Project Overview

The goal of this project is to learn a mapping from input satellite images to output map images. The model consists of two main components:

* **Generator (U-Net):** Generates the map from the satellite image.
* **Discriminator (PatchGAN):** Distinguishes between real maps and generated maps, enforcing high-frequency structural accuracy.

## 📂 Dataset

The project uses the **Pix2Pix Maps Dataset**.

* **Source:** [Maps Dataset on Kaggle](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset) (or similar).
* **Structure:** Paired images. Each file contains the **Satellite Image** and the **Map Ground Truth** concatenated side-by-side.
* **Preprocessing:** The `MapDataset` class dynamically splits these images to separate inputs from targets.

## 🛠️ Installation & Setup

### Option 1: Running on Kaggle (Recommended)

This project was designed to run in a Kaggle Notebook environment.

1. Create a new Notebook.
2. Add the **Maps Dataset** to your input (`/kaggle/input/`).
3. Copy the scripts (`train.py`, `models.py`, etc.) into the notebook cells or upload them as utility scripts.
4. Run `train.py`.

### Option 2: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/arnav-chauhan-kgpian/pix2pix.translate.git

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```



## 🚀 Usage

### 1. Configuration (`config.py`)

Ensure your paths point to the correct dataset location. If running on Kaggle, the paths usually look like this:

```python
TRAIN_DIR = "/kaggle/input/pix2pix-dataset/maps/train"
VAL_DIR = "/kaggle/input/pix2pix-dataset/maps/val"
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 500

```

### 2. Training

To start training the model:

```bash
python train.py

```

* **Mixed Precision:** The training loop uses `torch.cuda.amp` for faster training and lower memory usage.
* **Checkpoints:** The model saves weights (`gen.pth.tar` and `disc.pth.tar`) periodically.
* **Evaluation:** Sample predictions are saved to the `evaluation/` folder during training to visualize progress.

## 🧠 Model Architecture

### Generator (U-Net)

* **Encoder-Decoder** structure with skip connections.
* Downsampling layers capture the context of the satellite terrain.
* Upsampling layers reconstruct the abstract map features.
* Skip connections preserve spatial information (road alignments, building footprints).

### Discriminator (PatchGAN)

* Instead of classifying the whole image as real/fake, it classifies  patches.
* This forces the generator to produce sharp, high-frequency details (clean lines and edges) rather than blurry averages.

## 📁 File Structure

* `config.py`: Hyperparameters, file paths, and Albumentations transforms.
* `dataset.py`: Custom `MapDataset` class for loading and splitting paired images.
* `generator_model.py`: U-Net implementation.
* `discriminator_model.py`: PatchGAN implementation.
* `train.py`: Main training loop with Mixed Precision (AMP).
* `utils.py`: Helper functions for saving checkpoints and evaluation images.

## 📦 Requirements

```text
torch>=1.10.0
torchvision>=0.11.0
albumentations>=1.1.0
tqdm
numpy
Pillow

```

## 📄 References

* [Pix2Pix Paper: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
