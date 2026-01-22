# MNIST Diffusion Comparison: DDPM, DDIM, & Stable Diffusion

A professional benchmark suite for **Image-to-Image (I2I) translation** comparing three industry-standard diffusion paradigms. This project evaluates models on **Training Time**, **Inference Latency**, and **Structural Accuracy (MSE)** using a modular, YAML-configured pipeline with **MLflow** tracking.

## 🚀 Project Overview

This repository implements and compares:

1. **DDPM (Denoising Diffusion Probabilistic Models):** The stochastic baseline using 1000 sampling steps.
2. **DDIM (Denoising Diffusion Implicit Models):** A deterministic ODE solver that accelerates inference by jumping through the noise schedule.
3. **Stable Diffusion (Latent Diffusion):** High-efficiency translation performed in a compressed  latent space using a Variational Autoencoder (VAE).

## 🛠 Project Structure

```text
mnist_diffusion_comparison/
├── configs/            # YAML-based hyperparameters for all runs
├── src/
│   ├── models/         # U-Net, Latent U-Net, and VAE architectures
│   ├── schedulers/     # Gaussian (DDPM) and DDIM Solver math logic
│   ├── dataset.py      # Normalized MNIST loaders
│   └── utils.py        # MLflow setup & @time_execution decorators
├── train_ddpm.py       # Pixel-space training script
├── train_sd.py         # Latent-space training script
└── compare_inference.py # The I2I benchmark dashboard

```

## 📊 Benchmarking Methodology

### 1. Time Calculation

Every core function is wrapped in a custom `@time_execution` decorator. This automatically calculates the execution time and logs it as a metric to **MLflow**, allowing for precise hardware-agnostic comparisons.

### 2. Image-to-Image (I2I) Logic

Instead of generating from pure noise, we apply a **Noise Strength ()** to source MNIST digits:



We then evaluate how effectively each model reconstructs the digit from that partial noise.

## 📈 Theoretical Comparison

| Feature | DDPM | DDIM | Stable Diffusion |
| --- | --- | --- | --- |
| **Space** | Pixel () | Pixel () | Latent () |
| **Sampling** | Stochastic (Random) | Deterministic (ODE) | Stochastic/Deterministic |
| **Steps** | 1000 (Fixed) | 20 - 100 (Variable) | 1000 (Latent) |
| **Inference Speed** | Slowest | Fast | **Fastest** |

## ⚙️ Usage

### Installation

```bash
pip install torch torchvision mlflow pyyaml matplotlib tqdm

```

### Running the Comparison

1. **Train the Pixel Baseline:**
```bash
python train_ddpm.py

```


2. **Train the Latent Model:**
```bash
python train_sd.py

```


3. **Run Inference Benchmark:**
```bash
mlflow ui  # Start the dashboard in one terminal
python compare_inference.py

```



## 🖼 Results

The comparison script generates an `i2i_comparison.png` grid showing the **Source Image** followed by reconstructions from each model. You can view the live performance curves and timing logs in the MLflow UI at `http://localhost:5000`.

---
