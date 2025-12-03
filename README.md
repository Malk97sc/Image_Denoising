# Image Denoising

The goal of this project is understand how various Deep Neural Network architectures, like traditional Convolutional Neural Networks (CNNs) to Transformer-based models learn to remove noise from images.

## Problem Statement

This is a classic problem in Computer Vision, specifically focusing on the subfield of Image Restoration. The core problem is to learn a function $f_{\theta}$ such that:

$$
\hat{I}_{\text{clean}} = f_{\theta}(I_{\text{noisy}})
$$

where $f_{\theta}$ is a Neural Network trained to transform a noisy image $I_{\text{noisy}}$ into a clean (without noise) image $I_{\text{clean}}$.


## Objetives

### General 

- Analyze and compare the performance of different deep learning architectures in the task of image denoising.

### Specific 

- Implement and evaluate models based on Convolutional Neural Networks (CNNs), like DnCNN and NAFNet.

- Integrate variants of these models that incorporate attention mechanisms.

- Investigate Transformer-based architectures like Restormer.

- Compare the results of different models and analyze the strengths and weaknesses of each approach.

## Project Structure

```bash
Image-Denoising/
├── src/
│   ├── denoising/
│   │   ├── data/
│   │   ├── losses/
│   │   ├── metrics/
│   │   ├── models/
│   │   ├── train/
│   │   ├── utils/
│   │   ├── visualization/
│   │   └── __init__.py
│   └── README.md
├── datasets/
├── results/
│   ├── logs/
│   ├── denoised_samples/
│   └── comparisons/
├── notebooks/
├── README.md
└── setup.py
```

## References

- Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.
[Paper on arXiv](https://arxiv.org/abs/1608.03981)

- U-Net: Convolutional Networks for Biomedical Image Segmentation.
[Paper on arXiv](https://arxiv.org/abs/1505.04597)