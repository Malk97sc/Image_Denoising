# Package Desing

This package was designed to keep the project clean, modular, and easy to extend.  
Each component has a purpose, making it simple to experiment with different architectures and datasets without breaking the rest of the code.

## Structure

```bash
Image-Denoising/
├── src/
│   ├── denoising/ 
│   ├── data/       # Custom datasets and PyTorch DataLoaders
│   ├── models/     # Neural network architectures 
│   ├── utils/      # Helper modules: metrics, logging, visualization tools
│   ├── train.py    # Central training pipeline
│   ├── test.py     # Evaluation and inference routines
│   └── __init__.py
```

## Local Installation

```bash
pip install -e .
```

## Use

```bash
import denoising
```