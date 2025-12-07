# Lab-on-Python-for-Deep-Learning-Challenge-0

# PDL Challenge 0 — ResNet18 on Heterogeneous CIFAR-10

This repository contains our implementation and materials for **PDL Challenge 0**, which investigates how **ResNet18** performance changes when the **CIFAR-10** class distribution is made heterogeneous/imbalanced, with a focus on **per-class accuracy** and mitigation via **loss re-weighting**. :contentReference[oaicite:0]{index=0}

## Overview

Real-world datasets are rarely perfectly balanced. This challenge explores the robustness and fairness of CNN classifiers under controlled class imbalance using CIFAR-10. The core workflow is:

1. Load/preprocess CIFAR-10 with a **custom per-class sample vector**  
   \(N = [N1, N2, \ldots, N10]\).  
2. Train **ResNet18** for **20 epochs** on a balanced setting  
   \(N = [4000, \ldots, 4000]\).  
3. Repeat training across different random initializations (target: 10 runs).  
4. Compute **per-class accuracy**.  
5. Create **10 imbalanced variants** where one class is over-represented  
   \(Nk = 6000\), others \(Nj = 3777\).  
6. Visualize effects (confusion matrices, radar/heatmaps, etc.).  
7. Mitigate imbalance by modifying training (e.g., **weighted CE**, class-balanced loss). :contentReference[oaicite:1]{index=1}

## What’s in this repo

- `Challenge_0.pdf` — official problem description. :contentReference[oaicite:2]{index=2}  
- `PDL Challenge 0.ipynb` — notebook implementation. :contentReference[oaicite:3]{index=3}  
- `PDL Challenge 0.py` — script version (see notes about a small typo). :contentReference[oaicite:4]{index=4}  
- `Challenge 0.pdf` — our slides/report summary. :contentReference[oaicite:5]{index=5}  

## Method

### Data loader
We implement a custom loader that:
- downloads CIFAR-10,
- builds indices per class,
- selects the first `Nk` samples per class based on the input vector `N`,
- returns a training subset loader and the standard test loader. :contentReference[oaicite:6]{index=6}

### Model
- **ResNet18** imported from `torchvision.models`. :contentReference[oaicite:7]{index=7}  

### Training setup (current code)
- Loss: CrossEntropyLoss
- Optimizer: SGD (lr=0.01, momentum=0.9)
- Epochs: 20 :contentReference[oaicite:8]{index=8}

### Imbalance mitigation
The code demonstrates a simple **weighted CrossEntropy** using inverse-frequency-style weights for the imbalanced setting. :contentReference[oaicite:9]{index=9}

## Results (from our slides)

Key observations we reported:
- Training accuracy typically starts around **42–44% at epoch 1** and rises to about **92–93% by epoch 10**.  
- A longer 20-epoch run reached **100% training accuracy by ~epoch 18**, suggesting potential **overfitting**.  
- **Dataset heterogeneity significantly impacts performance**, and **weighted loss functions** can help mitigate imbalance but may require tuning. :contentReference[oaicite:10]{index=10}

## How to run

### Requirements
- Python 3.8+
- `torch`, `torchvision`
- `numpy`
- `matplotlib`
- `seaborn` (used for a bar plot in the current code) :contentReference[oaicite:11]{index=11}

### Install
```bash
pip install torch torchvision numpy matplotlib seaborn
````

### Run script

```bash
python "PDL Challenge 0.py"
```

### Run notebook

Open:

```text
PDL Challenge 0.ipynb
```

and run all cells.

## Important notes / suggested fixes

1. **Script import typo**
   The first line in `PDL Challenge 0.py` appears to have a typo. Replace it with:

   ```python
   import torch
   import torch.nn as nn
   ```

   The notebook version already has correct imports. 

2. **Match the assignment’s 10-run protocol**
   The current code shows the pipeline for one run per setting. To fully match the spec, wrap training/evaluation in a loop over **10 random seeds** and average accuracy curves and per-class results. 

3. **Train/test split per class (if you extend the loader)**
   The challenge suggests splitting train/test **proportionally for each class according to `N`**. The provided code uses the default CIFAR-10 test set; consider implementing a class-proportional split if required by your grader. 

4. **Memory guidance**
   When running the full imbalance campaign, avoid saving all models; collect statistics and free memory as instructed. 

## Possible extensions

* Add:

  * radar plots / heatmaps for per-class accuracy,
  * confusion matrices per setting,
  * class-balanced loss variants,
  * data augmentation strategies,
  * clearer experiment logging (CSV/JSON). 

## Acknowledgements

* CIFAR-10 dataset
* ResNet18 implementation from `torchvision`
* Course/Challenge material: **PDL Challenge 0 (SR+ITL, Feb 21, 2025)**. 
