# Hybrid-CNN-QNN-Image-Classifier
This repository is part of my final year capstone project which is bay leaf classification using quantum computing where I introduced hybrid CNN-QNN model for higher accuracy.

# Hybrid CNN–QNN for Indian Bay Leaf Classification (PyTorch + PennyLane)

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
See the [LICENCE](./LICENCE) file for details.


## Project Overview

This project implements a hybrid **classical–quantum neural network** for image classification of Indian bay leaves. A lightweight CNN is used to extract image features, which are then processed by a quantum variational layer implemented in PennyLane. The final classifier predicts one of three classes: Disease, Dry_Leaf, or Healthy.

---

## Quick Start

### Environment

- Python ≥ 3.9
- PyTorch, torchvision
- scikit-learn, Pillow, matplotlib, seaborn
- PennyLane ≥ 0.42 and (optional) `pennylane-lightning` for faster simulation

Install requirements:

```bash
pip install torch torchvision scikit-learn pillow matplotlib seaborn
pip install pennylane pennylane-lightning
```

### Dataset

Dataset directory structure:

```
<DATASET_ROOT>/
 ├─ Disease/
 ├─ Dry_Leaf/
 └─ Healthy/
```

Update the dataset path in the notebook:

```python
base_path = "/content/drive/MyDrive/Indian_Bay_leaf_or_Tejpatta_Dataset_Balanced"
```

### Running

Open `Hybrid_CNN_QNN.ipynb` and execute all cells. The notebook performs:

1. Data loading and preprocessing (resize 128×128, normalize).
2. Stratified 80/20 train/test split.
3. Model construction.
4. Training for 100 epochs.
5. Evaluation with classification report and confusion matrix.
6. Optional validation loop with optimizer variants and different qubit counts.

---

## Model Architecture

### CNN (classical feature extractor)
- `Conv(3→8, k=3, s=2)` → ReLU → `MaxPool(2)`
- `Conv(8→16, k=3, s=2)` → ReLU → `MaxPool(2)`
- Flatten → `Linear(flattened → n_qubits)`

### QNN (quantum variational layer)
- Device: `default.qubit` (or `lightning.qubit`)
- Circuit:
  - `AngleEmbedding(features)`
  - `RY(weights)` on each wire
  - Entanglement via `CNOT([0,1], [2,3], [1,2])`
  - Measurement: `expval(Z)` on each wire
- Output: vector of length `n_qubits` (via PennyLane `qnn.TorchLayer`)

### Classifier Head
- `Linear(n_qubits → 3)` → logits
- Loss: `CrossEntropyLoss`

---

## Hyperparameters

- Image size: `128×128`
- Batch size: `32`
- Epochs: `100`
- Qubits: `4` or `8`
- Optimizers: Adam, SGD + StepLR, RMSprop, NAdam
- Learning rates: Adam `1e-2`, SGD `1e-2`, RMSprop `1e-3`, NAdam `1e-3`

---

## Results (Representative)

| Configuration                  | Test Accuracy | Notes |
|--------------------------------|---------------|-------|
| 4 qubits + SGD (StepLR)        | ~0.88         | Baseline |
| 4 qubits + RMSprop             | ≈0.99         | Strong performance |
| 4 qubits + NAdam               | 0.99–1.00     | Excellent |
| 8 qubits + Adam (1e-2)         | 0.98–0.99     | Fast convergence |
| 8 qubits (extended training)   | 1.00          | Perfect accuracy |

---

## Best Practices

- Set seeds (`random`, `numpy`, `torch`, CUDA) for reproducibility.
- Adjust learning rates: for Adam, try `1e-3` or `3e-4` for more stability.
- Use early stopping or `ReduceLROnPlateau` to avoid overfitting.
- Apply data augmentation for generalization.
- For larger qubit counts, experiment with deeper variational blocks.

---

## Citation

If you use this repository, please cite:

```bibtex
@software{hybrid_cnn_qnn_classifier,
  title        = {Hybrid CNN–QNN for Indian Bay Leaf Classification},
  author       = {Apoorv Jadhav},
  year         = {2025},
  url          = {https://github.com/apoorv404/hybrid-cnn-qnn-image-classifier},
}
```

---

## Acknowledgements

- PyTorch (https://pytorch.org/)
- PennyLane (https://pennylane.ai/)
- Dataset: Indian Bay Leaf (Tejpatta) Balanced Dataset
PAYGUDE, PRIYANKA; CHAVAN, PRASHANT (2024), “Indian Bay Leaves or Cinnamomum Tamala Leaves Dataset”, Mendeley Data, V1, doi: 10.17632/s9t7sr52wg.1
