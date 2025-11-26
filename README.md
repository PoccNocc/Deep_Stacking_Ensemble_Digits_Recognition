# Deep Stacking Neural Network Ensemble

An advanced ensemble learning solution for handwritten digit recognition, implementing a Deep Stacking architecture with multiple diverse neural network base models and a meta-learner. This project serves as a comprehensive solution for the **Machine Learning Specialization - C2W2 Assignment**.

## Project Overview

This repository contains the implementation of a **Stacked Generalization (Stacking)** ensemble. Instead of relying on a single neural network, this system trains five distinct "base" models and uses a secondary "meta-learner" to learn how to best combine their predictions.

The project demonstrates:
1.  **Single Model Baseline**: A standard neural network implementation (`C2_W2_Solution.ipynb`).
2.  **Deep Stacking Ensemble**: A complex ensemble pipeline with data leakage prevention (`C2_W2_DeepStacking_Solution.ipynb`).

## Features

### Deep Stacking Architecture
- **5 Diverse Base Models**:
    - **Original Model**: Balanced baseline architecture.
    - **Deep Model**: Multiple hidden layers for hierarchical feature extraction.
    - **Wide Model**: Large layer width for parallel feature detection.
    - **Dropout Model**: Regularized network to prevent overfitting.
    - **Tanh Model**: Alternative activation function for error decorrelation.
- **Meta-Learner**: A specialized neural network that takes the probability outputs of base models as input to make the final prediction.

### Advanced Methodology
- **Data Leakage Prevention**: Implements a strict 3-way data split (Base Train, Meta Train, Test) to ensure the meta-learner is trained on unseen predictions.
- **Robust Evaluation**: Comprehensive accuracy reporting for both individual models and the final ensemble.
- **Convergence Analysis**: Detailed tracking of training loss and accuracy across all models.

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow / Keras
- NumPy, Scikit-learn, Matplotlib

### Dependencies
```bash
pip install tensorflow numpy scikit-learn matplotlib
```

## Dataset Preparation

### Machine Learning Specialization Data
The dataset used in this project is the handwritten digit dataset from the **Machine Learning Specialization (Course 2, Week 2)** by DeepLearning.AI.

1.  **Source**: Access the assignment materials from the [Coursera Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction).
2.  **Setup**: Place the `X.npy` and `y.npy` files into the `data/` directory:
    ```
    data/
    ├── X.npy
    └── y.npy
    ```

## Usage

### 1. Deep Stacking Solution
Open and run the main stacking notebook:
```bash
jupyter notebook C2_W2_DeepStacking_Solution.ipynb
```
This notebook will:
- Load and split the data into Base, Meta, and Test sets.
- Construct and train the 5 diverse base models.
- Generate meta-features (predictions) from the base models.
- Train the Meta-Learner on these features.
- Evaluate the final ensemble accuracy on the hold-out Test set.


## Model Performance

| Model | Architecture | Test Accuracy | Role |
|-------|-------------|---------------|----------------|
| **Original** | 400->25->15->10 | ~92.0% | Baseline |
| **Deep** | 400->128->64->10 | ~93.4% | Hierarchical Features |
| **Wide** | 400->128->10 | ~93.5% | Parallel Features |
| **Dropout** | 400->64->32->10 | ~93.1% | Regularization |
| **Tanh** | 400->50->10 | ~91.9% | Diversity |
| **Stacking Ensemble** | **Meta-Learner** | **~94-95%** | **Consensus Prediction** |

*Note: The ensemble consistently outperforms individual models by reducing variance and correcting individual biases.*

## File Structure

```
Neural_Networks_Project/
├── README.md                          # Project documentation
├── C2_W2_DeepStacking_Solution.ipynb  # Main Stacking Ensemble implementation
├── data/
│   ├── X.npy                          # Input features (images)
│   └── y.npy                          # Target labels
```

## Technical Details

### Why Stacking?
Stacking works because different models make different *types* of errors. By combining them:
- **Diversity**: We use different architectures (Deep vs Wide) and activations (ReLU vs Tanh) to ensure errors are uncorrelated.
- **Meta-Learning**: The meta-model learns which base model is most reliable for specific types of inputs, effectively acting as a "manager" for the committee of neural networks.

## Citation

If you use this work, please cite:

```bibtex
@misc{Deep_Stacking_Ensemble,
  title={Deep Stacking Ensemble for Digit Recognition},
  author={Sébastien L'Huillier},
  year={2025},
  publisher={GitHub}
}
```

## Contact

For questions or collaboration opportunities, please open an issue or contact sebastien.lhui@gmail.com
