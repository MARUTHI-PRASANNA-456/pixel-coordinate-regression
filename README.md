# Pixel Coordinate Regression using Deep Learning

## Overview

This repository contains my solution to the Supervised Regression.
The objective of this is to predict the (x, y) coordinates of a single bright pixel (value = 255) in a 50×50 grayscale image, where all other pixels have a value of 0.

The problem is formulated and solved as a supervised regression task using deep learning, with a strong emphasis on correct problem framing, data reasoning, clean implementation, and interpretability.

---

## Problem Statement

### Given:

- A 50×50 grayscale image
- Exactly one pixel has intensity 255
- All other pixels are 0

The location of the bright pixel is randomly assigned

### Goal:

Predict the (x, y) coordinates of the bright pixel using deep learning techniques

---

## Key Design Decisions

### 1. Why This Is a Regression Problem

The output consists of continuous-valued coordinates (x,y).
Therefore, the problem is correctly formulated as a supervised regression task, not classification.

- Input: Image (50×50)
- Output: Continuous values (x,y)
- Loss Function: Mean Squared Error (MSE)

### 2. Dataset Generation Strategy

There is no publicly available dataset that satisfies the strict constraints of this problem.
Hence, a synthetic dataset is generated programmatically

Dataset Characteristics:

- Each image contains exactly one bright pixel
- Pixel location is sampled uniformly at random
- Ensures no spatial bias
- Covers the entire coordinate space

Rationale:

- Uniform sampling ensures fair learning across all pixel locations
- Synthetic generation allows full control over constraints
- Avoids unnecessary disk I/O by generating data directly as tensors

This approach is standard practice in deep learning for synthetic and controlled tasks.

### 3. Why Use Images as Tensors (Not Image Files)

In deep learning, grayscale images are represented as 2D tensors.
Saving images as PNG/JPEG files provides no learning benefit for this task and introduces unnecessary overhead.

The model operates directly on tensors of shape:
(batch_size, 1, 50, 50)

This preserves full spatial structure and is equivalent to standard image-based pipelines used in practice.

### 4. Model Choice

A lightweight Convolutional Neural Network (CNN) is used because:

- CNNs preserve spatial locality
- They are well-suited for coordinate regression
- The task is simple and deterministic, so a deeper model is unnecessary

The architecture balances simplicity and correctness, avoiding overengineering.

---

## Project Structure

```text
.
│
├── pixel_regression.ipynb   # Complete notebook (training, evaluation, plots)
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── results/                 # Saved plots (optional)
```

---

## Installation & Setup

### 1. Clone the Repository:

```bash
git clone https://github.com/<your-username>/pixel-coordinate-regression.git
cd pixel-coordinate-regression
```

### 2. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook:

```bash
jupyter notebook pixel_regression.ipynb
```

---

## Training Details

- Framework: PyTorch
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Train / Validation Split: 80 / 20
- Batch Size: 64
- Epochs: 10 (early convergence observed)

---

## Results & Observations

### Training Behavior

- Rapid convergence within a few epochs
- Training and validation losses closely aligned
- No signs of overfitting

This behavior is expected due to:

- Deterministic nature of the task
- Absence of noise
- Strong inductive bias of CNNs for spatial problems

### Performance Interpretation

- Near-zero validation loss indicates sub-pixel localization accuracy
- Loss saturation reflects convergence to numerical precision limits, not memorization

Additionally, localization accuracy is quantified using Mean Absolute Error (MAE) for x and y coordinates, as well as the mean Euclidean distance error in pixel space. These metrics confirm sub-pixel localization accuracy across the validation set.


---

## Visualization & Evaluation

The notebook includes:

- Training and validation loss curves
- Visualization of predicted vs ground-truth pixel locations
- Qualitative confirmation of spatial localization

These plots provide clear evidence that the model performs as intended.

### Coordinate Visualization

For qualitative evaluation, the notebook explicitly visualizes both the **ground truth** and **predicted (x, y) coordinates** on the input image:

- **Green dot**: Ground truth pixel location  
- **Red dot**: Predicted pixel location  

The corresponding coordinate values are also displayed alongside each visualization.  
This directly demonstrates the model’s ability to localize the bright pixel spatially, as required by the assignment.

---

## Code Quality & Best Practices

- Modular and readable code
- PEP8-compliant formatting
- Clear comments and function docstrings
- Deterministic data generation
- Reproducible training pipeline

---

## Conclusion

This project demonstrates:

- Correct framing of a supervised regression problem
- Thoughtful dataset design and justification
- Appropriate use of deep learning techniques
- Clean, interpretable, and reproducible implementation

The solution prioritizes clarity and correctness over unnecessary complexity, in line with the assignment’s evaluation guidelines.

---

## Author

Maruthi Prasanna Reddy<br>
Phone : +91 8105133095<br>
Email ID : maruthipr456@gmail.com
