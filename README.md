![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![Status](https://img.shields.io/badge/Project-Completed-green)

# Retinal Blood Vessel Segmentation using Unsupervised Deep Learning

Deep learning pipeline for **retinal blood vessel segmentation** using an **unsupervised autoencoder-based approach**.
The system learns structural patterns of retinal images and extracts vascular structures without requiring manual annotations.

This project demonstrates a full medical image segmentation workflow including **data preprocessing, model training, clustering-based segmentation, and post-processing**.

Unsupervised deep learning approach for retinal vessel segmentation using autoencoder feature learning and clustering.

---

## Project Overview

![Pipeline Overview](docs/pipeline_overview.png)

```mermaid
flowchart LR

A[Retinal Fundus Image] --> B[Image Preprocessing]

B --> C[Convolutional Autoencoder]

C --> D[Latent Feature Representation]

D --> E[K-Means Clustering]

E --> F[Vessel Pixel Identification]

F --> G[Segmentation Mask Reconstruction]

G --> H[Final Retinal Vessel Segmentation]

---

## Research Motivation

Retinal vessel segmentation is a critical step in automated diagnosis of ophthalmic diseases such as diabetic retinopathy and glaucoma.

Manual annotation of retinal vessels is expensive and time-consuming. This project explores an unsupervised approach using deep learning representations to automatically extract vascular structures.

## Project Highlights

- Unsupervised retinal vessel segmentation
- Autoencoder-based feature learning
- K-Means clustering for vessel extraction
- Post-processing to refine vascular structures
- Implemented in **PyTorch**

---

## Example Result

| Input Retina Image             | Segmented Vessels                     |
| ------------------------------ | ------------------------------------- |
| ![](outputs/input_example.png) | ![](outputs/retinal_segmentation.png) |

---

## Project Pipeline

1. Retina Image Acquisition
2. Image Preprocessing
3. Autoencoder Feature Learning
4. Feature Extraction
5. K-Means Clustering
6. Vessel Mask Reconstruction
7. Post-processing

## Segmentation Pipeline

```mermaid
flowchart LR

A[Retinal Image] --> B[Preprocessing]

B --> C[Autoencoder Encoder]

C --> D[Latent Feature Representation]

D --> E[K-Means Clustering]

E --> F[Vessel Pixel Identification]

F --> G[Segmentation Mask Reconstruction]

G --> H[Post Processing]

H --> I[Final Vessel Segmentation]
```


---

## Repository Structure

```
Retinal-Unsupervised-Segmentation
│
├── data
│   └── images
│
├── models
│   └── best_autoencoder.pth
│
├── outputs
│   └── retinal_segmentation.png
│
├── src
│   ├── dataset.py
│   ├── train.py
│   └── utils.py
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```
git clone https://github.com/annikanatarajan1-design/retinal-unsupervised-segmentation.git
cd retinal-unsupervised-segmentation
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Quick Start

Run the full segmentation pipeline:

python main.py

---


## Data Source

Retinal images used in this project were obtained from a university laboratory dataset.

The dataset is used for research and educational purposes. Due to usage restrictions, the full dataset is not publicly distributed in this repository.

Total images: 256  
Image format: PNG  
Average size: ~260 KB

Due to dataset usage restrictions, the full dataset is not included in this repository.
A small sample is provided in `data/images/NORMAL*.png`.

Users can replace the images in `data/images` with their own retinal datasets.

---

## Running the Project

Train the autoencoder:

```
python src/train.py
```

Run segmentation pipeline:

```
python main.py
```

---

## Model

The segmentation model is based on a **convolutional autoencoder** which learns compressed feature representations of retinal structures.

Architecture components:

* Encoder – extracts structural features
* Latent space – compressed representation
* Decoder – reconstructs retinal structures

Features extracted from the latent representation are clustered using **K-Means** to separate vessel pixels from background.

---

## Results

The model successfully extracts **fine retinal vascular structures**, including thin vessels, without requiring labeled segmentation masks.

Advantages of the method:

- No manual annotation required
- Works with limited data
- Preserves vascular topology

---

## Limitations

- Performance may vary across datasets
- Very thin vessels may be missed
- Model sensitive to image quality

---

## Future Improvements

- U-Net based segmentation
- Self-supervised representation learning
- Multi-scale vessel detection
- Quantitative evaluation (Dice, IoU)

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-Learn

---

## Citation

If you use this repository in research or projects, please cite:

Annika Natarajan (2026)  
**Unsupervised Retinal Vessel Segmentation using Deep Learning**.  
GitHub Repository.

---

## Author

Annika Natarajan  
Machine Learning / Computer Vision Projects