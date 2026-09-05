#  Skin Lesion Classification with Deep Learning

> **Can a neural network learn to distinguish between visually similar skin lesions and how much does transfer learning actually help?**

This project explores that question using the **HAM10000 dataset**, a collection of dermatoscopic images representing seven different types of skin lesions.

Instead of building only one model, we approached this as a model comparison problem:

**Start simple → establish a baseline → introduce transfer learning → measure what changes.**

The result was a comparison between a lightweight **LeNet-inspired CNN** and a pretrained **ResNet-18** model.

---

##  The Result at a Glance

| Model                     | Approach             | Test Accuracy | Macro F1 |
| ------------------------- | -------------------- | ------------: | -------: |
| 🟡 **LeNet-inspired CNN** | Trained from scratch |    **67.76%** | **0.31** |
| 🟢 **ResNet-18**          | Transfer Learning    |    **89.67%** | **0.84** |

 **Transfer learning improved accuracy by ~22 percentage points.**

But the more interesting story is not just the accuracy.

This project highlights why **medical image classification requires looking beyond a single performance metric**, especially when the dataset is highly imbalanced.

---

#  The Problem

Skin cancer is one of the most common cancers worldwide, and early detection can be critical.

However, classifying skin lesions from images is challenging because:

* Benign and malignant lesions can look visually similar
* Important differences may be subtle
* Lesion boundaries, pigmentation, and texture vary significantly
* Medical datasets often have limited examples of rare conditions

The goal of this project was to build a deep learning pipeline capable of classifying dermatoscopic images into **7 skin lesion categories**.

---

#  Dataset: HAM10000

**HAM10000 — Human Against Machine with 10,000 training images**

The dataset contains **10,015 dermatoscopic images** across seven diagnostic categories.

| Class   | Diagnosis                     | Images |
| ------- | ----------------------------- | -----: |
| `nv`    | Melanocytic nevi              |  6,705 |
| `mel`   | Melanoma                      |  1,113 |
| `bkl`   | Benign keratosis-like lesions |  1,099 |
| `bcc`   | Basal cell carcinoma          |    514 |
| `akiec` | Actinic keratoses             |    327 |
| `vasc`  | Vascular lesions              |    142 |
| `df`    | Dermatofibroma                |    115 |

---

##  Challenge #1: Extreme Class Imbalance

The dataset is heavily dominated by one class.

`nv` represents approximately **67% of all images**, while `df` represents only about **1.1%**.

That creates a major machine learning problem:

> A model can achieve a seemingly good accuracy score while performing poorly on rare classes.

For this reason, the project used more than accuracy to evaluate the models, including:

* Precision
* Recall
* F1-score
* Macro F1
* Weighted F1
* Confusion Matrices

---

#  Data Pipeline

The project began with preparing the dataset for model training.

###  Duplicate Lesion Analysis

The HAM10000 dataset contains multiple images associated with the same lesion.

Using `lesion_id`, we analyzed duplicate lesions to reduce the risk of **data leakage** between training and evaluation.

This is especially important in medical machine learning because evaluating a model on images of lesions it has effectively already seen can produce misleading results.

---

###  Train / Validation / Test Preparation

Images were organized into separate directories for:

```text
Training
Validation
Testing
```

The pipeline automatically copied images from the original HAM10000 image folders into class-based directories.

---

###  Data Augmentation

To increase representation for minority classes, image augmentation was applied using:

* 🔄 Rotation
* ↔️ Width shifting
* ↕️ Height shifting
* 🔍 Zooming
* ↔️ Horizontal flipping
* ↕️ Vertical flipping

The goal was to give the models more variation during training and reduce the dominance of the majority class.

---

#  Model 1 — LeNet-Inspired CNN

The first model was designed as a lightweight baseline.

Rather than immediately using a large pretrained network, we first asked:

> **How far can a relatively simple CNN go when trained from scratch?**

### Architecture

```text
Input Image (28 × 28 × 3)
          │
          ▼
Conv2D → ReLU
          │
          ▼
Max Pooling
          │
          ▼
Conv2D → ReLU
          │
          ▼
Max Pooling
          │
          ▼
Flatten
          │
          ▼
Dense (120)
          │
          ▼
Dense (84)
          │
          ▼
Softmax → 7 Classes
```

### Training Configuration

| Setting       | Value                     |
| ------------- | ------------------------- |
| Framework     | TensorFlow / Keras        |
| Input Size    | 28 × 28                   |
| Optimizer     | RMSprop                   |
| Loss Function | Categorical Cross-Entropy |
| Epochs        | 20                        |
| Batch Size    | 10                        |

---

##  What Did the CNN Learn?

One interesting part of this project was visualizing filters from the first convolutional layer.

### Before Training

The filters contained mostly random patterns resulting from initialization.

### After Training

The network developed more meaningful visual features, including patterns resembling:

* Edge detection
* Color contrast
* Shape and boundary detection
* Blob-like visual structures

This provided a useful look inside the CNN rather than treating it entirely as a black box.

---

#  Model 2 — ResNet-18 with Transfer Learning

The second model used **ResNet-18**, pretrained on ImageNet.

Instead of learning everything from scratch, the network begins with features already learned from millions of images.

Those learned features can then be adapted to the skin lesion classification problem.

### Why Transfer Learning?

Pretrained networks already understand many fundamental visual concepts:

```text
Edges
   ↓
Textures
   ↓
Shapes
   ↓
Complex Visual Patterns
```

The final classification layer was modified to predict the **7 HAM10000 classes**.

### Training Configuration

| Setting      | Value              |
| ------------ | ------------------ |
| Framework    | PyTorch            |
| Architecture | ResNet-18          |
| Pretraining  | ImageNet           |
| Input Size   | 224 × 224          |
| Optimizer    | Adam               |
| Epochs       | 5                  |
| Batch Size   | 32                 |
| Device       | GPU when available |

---

#  Model Comparison

## LeNet vs. ResNet-18

| Feature           | LeNet-inspired CNN | ResNet-18             |
| ----------------- | ------------------ | --------------------- |
| Training Approach | From Scratch       | Transfer Learning     |
| Input Resolution  | 28 × 28            | 224 × 224             |
| Architecture      | Shallow CNN        | Deep Residual Network |
| Parameters        | ~44K               | ~11M                  |
| Pretrained        | ❌                  | ✅                     |
| Test Accuracy     | **67.76%**         | **89.67%**            |
| Macro F1          | **0.31**           | **0.84**              |

###  Key Takeaway

The ResNet-18 model significantly outperformed the baseline CNN.

The improvement came from several factors:

* Higher-resolution images preserved more lesion detail
* Pretrained weights provided strong visual feature initialization
* Residual connections enabled a deeper architecture
* Transfer learning reduced the need to learn everything from scratch

---

#  Evaluation

Both models were evaluated using more than a single accuracy number.

The evaluation pipeline includes:

✅ Test Accuracy
✅ Confusion Matrix
✅ Classification Report
✅ Precision
✅ Recall
✅ F1-score
✅ Per-class performance analysis

This is particularly important because the HAM10000 dataset is highly imbalanced.

A strong overall accuracy does **not automatically mean strong performance across every lesion category**.

---

#  What I Learned

This project reinforced several important machine learning lessons.

### 1.  Accuracy can hide important problems

A model may perform well overall while struggling with minority classes.

**Macro F1 and per-class metrics provide a more balanced evaluation.**

---

### 2.  Medical datasets require careful preprocessing

Duplicate lesions can create data leakage if related images appear across training and evaluation datasets.

Data preparation is just as important as model architecture.

---

### 3.  Image resolution matters

Reducing medical images to **28 × 28 pixels** removes significant visual information.

Higher-resolution input helped ResNet-18 capture more detailed features.

---

### 4.  Transfer learning is powerful

Starting from pretrained visual features produced substantially stronger performance than training a lightweight CNN from scratch.

---

#  Tech Stack

### Languages & Data

* Python
* Pandas
* NumPy

### Deep Learning

* TensorFlow
* Keras
* PyTorch
* Torchvision

### Machine Learning & Evaluation

* Scikit-learn

### Visualization

* Matplotlib

### Development Environment

* Google Colab

---

#  Repository Structure

```text
skin-lesion-classification/
│
├── skin_lesion_classification.py
├── skin_lesion_classification.ipynb
├── requirements.txt
└── README.md
```

> The HAM10000 dataset images are not included in this repository due to their size.

---

#  Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/skin-lesion-classification.git
cd skin-lesion-classification
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Download the Dataset

Download the HAM10000 dataset and organize the dataset paths according to your local environment.

The original project was developed in **Google Colab**, so you may need to update the Google Drive paths in the notebook or Python script.

---

#  Future Improvements

There are several directions I would explore to improve this project further.

### Model Improvements

* Fine-tune ResNet-18 for additional epochs
* Experiment with ResNet-50
* Compare EfficientNet architectures
* Add learning-rate scheduling

### Better Handling of Class Imbalance

* Class-weighted loss
* Focal Loss
* Weighted sampling

### Explainability

* Grad-CAM
* Saliency Maps
* Visual analysis of melanoma misclassifications

### Evaluation

* More detailed per-class analysis
* Cross-validation
* External validation datasets

---

#  Disclaimer

This project was developed for **educational and research purposes**.

It is **not a clinical diagnostic system** and should not be used as a replacement for professional medical evaluation.


---

#  My Focus

My work on this project focused on developing and understanding the machine learning pipeline, including:

* Data preparation and preprocessing
* Image augmentation
* CNN experimentation
* Transfer learning with ResNet-18
* Model evaluation
* Confusion matrix analysis
* Deep learning model interpretation through convolutional filter visualization

---

###  If you found this project interesting

The biggest lesson from this project is simple:

> **Better models matter but better data preparation and evaluation matter just as much.**
