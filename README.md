# 🧠 Digital Image Processing Project: Binary Imbalanced Classification

🚀 **Brief Description**  
Binary image classification project using HOG, LBP, GLCM, and VGG19 (FC7 layer) features. Features are fused and reduced using PCA, then classified using a Linear SVM. The model is evaluated using 10-fold cross-validation and standard metrics.

---

## 📌 Project Steps

### 1️⃣ Data Preparation
- **Data Augmentation**: Flip, Rotate
- **Preprocessing**: Image enhancement techniques

### 2️⃣ Feature Engineering
- **Low-Level Features**: 
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Patterns)
  - GLCM (Gray-Level Co-occurrence Matrix)
- **High-Level Features**: 
  - Deep features from **FC7 layer of VGG19**

### 3️⃣ Feature Fusion & Dimensionality Reduction
- **Feature Fusion**: Serial-based combination of features  
- **Dimensionality Reduction**: PCA (Principal Component Analysis)

### 4️⃣ Classification
- **Model**: Linear Support Vector Machine (SVM)  
- **Validation**: 10-fold Cross-Validation (K=10)  
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 📊 Confusion Matrix

![Confusion Matrix](https://github.com/ak-abdullah/Digital-Image-Processing/raw/main/output.png)

---

## 🛠️ Tools & Technologies

- **Languages**: Python  
- **Libraries**:  
  - `OpenCV` (Image Processing)  
  - `scikit-learn` (SVM, PCA, Evaluation Metrics)  
  - `TensorFlow/Keras` (VGG19 feature extraction)  
  - `NumPy`, `Pandas` (Data handling)  
  - `Matplotlib`, `Seaborn` (Visualization)

---

