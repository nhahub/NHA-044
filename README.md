Arabic Hand Sign Recognition System 🤖✋
A comprehensive deep learning system for recognizing Arabic sign language gestures using advanced computer vision and ensemble machine learning techniques.
### 🔗 Access to Dataset
📂 **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1cxJNUv6bMzfTW6X2_9W1mk8VFZmpP3ji)**  
All images and labels are available for direct use by the training and evaluation teams.

## 📂 Project Structure
- **data/** → contains dataset and annotations
- **src/** → preprocessing, filtering, and augmentation scripts
- **reports/** → weekly reports and documentation
- **models/** → (to be added later)
- **requirements.txt** → project dependencies
- **.gitignore** → excluded large files and cache

### 📊 Dataset Summary
| Process | Result |
|----------|--------|
| Total Images (original) | 14,200 |
| After Cleaning | 11,670 |
| After Augmentation | ~44,000 |
| Train/Val Split | 80% / 20% |
| Ready For Model Training | ✅ Yes |



## 📖 Abstract

This project implements a sophisticated **Arabic Hand Sign Recognition System** that accurately classifies 32 different Arabic letter gestures. The system combines traditional machine learning with deep learning approaches, achieving exceptional performance through advanced feature engineering and ensemble methods. The solution includes real-time webcam processing, image upload capabilities, and a user-friendly web interface.

## 🚀 Key Features

- 🎥 **Real-time Gesture Recognition** - Live webcam processing with instant predictions
- 📁 **Multiple Input Modes** - Webcam streaming and image upload support  
- 🔧 **Advanced Feature Engineering** - 94-dimensional feature vectors combining geometric and spatial characteristics
- 🤝 **Ensemble Learning** - Combines SVM, Random Forest, and Neural Networks for robust performance
- 🎨 **Professional Preprocessing** - Automated background removal and quality filtering
- 📊 **Comprehensive Evaluation** - Detailed performance analysis with confusion matrices and per-class metrics

## 🏗️ System Architecture

### 🔄 Multi-Model Approach

| Model | Architecture | Features | Key Components |
|:-----:|:------------:|:--------:|:---------------|
| 🛠️ **Basic Enhanced** | SVM | 69 | Hand landmarks + Geometric features |
| 🚀 **Advanced Ensemble** | Voting Classifier | 94 → 80 selected | Enhanced geometric + Curvature + Convexity |
| 🧠 **Deep Learning** | MobileNetV2 | Transfer Learning | CNN features + Fine-tuning |

Processing Pipeline
text
Raw Images → Background Removal → Quality Filtering → Feature Extraction → Classification → Results

## 🏆 Model Performance Comparison

### 📊 Performance Summary

| Model | Test Accuracy | Features Used | Classes | Key Innovations |
|-------|---------------|---------------|---------|-----------------|
| 🟢 **Basic Enhanced Model** | **90.24%** | 69 Features | 32 | Hand landmarks + Basic geometric features |
| 🚀 **Advanced Ensemble Model** | **99.85%** | 94 → 80 Selected Features | 32 | Enhanced geometric + Curvature + Convexity features |


## 🛠️ Technologies Used

### 💻 Core Technologies

- **Python 3.11+** - Primary programming language
- **OpenCV** - Image processing and computer vision
- **MediaPipe** - Hand landmark detection
- **Scikit-learn** - Machine learning algorithms and evaluation
- **PyTorch** - Deep learning framework
- **Streamlit** - Web application deployment

### 🤖 Machine Learning Models

- **Support Vector Machines (SVM)** - RBF kernel with probability estimates
- **Random Forest** - 200 estimators with balanced class weights
- **Multi-Layer Perceptron (MLP)** - 128-64 hidden architecture
- **MobileNetV2** - Transfer learning with fine-tuning
- **Voting Classifier** - Soft voting ensemble

### 🔧 Feature Engineering

- **Basic Landmarks**: 63 features (21 landmarks × 3 coordinates)
- **Enhanced Features**: 6 geometric measurements (finger lengths, palm size)
- **Advanced Features**: 25 sophisticated features:
  - Relative finger lengths (scale-invariant)
  - Finger curvature calculations
  - Hand convexity and compactness
  - Inter-finger distances
  - Palm-center to fingertip distances
python
## 🔧 Geometric Features & Algorithms

### 📐 Geometric Feature Extraction

- **📏 Euclidean distances** between key points
- **📊 Finger length ratios** (scale invariant)
- **🔄 Curvature measurements** using cross products
- **🔵 Convex hull analysis** for hand shape
- **🎯 Spatial relationships** between fingers

### 🤝 Ensemble Learning Strategy

```python
VotingClassifier([
    ('svm', SVC(C=10, kernel='rbf', probability=True)),
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64)))
], voting='soft')
```


## 🔄 Data Preprocessing Pipeline


### 🎨 Background Removal
- **Improved GrabCut** with multi-method initialization
- Skin color detection + geometric priors
- Professional white background application

### ✅ Quality Filtering
- **10-point quality assessment system**
- Blur, contrast, and composition analysis
- Automated rejection of poor-quality images

### 🔄 Data Augmentation
- **Targeted augmentation** for weak classes
- Gaussian noise injection for underrepresented gestures
- Balanced dataset generation

**🎯 Feature Selection**
- **SelectKBest** with ANOVA F-value
- **80/94 features** selected for optimal performance
- Reduced overfitting + improved generalization


## 🌐 Deployment

### 🚀 Streamlit Web Application

The system is deployed as an interactive web application with real-time processing capabilities.

#### 🎥 Real-time Webcam Mode

- **Live hand tracking** with landmark visualization
- **Confidence threshold adjustment** (0.0-1.0)
- **Real-time prediction statistics**
- **Most common sign tracking**
- **Mirror view** for intuitive interaction

#### 📁 Upload Image Mode

- **Drag-and-drop image upload**
- **Side-by-side original/processed comparison**
- **Detailed confidence scoring**
- **Professional result display**
- **Batch processing support**
