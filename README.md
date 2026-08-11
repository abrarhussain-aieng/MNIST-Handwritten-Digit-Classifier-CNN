# 🔢 MNIST Handwritten Digit Classifier — CNN

An end-to-end Deep Learning project that recognizes handwritten digits from images using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.

The project covers the complete workflow:

**Data Loading → Preprocessing → Model Training → Model Comparison → CNN Evaluation → Model Saving → Streamlit Deployment → Real-Time Image Prediction**

🌐 **Live Demo:**  
https://mnist-handwritten-digit-classifier-cnn-ovb49v3whqcuopc4tvzbrc.streamlit.app/

---

## 📌 Project Overview

Handwritten digit recognition is a classic computer vision problem and an excellent introduction to Convolutional Neural Networks.

In this project, three different neural-network approaches are developed and compared:

- **Perceptron**
- **Artificial Neural Network (ANN)**
- **Convolutional Neural Network (CNN)**

The CNN is selected as the final model because it is specifically designed to learn spatial patterns in images.

The trained CNN is then integrated into a **Streamlit web application**, where users can upload an image of a handwritten digit and receive:

- Predicted digit
- Prediction confidence
- Probability distribution across all 10 digit classes
- Processed image sent to the CNN

---

## 🚀 Live Demo

Try the deployed application:

👉 **[MNIST Handwritten Digit Classifier](https://mnist-handwritten-digit-classifier-cnn-ovb49v3whqcuopc4tvzbrc.streamlit.app/)**

Upload an image containing a handwritten digit and click **Predict Digit**.

---

## ✨ Features

### 🧠 Deep Learning

- MNIST handwritten digit classification
- CNN-based image classification
- 10 digit classes: `0–9`
- Softmax probability output

### 🖼️ Image Processing

The Streamlit application preprocesses uploaded images before sending them to the CNN:

1. Convert image to grayscale
2. Detect image background
3. Invert image when necessary
4. Improve contrast
5. Threshold the image
6. Detect and crop the digit
7. Add padding around the digit
8. Center the digit
9. Resize to `28 × 28`
10. Normalize pixel values to `[0, 1]`
11. Reshape for CNN input

This makes uploaded images more compatible with the MNIST-style input expected by the trained CNN.

### 📊 Prediction

The application displays:

- Uploaded image
- Processed image
- Predicted digit
- Confidence score
- Probability for every digit from `0` to `9`

---

# 🏗️ Project Architecture

```text
                    ┌──────────────────────┐
                    │     MNIST Dataset    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Preprocessing │
                    │                      │
                    │  • Normalize pixels  │
                    │  • Reshape images    │
                    │  • One-hot labels    │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │ Perceptron│ │    ANN     │ │    CNN     │
          └────────────┘ └────────────┘ └─────┬──────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Model Evaluation │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ mnist_cnn.keras  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Streamlit App    │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                      🖼️ Upload Image
                                             │
                                             ▼
                                    🔄 Preprocessing
                                             │
                                             ▼
                                       🧠 CNN Model
                                             │
                                             ▼
                                     🎯 Prediction
