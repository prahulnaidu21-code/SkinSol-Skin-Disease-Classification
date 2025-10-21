# 🩺 Skin Disease Detection and Diagnosis (EfficientNet+ Gemini AI)
A Flask-based deep learning web app that uses a  EfficientNet to  detect and classify skin diseases from images.It allows users to upload or capture photos, predicts the disease with confidence, and provides an AI-generated medical explanation using Google Gemini API.


---

## 🚀 Features

- 🖼️ Upload or capture skin images directly from the browser  
- 🧠 Classify among **12 common dermatological conditions**  
- 🔗 **EfficientNetB0**  
- 💬 **Gemini API** generates short medical diagnosis summaries  
- 🌐 Simple, responsive web interface using **Flask, HTML, CSS, and JS**  
- ⚡ Optimized for GPU-based training and inference  

---

## 🧠 Model Overview

### **Architecture**
- **EfficientNetB0** → Captures fine-grained color and texture details  

### **Input**
- RGB Image — 256×256 pixels  
- Normalized to [0, 1]

### **Output**
- Softmax probabilities for 12 disease classes  
- Displays the top-1 predicted disease and confidence percentage  

---

## 🩹 Supported Diseases

| # | Disease Class |
|---|----------------|
| 1 | Warts & Viral Infections |
| 2 | Eczema |
| 3 | Melanoma |
| 4 | Atopic Dermatitis |
| 5 | Basal Cell Carcinoma |
| 6 | Melanocytic Nevi |
| 7 | Benign Keratosis |
| 8 | Psoriasis / Lichen Planus |
| 9 | Seborrheic Keratoses |
| 10 | Tinea (Fungal Infections) |
| 11 | Acne |
| 12 | Vitiligo |

---

## ⚙️ Installation and Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection
