# A Cross-Cultural, Multilingual & Edge-AI System for Mental and Emotion Disorder Prediction

## 📌 Project Overview

This project is an advanced AI-powered system designed to predict **mental and emotional disorders** using **multimodal data**. The system is built to work across **different cultures, languages, and devices**, including **edge devices** for real-time, privacy-preserving predictions.

The system combines **Natural Language Processing (NLP)**, **Deep Learning** methods to deliver highly accurate mental health insights.

---

## 🚀 Key Features

### 🧠 Mental Health Prediction

* Detects disorders such as:

  * Depression
  * Anxiety
  * ADHD
  * Bipolar Disorder
  * PTSD
  * Schizophrenia
  * Autism Spectrum Disorder
  * Borderline Personality Disorder (BPD)

### 🌍 Cross-Cultural Intelligence

* Trained on **multi-region datasets** to reduce bias
* Supports users from different cultural and social backgrounds
* Adaptive models to handle cultural variations in language and behavior

### 🌐 Multilingual Support

* Supports multiple languages for text and perform binary classification
* Automatic language detection
* Handles:

  * English
  * Hindi/English
  * Russian
  * Portuguese
  * Spanish
  * Arabic
  * Bengali
    

### 📊 Multimodal Input System

Accepts multiple input types:

* 📝 Text (journals, social media posts, chat messages in different languages)
* 🎙️ Audio (voice recordings) .mp3/.wav
* 🧠 EEG Signals (brainwave data) .mat format
* 🗺️ Spatial Data (movement, risk analysis) using FIPS code 

### 🧩 Deep Learning Models

Uses advanced AI models such as:

* **DistilBERT / mBERT** for NLP-based text classification
* **CNN** for pattern extraction from EEG and spectrograms
* **GRU (Gated Recurrent Units)** for sequential time-series modeling
* **Hybrid CNN-GRU architectures**

### ⚡ Edge-AI Capabilities

* Lightweight model versions for deployment on:

  * Mobile devices
  * Offline environments


### 🔐 Privacy-Focused Design

* Local inference without sending sensitive data to the cloud
* Encrypted data storage
* GDPR-compliant architecture

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI / Flask
* PyTorch / TensorFlow
* HuggingFace Transformers
* Scikit-learn

### Frontend

* React.js
* Bootstrap
* Recharts / Chart.js for visualizations

### AI/ML Tools

* HuggingFace Transformers
* Librosa (Audio Processing)
* MNE / SciPy (EEG Processing)
* OpenCV (for visual processing, if used)

---

## 📊 System Architecture

```
User Input (Text / Audio / EEG / Spatial)
            ↓
   Data Preprocessing & Cleaning
            ↓
    Feature Extraction (MFCC / PSD / Tokenization)
            ↓
       Deep Learning Models (CNN / GRU / BERT)
            ↓
    Risk Scoring & Classification Engine
            ↓
       Visualization Dashboard (Graphs & Reports)
```

---

## 📂 Project Structure

```
project-root/
│
├── frontend/           # React UI
├── backend/            # Python FastAPI
├── checkpoints/        # Loaded checkpoints of individual models
├── venv/               # Python virtual environment
├── notebooks/          # Jupyter experiments
├── utils/              # Helper functions
└── README.md           # Documentation
```

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

## ✅ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2️⃣ Backend Setup

```bash
pip install -r requirements.txt
python backend.py
```

### 3️⃣ Frontend Setup

```bash
npm install
npm start
```

---

## 📌 Future Enhancements

* Real-time EEG device integration
* Multilingual speech-to-text pipeline
* Cloud + Edge hybrid inference system
* Mobile application deployment

---

## 👨‍💻 Author

**Tanishq Palkhe**

MCA Student | AI/ML & Full-Stack Developer

---

## ⚠️ Disclaimer

This system is designed for **research and educational purposes only**. It is not a certified medical diagnostic tool.
