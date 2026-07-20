# ML-DL-Capstone-Project

# HHAR Human Activity Recognition Classifier

A Human Activity Recognition (HHAR) system built using a **1D Convolutional Neural Network (CNN)** to classify human activities from smartphone and smartwatch sensor data. The project demonstrates machine learning deployment using **FastAPI** as the backend inference service and **Streamlit** as the interactive frontend.

## Project Structure

```
ML-DL-Capstone-Project/
│
├── .python-version          # Python version for deployment
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
│
├── app.py                   # FastAPI Backend
├── main.py                  # Streamlit Frontend
│
├── best_hhar.keras          # Trained CNN Model
├── scaler.pkl               # StandardScaler
├── encoder.pkl              # Label Encoder
│
├── MLDL_Capstone.ipynb      # Model training notebook
├── HHAR_CNN.keras           # Saved training model/checkpoint
```

---

## Model Information

The model is a **1D Convolutional Neural Network (CNN)** trained on the HHAR (Heterogeneity Human Activity Recognition) dataset.

### Input Shape

```
(128, 12)
```

where

- **128** → Time steps
- **12** → Sensor features

The 12 features correspond to:

- Phone Accelerometer (X, Y, Z)
- Phone Gyroscope (X, Y, Z)
- Watch Accelerometer (X, Y, Z)
- Watch Gyroscope (X, Y, Z)

---

## Supported Activities

- Walking
- Sitting
- Standing
- Biking
- Upstairs
- Downstairs

---

## Deployment

### Backend

- FastAPI
- Uvicorn
- TensorFlow
- Render

### Frontend

- Streamlit
- Streamlit Community Cloud

---

## Modules

- Python
- TensorFlow / Keras
- FastAPI
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Requests
- Uvicorn

---

## Model Workflow

1. Receive a sensor window of shape **(128, 12)**.
2. Normalize features using the saved **StandardScaler**.
3. Pass the normalized window through the trained CNN model.
4. Compute prediction probabilities.
5. Decode the predicted class using the saved **LabelEncoder**.
6. Return the predicted activity, confidence score, and class probabilities.

---

## Author

**Pranav S Gaonkar**
