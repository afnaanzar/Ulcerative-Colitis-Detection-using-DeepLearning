# Ulcerative Colitis Detection using Deep Learning
A CNN-based deep learning web application that detects Ulcerative Colitis from endoscopic images, built entirely from scratch and deployed as an interactive Flask web app.

## Overview
Ulcerative Colitis (UC) is a chronic inflammatory bowel disease typically diagnosed through colonoscopy. Since colonoscopy is a type of endoscopic imaging, this project uses endoscopic image data to train a binary classification model that distinguishes between **Ulcerative Colitis** and **Normal** cases.

## Model Details
- **Architecture:** Custom Convolutional Neural Network (CNN), built from scratch
- **Task:** Binary classification (Ulcerative Colitis vs. Normal)
- **Output Activation:** Sigmoid
-  **Hidden Layer Activation:** ReLU
- **Dataset:** Endoscopic image dataset (sourced from Kaggle)

## Performance

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.90  |
| Precision | 0.87  |
| Recall    | 0.94  |
The high recall indicates the model is effective at correctly identifying actual Ulcerative Colitis cases, which is critical in a medical screening context.

## Web Application
A Flask-based web app (`app.py`) allows users to upload an endoscopic image and receive an instant prediction along with the model's confidence score.

## Demo

**Image Upload Interface:**
![Upload Page](Screenshot%20uploadpage.png)

**Ulcerative Colitis Detected:**
![UC Prediction](screenshot%20uc.png)

**Normal Case:**
![Normal Prediction](Screenshot%20normal.png)

## Tech Stack
- Python
- PyTorch (CNN model)
- Flask (web app backend)
- HTML/CSS (frontend)

## Project Structure
├── app.py              # Flask web application
├── train_model.py      # Model training script
├── predict.py          # Prediction script
├── requirements.txt    # Project dependencies
├── templates/          # HTML templates
├── static/             # CSS, images, uploads

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train_model.py`
4. Run predictions (optional, for testing): `python predict.py`
5. Launch the web app: `python app.py`
6. The app will open in your browser — upload an endoscopic image to get an instant prediction (Ulcerative Colitis or Normal)

## Disclaimer
This project is for academic and educational purposes only and is not intended for clinical or diagnostic use.
