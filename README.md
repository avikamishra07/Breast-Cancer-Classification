# Breast Cancer Classification

## Project Overview

This project uses machine learning classification techniques to predict whether a breast tumor is **Benign** or **Malignant** using the Breast Cancer Wisconsin dataset.

The project includes data preprocessing, exploratory data analysis, correlation and covariance analysis, model training, model comparison, saving the best model using Joblib, and a prediction interface.

## Dataset

The dataset used is the **Breast Cancer Wisconsin dataset**.

### Target Variable

* `B` → Benign
* `M` → Malignant

The dataset contains tumor-related numerical features such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

## Machine Learning Models

The following classification models were implemented:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

## Model Evaluation

The models were compared using:

* Accuracy
* Precision
* Recall
* F1 Score

The model with the highest accuracy was selected as the best model.

## Data Preprocessing

The project includes:

* Missing value handling
* Duplicate record checking and removal
* Identification of numerical and categorical variables
* Conversion of the target variable into numerical form
* Feature scaling where required

## Exploratory Data Analysis

The project includes:

* Distribution plots
* Box plots
* Scatter plots
* Correlation analysis
* Covariance analysis
* Correlation heatmap

## Saved Model Files

The following files are generated using Joblib:

```text
best_breast_cancer_model.pkl
breast_cancer_scaler.pkl
breast_cancer_features.pkl
breast_cancer_model_name.pkl
```

## Gradio Interface

A Gradio interface is created to enter the breast tumor features and predict whether the tumor is:

**Benign** or **Malignant**

## Streamlit Application

A Streamlit application is also created for the prediction interface.

Run the application using:

```bash
python -m streamlit run app.py
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Gradio
* Streamlit

## Project Structure

```text
Breast-Cancer-Classification/
│
├── app.py
├── best_breast_cancer_model.pkl
├── breast_cancer_scaler.pkl
├── breast_cancer_features.pkl
├── breast_cancer_model_name.pkl
├── README.md
└── .gitignore
```

## Author

**Avika Mishra**
