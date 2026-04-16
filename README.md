# Accident Severity Prediction

### Explainable Accident Severity Prediction Using Deep Learning and Large Language Models

---

## Overview

This project predicts road accident severity (**Slight, Serious, Fatal**) using Machine Learning and Deep Learning models.

It combines **predictive modelling, explainable AI (SHAP), and Large Language Models (LLMs)** to generate both accurate predictions and human-readable explanations for decision support in road safety analysis.

---

## Objectives

* Predict accident severity using multiple ML & DL models
* Compare classical models with deep learning performance
* Apply **SHAP** for model interpretability
* Use **LLMs** to generate natural-language explanations
* Build an interactive **Streamlit dashboard** for real-time predictions

---

## Key Features

* Multi-model prediction (Logistic Regression, Random Forest, XGBoost, Deep Learning)
* Dimensionality reduction using **Truncated SVD**
* SHAP-based explainability
* LLM-generated narrative insights
* Interactive Streamlit web application
* Modular and reproducible pipeline

---

## Tech Stack

**Languages & Frameworks**

* Python
* Scikit-learn
* XGBoost
* TensorFlow / Keras

**Data Processing**

* Pandas
* NumPy
* Scikit-learn Pipelines
* TruncatedSVD

**Visualization & Explainability**

* Matplotlib
* SHAP

**Deployment**

* Streamlit

---

## Model Performance

* **XGBoost** achieved the highest accuracy on tabular data
* **Deep Learning** captured complex non-linear relationships and performed competitively
* Logistic Regression provided a baseline for comparison

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/keerthikaskrishnan/AccidentSeverityProject.git
cd AccidentSeverityProject
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## ⚙️ Environment Variables *(Optional)*

Create a `.env` file if required:

```env
MODEL_PATH=models/
DATA_PATH=data/
```

---

## Usage

1. Launch the Streamlit app
2. Select a model (Logistic Regression, Random Forest, XGBoost, Deep Learning)
3. Enter accident-related features
4. Click **Predict Severity**
5. View prediction results and explanations


---

## Project Structure

```
ACCIDENTSEVERITYPROJECT/
│── data/
│── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── dl_model.keras
│   └── figures/
│── src/
│   ├── preprocess.py
│   ├── preprocess_svd.py
│   ├── train_models.py
│   ├── train_dl.py
│   ├── evaluation_full.py
│── utils/
│── generate_shap_plots.py
│── app.py
│── README.md
```

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**

---

## Authors

**Keerthika Santhanakrishnan**

* GitHub: https://github.com/keerthikaskrishnan
* Email: [keerthikaskrishnan@gmail.com]
  
**Kishorekumar Dhanabalan**
* Email:"kishorekumard505@gmail.com"
  
**Kabilan Ponnusamy**
* Email:"kabilanponnusamy14@gmail.com"

**Abrarudin Zahiruddin**
* Email:"itsabrar0301@gmail.com"
---

##  Acknowledgements

* UK Department for Transport – Road Accident Dataset
* SHAP Explainability Framework
* Streamlit for interactive application development

---

##  Project Highlights

* Combines **Deep Learning + Explainable AI + LLMs**
* End-to-end pipeline from preprocessing to deployment
* Focus on **real-world impact in road safety analytics**

---
