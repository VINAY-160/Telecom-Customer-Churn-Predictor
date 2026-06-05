# 📶 Telecom Customer Churn Prediction using Machine Learning

Predict customer churn before it happens using an advanced Machine Learning ensemble model. This project helps telecom companies identify high-risk customers and take proactive retention actions.

## 🚀 Live Demo

**Streamlit Application:**
https://telecom-customer-churn-predictor-main.streamlit.app/

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. Acquiring a new customer is significantly more expensive than retaining an existing one.

This project uses Machine Learning to predict whether a customer is likely to leave the telecom service based on demographic information, account details, contract type, billing preferences, and subscribed services.

The solution includes:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Ensemble Learning
* Hyperparameter Tuning
* Stacking Classifier
* Model Serialization
* Streamlit Deployment

---

## 🎯 Business Problem

Telecom companies lose revenue when customers discontinue their services.

The objective of this project is to:

* Identify customers likely to churn
* Estimate churn probability
* Help businesses target retention campaigns
* Reduce customer acquisition costs
* Improve customer lifetime value

---

## 📊 Dataset

The dataset contains customer information including:

### Customer Information

* Gender
* Senior Citizen Status
* Partner
* Dependents

### Account Information

* Tenure
* Contract Type
* Paperless Billing
* Payment Method

### Services

* Phone Service
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### Financial Information

* Monthly Charges
* Total Charges

### Target Variable

```text
Churn
0 = Customer Stays
1 = Customer Leaves
```

---

## 🔧 Feature Engineering

The dataset was transformed using:

### Binary Encoding

* Gender
* Partner
* Dependents
* Phone Service
* Paperless Billing

### One-Hot Encoding

* Contract Type
* Internet Service
* Payment Method
* Additional Services

### Feature Selection

Top predictive features were selected before model training to improve performance and reduce noise.

---

## 🤖 Machine Learning Models Evaluated

The following classification algorithms were trained and compared:

### Baseline Models

* Logistic Regression
* Decision Tree Classifier

### Ensemble Models

* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

### Final Production Model

✅ Stacking Classifier

The final model combines:

* XGBoost
* LightGBM
* Random Forest

using Logistic Regression as the meta learner.

---

## 📈 Model Performance

Evaluation Metrics Used:

* Accuracy
* F1 Score
* ROC-AUC Score
* Precision
* Recall

### Final Model

| Metric     | Value             |
| ---------- | ----------------- |
| Accuracy   | 86.00%            |
| Model Type | Stacking Ensemble |
| Threshold  | 0.40              |

---

## 🖥️ Streamlit Application Features

### Single Customer Analysis

Analyze churn probability for an individual customer.

Features:

* Customer profile input
* Churn probability prediction
* Risk assessment
* Business recommendations

### Batch Prediction

Upload a CSV file and:

* Score multiple customers
* Calculate churn probability
* Generate risk categories
* Download prediction reports

---

## 🏗️ Project Structure

```text
Customer-Churn-Telecom-Predictor/
│
├── app.py
├── README.md
├── requirements.txt
│
├── best_churn_model.pkl
├── telecom_india_churn.csv
│
└── Telecom_Customer_Churn_Prediction.ipynb
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/VINAY-160/Telecom-Customer-Churn-Predictor.git
```

### Navigate to Project

```bash
cd Telecom-Customer-Churn-Predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Data Science Libraries

* Pandas
* NumPy
* Scikit-Learn

### Machine Learning

* Random Forest
* XGBoost
* LightGBM
* Stacking Classifier

### Deployment

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📚 Key Learnings

Through this project I learned:

* Customer churn analysis
* Classification algorithms
* Ensemble learning
* Hyperparameter tuning
* Model evaluation
* Threshold optimization
* Model deployment
* Interactive dashboard development

---

## 🔮 Future Improvements

* SHAP Explainability
* Real-time API Integration
* Automated Feature Engineering
* Customer Segmentation Dashboard
* Retention Strategy Recommendations
* Cloud-Based Batch Processing

---

## 👨‍💻 Author

### Vinay Mishra

Integrated MCA Student
Machine Learning & Data Science Enthusiast

GitHub: https://github.com/VINAY-160

---

## ⭐ Support

If you found this project useful, please consider giving it a star.
