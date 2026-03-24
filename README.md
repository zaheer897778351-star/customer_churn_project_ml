# Customer Churn Prediction
This project predicts whether a customer will churn using machine learning models. It includes data preprocessing, feature engineering, model training, and evaluation.

## Problem Statement

Customer churn is a major challenge for businesses. Retaining existing customers is often more cost-effective than acquiring new ones. 

The goal of this project is to build a machine learning model that can predict whether a customer is likely to leave (churn) based on their historical data.

## Dataset

- Source: [https://huggingface.co/datasets/muqsith123/telco-customer-churn]
- Number of rows: 7000
- Number of features: 49

## Dataset Features

The dataset contains **49 features** describing customer demographics, services, and account details.

### 🔹 Demographic Information

* Age
* Gender
* Married
* Dependents / Number of Dependents
* Senior Citizen
* Under 30

### 🔹 Location Information

* City
* State
* Country
* Zip Code
* Latitude / Longitude

### 🔹 Account & Contract Details

* Customer ID
* Contract
* Tenure in Months
* Payment Method
* Paperless Billing
* Monthly Charges
* Total Charges

### 🔹 Services Subscribed

* Phone Service
* Multiple Lines
* Internet Service / Internet Type
* Online Security
* Online Backup
* Device Protection Plan
* Streaming TV / Movies / Music
* Unlimited Data
* Premium Tech Support

### 🔹 Usage & Financial Metrics

* Avg Monthly GB Download
* Avg Monthly Long Distance Charges
* Total Revenue
* Total Refunds
* Total Extra Data Charges
* Total Long Distance Charges
* CLTV (Customer Lifetime Value)

### 🔹 Customer Engagement

* Number of Referrals
* Referred a Friend
* Offer
* Satisfaction Score

### 🔹 Target Variables

* Churn Label (Target)
* Churn Value
* Churn Category
* Churn Reason
* Churn Score


## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Selection
6. Model Training
7. Model Evaluation
8. Deployment (if done)

## Data Preprocessing

- Handled missing values
- Encoded categorical variables
- Feature scaling using StandardScaler
- Removed highly correlated features

## Models Used

- Random Forest
- Gradient Boosting

## Model Performance

| Model              | Accuracy |
|-------------------|----------|
| LogisticRegression | 96.63%  |
| DecisionTree       | 96.92%  |
| RandomForest       | 97.91%  |
| GradientBoosting   | 98.10%  |
| AdaBoost           | 97.53%  |
| SVM (RBF)          | 96.68%  |

Best Model: **Gradient Boosting**

## How to Run

1. Clone the repository
```bash
git clone https://github.com/zaheer897778351-star/customer_churn_project_ml.git
cd customer_churn_project_ml

2. Install dependencies
pip install -r requirements.txt

3.Run the project
python main.py


## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- flask

## Future Improvements

- Hyperparameter tuning
- Use deep learning models
- Deploy using Streamlit / Flask
- Add real-time prediction

## Author

Zaheer Ahmad
GitHub: https://github.com/zaheer8977783531-star