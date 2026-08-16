# Customer Churn Analysis & Prediction

An end-to-end machine learning project that analyzes customer behavior,
identifies factors associated with customer churn, and builds machine
learning models to predict customers who are likely to churn.

## 📌 Project Overview

Customer churn is an important business problem because losing existing
customers can directly affect revenue and business growth.

This project uses the Telco Customer Churn dataset to:

- Explore customer demographics and service-related patterns
- Clean and preprocess customer data
- Perform exploratory data analysis (EDA)
- Train machine learning models
- Handle class imbalance
- Perform hyperparameter tuning using GridSearchCV
- Evaluate models using multiple classification metrics
- Select and document the best-performing approach

## 🎯 Problem Statement

The objective is to predict whether a customer is likely to churn based
on customer information such as demographic characteristics, services,
contract details, payment methods, and account information.

This can help businesses identify customers who may be at higher risk
of leaving and support targeted retention strategies.

## 📊 Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information related to:

- Customer demographics
- Account information
- Internet and phone services
- Contract details
- Payment methods
- Monthly and total charges
- Customer churn status

Raw and processed datasets are kept outside the Git repository where
appropriate.

## 🛠️ Technologies Used

### Programming
- Python

### Data Analysis
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- SMOTE
- GridSearchCV

### Development Tools
- Jupyter Notebook
- VS Code
- Git
- GitHub

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Baseline Model
   ↓
Hyperparameter Tuning
   ↓
Class Imbalance Handling
   ↓
Model Evaluation
   ↓
Best Model Selection

