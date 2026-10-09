# Customer Churn Analysis & Prediction

## 📌 Project Overview

Customer churn refers to customers discontinuing their services with a company. Predicting customer churn helps businesses identify customers who may leave and develop strategies to improve customer retention.

This project uses **Python, Exploratory Data Analysis (EDA), and Machine Learning** to analyze customer behavior and predict whether a customer is likely to churn.

The project uses the Telco Customer Churn dataset to explore customer characteristics, identify patterns associated with churn, and compare machine learning classification models.

## 🎯 Project Objectives

- Analyze customer demographics, account information, and service usage.
- Identify patterns and factors associated with customer churn.
- Clean and preprocess the dataset for machine learning.
- Train and compare classification models.
- Evaluate model performance using classification metrics.
- Demonstrate how predictive analytics can support customer retention strategies.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Development Environment:** VS Code, Jupyter Notebook

## 📊 Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer information, including demographics, subscribed services, contract details, payment methods, monthly charges, and churn status.

- **Original dataset size:** 7,043 rows and 21 columns.
- **Target variable:** `Churn`
- **Prediction task:** Binary classification — predict whether a customer will churn.

The project preprocessing removes the customer identifier and handles missing or invalid values before model training.

*Note: Add the dataset source and license information if you obtained it from an external provider.*

## ⚙️ Project Workflow

### 1. Data Loading
Load the Telco Customer Churn dataset using Pandas.

### 2. Data Cleaning
- Inspect missing values and duplicate records.
- Convert columns to appropriate data types.
- Handle invalid or missing values.
- Remove unnecessary identifier columns.

### 3. Exploratory Data Analysis (EDA)
Explore customer demographics, contract types, service usage, payment methods, and churn distribution to identify useful patterns.

### 4. Data Preprocessing
- Separate input features and the target variable.
- Encode categorical variables.
- Prepare training and testing datasets.
- Apply the same preprocessing logic consistently to model inputs.

### 5. Model Training
Train and compare the following classification algorithms:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### 6. Model Evaluation
Evaluate the models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score

### 7. Model Selection
Compare model performance and select a suitable model based on the evaluation results and business requirements.

## 🤖 Model Performance

The following results were obtained during the project evaluation:

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| Accuracy | 80.38% | 78.96% | 73.92% |
| Precision | 64.85% | 60.21% | 50.62% |
| Recall | 57.22% | 61.50% | 76.20% |
| F1-score | 60.80% | 60.85% | 60.83% |
| ROC-AUC | 83.59% | 82.96% | 82.98% |

*These metrics are from the project's evaluated models and test split. Results may vary with changes to the data, preprocessing, or model configuration.*

### Key Observations

- **Logistic Regression** achieved the highest accuracy and ROC-AUC among the evaluated models.
- **Random Forest** achieved the highest recall, making it more likely to identify customers who churned in this evaluation.
- **Decision Tree** provided a comparable F1-score to the other evaluated models.

The best model depends on the business objective. If missing potential churners is costly, recall may be particularly important.

## 📁 Project Structure

```text
customer-churn-analysis-prediction/
│
├── data/
│   └── raw/
│       └── telco.csv
│
├── models/
│   └── model_class_weight.joblib
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── reports/
│
├── scripts/
│
├── src/
│
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

*Folder and filename examples should be adjusted to match the actual repository contents.*

## 🚀 Installation and Setup

### Prerequisites

- Python installed
- Git installed
- VS Code or Jupyter Notebook

### 1. Clone the repository

```bash
git clone https://github.com/parthkale1711/customer-churn-analysis-prediction.git
```

### 2. Navigate to the project directory

```bash
cd customer-churn-analysis-prediction
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the analysis

Open the project notebook in VS Code or Jupyter Notebook and execute the cells in order.

If your repository includes a Python script for training or analysis, use the appropriate script from the `scripts/` directory.

## 💡 Business Applications

This project demonstrates how churn prediction can help businesses:

- Identify customers who may discontinue their services.
- Understand patterns associated with customer churn.
- Prioritize customer retention efforts.
- Support data-driven business decisions.
- Evaluate customer retention strategies.

Predictions should be used as decision-support information, not as a guarantee that a customer will leave.

## 📚 Key Learnings

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature encoding and preparation
- Classification model training
- Handling class imbalance
- Model evaluation and comparison
- Saving trained models with Joblib
- Applying machine learning to a business problem

## 🔮 Future Improvements

- Deploy an interactive prediction application using Streamlit.
- Improve model performance through hyperparameter tuning.
- Add explainability techniques to understand feature importance.
- Monitor model performance on new customer data.
- Integrate customer retention recommendations into a dashboard.

## 👨‍💻 Author

**Parth Kale**

- GitHub: [parthkale1711](https://github.com/parthkale1711)
- LinkedIn: [Parth Kale](https://linkedin.com/in/parth-kale-a3091b290)

## 📄 License

This project is developed for educational and portfolio purposes. Add an appropriate open-source license if you intend to permit others to reuse or distribute the code.
