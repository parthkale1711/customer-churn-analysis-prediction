## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed the `customerID` identifier column
- Converted the target variable `Churn` into an encoded target
- Separated features (`X`) and target (`y`)
- Split the dataset into training and testing sets using an 80/20 split
- Used stratified sampling to maintain the churn class distribution
- Applied `StandardScaler` to numerical features
- Applied `OneHotEncoder` to categorical features

The dataset contained 19 input features before preprocessing.

After one-hot encoding the categorical variables, the feature space increased to 45 machine-learning features.

## 🤖 Machine Learning Models

Three classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## 📈 Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **80.38%** | **64.85%** | 57.22% | 60.80% | **83.59%** |
| Decision Tree | 78.96% | 60.21% | **61.50%** | **60.85%** | 82.96% |
| Random Forest | 73.92% | 50.62% | **76.20%** | 60.83% | 82.98% |

## 🏆 Final Model

**Logistic Regression** was selected as the final model based on its overall performance.

### Final Model Performance

- **Accuracy:** 80.38%
- **Precision:** 64.85%
- **Recall:** 57.22%
- **F1-score:** 60.80%
- **ROC-AUC:** 83.59%

Logistic Regression achieved the highest accuracy, precision, and ROC-AUC among the three evaluated models.

Random Forest achieved the highest recall, which can be useful when the primary objective is to identify as many potential churners as possible.

## 🔍 Feature Importance

Logistic Regression coefficients were analyzed to identify the features that had the strongest influence on the model's predictions.

Both positive and negative coefficients were considered to understand the direction and strength of the relationship between features and predicted churn.

## 💾 Saved Model

The final Logistic Regression model and preprocessing pipeline were saved using Joblib.

```text
models/
├── logistic_regression_model.pkl
└── preprocessor.pkl
