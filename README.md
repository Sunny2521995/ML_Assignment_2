# Machine Learning Assignment 2 - Bank Marketing Classification

## a. Problem statement
The objective of this assignment is to build, evaluate, and deploy multiple classification models to predict whether a bank client will subscribe to a term deposit (binary classification: 'yes' or 'no') based on demographic, campaign, and economic attributes.

## b. Dataset description
- **Dataset Chosen:** Bank Marketing Dataset (UCI Repository, ID: 222)
- **Features Size:** 15 input features (mix of numerical and categorical variables such as age, job, marital status, education, balance, housing, loan, contact, campaign, pdays, previous, etc.)
- **Instance Size:** 45,211 total instances (split into training and testing sets)
- **Target Variable:** `y` (Binary indicator: 1 for subscription 'yes', 0 for 'no')

## c. GitHub Repository Link
[Insert your public GitHub repository link here]

## d. Models Used & Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8918 | 0.7735 | 0.6351 | 0.1777 | 0.2777 | 0.2965 |
| **Decision Tree** | 0.8269 | 0.6146 | 0.2924 | 0.3374 | 0.3133 | 0.2155 |
| **KNN** | 0.8841 | 0.6918 | 0.5116 | 0.2079 | 0.2957 | 0.2743 |
| **Naive Bayes** | 0.8442 | 0.7564 | 0.3702 | 0.4716 | 0.4148 | 0.3296 |
| **Support Vector Machine** | 0.8921 | 0.7695 | 0.6370 | 0.1824 | 0.2836 | 0.3012 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved high overall accuracy (89.18%) and strong precision (63.51%), but suffered from low recall on the minority class due to dataset class imbalance. |
| **Decision Tree** | Provided balanced recall (33.74%) for the positive class, but had lower overall accuracy and precision, leading to a weaker MCC score. |
| **KNN** | Performed competitively with an accuracy of 88.41%, showing moderate precision and lower recall on unseen test samples. |
| **Naive Bayes** | Achieved the highest recall (47.16%) and best F1-score (0.4148) for the minority class among parametric/non-linear models, balancing sensitivity well. |
| **Support Vector Machine** | Tied with Logistic Regression for highest accuracy (89.21%) and strongest precision (63.70%), exhibiting robust linear separation capabilities. |
| **Overall, Winner for your dataset?** | **Logistic Regression / SVM** achieved the highest overall accuracy and MCC score for overall correctness, while **Naive Bayes** provided the best balance for minority class recall. |