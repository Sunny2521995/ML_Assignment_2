
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load training and testing datasets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

X_train = train_data.drop("y", axis=1)
y_train = train_data["y"]

X_test = test_data.drop("y", axis=1)
y_test = test_data["y"]

# Identify numerical and categorical columns
numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

# Numerical preprocessing
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Define the five machine learning models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "SVM": LinearSVC(random_state=42, max_iter=5000)
}

# Create a folder to save trained models
os.makedirs("models", exist_ok=True)

results = []

# Train and evaluate each model
for model_name, classifier in models.items():

    print("\nTraining:", model_name)

    # Create a complete preprocessing and model pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Predict test results
    y_pred = model_pipeline.predict(X_test)

    # Calculate scores for AUC
    if hasattr(model_pipeline, "predict_proba"):
        y_score = model_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = model_pipeline.decision_function(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    precision = precision_score(
        y_test, y_pred, pos_label="yes", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label="yes", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label="yes", zero_division=0
    )
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Save the complete pipeline
    filename = model_name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model_pipeline, os.path.join("models", filename))

    print("Completed:", model_name)

# Create comparison table
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv("model_comparison.csv", index=False)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

print("\nAll models trained and saved successfully!")
