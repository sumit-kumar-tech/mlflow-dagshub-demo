import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub 
dagshub.init(repo_owner='sumitme752', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sumitme752/mlflow-dagshub-demo.mlflow")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train a Decision Tree classifier
max_depth = 15
n_estimators = 8

mlflow.set_experiment("iris_dt")

# Apply mlflow to train
with mlflow.start_run(run_name="pk_exp_with_confusion_matrix_log_artifact"):
    mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    # Evaluate the model
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    print("accuracy:", accuracy)

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,

    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log the model
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt, "decision_tree_model")

    mlflow.set_tag("author", "nitish")
    mlflow.set_tag("project", "iris-classification")
    mlflow.set_tag("algorithm", "decision-tree")