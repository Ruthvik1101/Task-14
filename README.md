Task 14: Model Comparison & Best Model Selection


Objective

The objective of this project is to evaluate and compare the performance of multiple machine learning classification models using standard evaluation metrics such as accuracy, precision, recall, and F1-score. By training different algorithms on the same dataset and assessing them using consistent criteria, the goal is to identify the model that performs best and generalizes well to unseen data.

Project Overview

In this project, multiple machine learning algorithms are trained and evaluated on the same dataset. Their performance is compared using statistical metrics and visualization techniques. The best-performing model is selected based on evaluation results and saved for future use, simulating a real-world machine learning workflow.

Technologies Used

Python

Pandas

Scikit-learn

Matplotlib

Joblib

Dataset

The project uses the Breast Cancer dataset from Scikit-learn.
It is a binary classification dataset used to predict whether a tumor is malignant or benign based on medical features.

Models Compared

The following classification models were trained and evaluated:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Evaluation Metrics

Each model was evaluated using:

Accuracy → Overall correctness of predictions

Precision → Correct positive predictions

Recall → Ability to identify actual positives

F1 Score → Balance between precision and recall

 Workflow

Load and preprocess the dataset

Split data into training and testing sets

Apply feature scaling using StandardScaler

Train multiple machine learning models

Predict results on test data

Evaluate models using performance metrics

Create a comparison table and visualization

Select and save the best-performing model

esults

All models were compared using performance metrics, and the best-performing model was selected based on F1-score and overall generalization ability.

The selected best model is saved as:

best_model.pkl
