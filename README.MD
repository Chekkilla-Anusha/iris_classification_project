# Iris Flower Classification Using Decision Tree

## Project Overview

This project demonstrates a Machine Learning classification model using the famous Iris Flower Dataset. A Decision Tree Classifier is trained to predict the species of an iris flower based on its physical measurements.

## Dataset

The Iris dataset contains the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target Variable:

- Species
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

## Technologies Used

- Python
- Pandas
- Scikit-learn

## Libraries Required

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## Project Workflow

### 1. Load Dataset

The Iris dataset is loaded using Pandas.

```python
iris = pd.read_csv('iris.csv')
```

### 2. Data Preparation

Separate features and target variable.

```python
X = iris.drop(columns='species')
y = iris['species']
```

### 3. Train-Test Split

Split the dataset into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 4. Model Training

Train a Decision Tree Classifier.

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 5. Prediction

Generate predictions on test data.

```python
y_pred = model.predict(X_test)
```

### 6. Model Evaluation

Evaluate model performance using:

- Accuracy Score
- Confusion Matrix
- Classification Report

```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Results

The model predicts iris flower species based on flower measurements and evaluates performance using standard classification metrics.
