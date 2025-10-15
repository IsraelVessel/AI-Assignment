"""Iris classification script using scikit-learn DecisionTree
Run: python iris_classification.py
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


def main():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    # Check for missing values
    print('Missing values per column:')
    print(X.isna().sum())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision (macro):', precision_score(y_test, y_pred, average='macro'))
    print('Recall (macro):', recall_score(y_test, y_pred, average='macro'))
    print('\nClassification report:\n', classification_report(y_test, y_pred, target_names=data.target_names))


if __name__ == '__main__':
    main()
