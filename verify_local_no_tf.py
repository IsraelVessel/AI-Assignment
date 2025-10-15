"""Lightweight verification script that runs Iris dataset without requiring pandas.
This avoids scikit-learn's as_frame=True path which needs pandas.
Run: python verify_local_no_tf.py
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    # load without as_frame to avoid pandas dependency
    data = load_iris(as_frame=False)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy (lightweight):', accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
