import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

class LottoBaseline:
    def __init__(self):
        self.model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        self.dummy = MultiOutputClassifier(DummyClassifier(strategy='most_frequent'))

    def train(self, X, y):
        print("[Baseline] Training Logistic Regression...")
        self.model.fit(X, y)
        print("[Baseline] Training Dummy Classifier...")
        self.dummy.fit(X, y)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_dummy = self.dummy.predict(X_test)
        
        metrics = {
            "baseline_f1": f1_score(y_test, y_pred, average='samples'),
            "dummy_f1": f1_score(y_test, y_dummy, average='samples')
        }
        return metrics
