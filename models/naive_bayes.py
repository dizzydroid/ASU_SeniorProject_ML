# models/naive_bayes.py

from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train, X_val, y_val):
    """
    Train Naïve Bayes classifier.
    GaussianNB does not have hyperparameters to tune in scikit-learn.
    """
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb
