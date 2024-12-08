# models/logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression classifier with hyperparameter tuning.
    """
    lr = LogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']
    }
    
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_lr = grid.best_estimator_
    print(f"Best Logistic Regression Params: {grid.best_params_}")
    
    return best_lr
