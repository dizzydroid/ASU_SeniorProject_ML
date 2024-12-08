# models/svm_model.py

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train, X_val, y_val):
    """
    Train Support Vector Machine classifier with hyperparameter tuning.
    """
    svm = SVC(probability=True)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_svm = grid.best_estimator_
    print(f"Best SVM Params: {grid.best_params_}")
    
    return best_svm
