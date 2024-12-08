# models/knn_model.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_knn(X_train, y_train, X_val, y_val):
    """
    Train K-Nearest Neighbors classifier with hyperparameter tuning.
    """
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': list(range(3, 21, 2)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_knn = grid.best_estimator_
    print(f"Best KNN Params: {grid.best_params_}")
    
    return best_knn
