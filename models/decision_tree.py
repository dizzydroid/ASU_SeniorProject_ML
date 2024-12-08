# models/decision_tree.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, y_train, X_val, y_val):
    """
    Train Decision Tree classifier with hyperparameter tuning.
    """
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    grid = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_dt = grid.best_estimator_
    print(f"Best Decision Tree Params: {grid.best_params_}")
    
    return best_dt
