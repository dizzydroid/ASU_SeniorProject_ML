# main.py

from utils.data_preprocessing import load_data, preprocess_data, split_data
from models.knn_model import train_knn
from models.logistic_regression import train_logistic_regression
from models.naive_bayes import train_naive_bayes
from models.decision_tree import train_decision_tree
from models.svm_model import train_svm
from utils.metrics import evaluate_model, print_classification_report
from utils.plot_utils import plot_roc_curve

import pandas as pd

def main():
    # Load and preprocess data
    data_filepath = 'data/data.csv'
    df = load_data(data_filepath)
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print("Data split into training, validation, and testing sets.")
    
    # Dictionary to store models and their performance
    models_performance = {}
    
    # Train K-Nearest Neighbors
    knn = train_knn(X_train, y_train, X_val, y_val)
    knn_pred = knn.predict(X_test)
    knn_proba = knn.predict_proba(X_test)[:,1] if hasattr(knn, "predict_proba") else None
    knn_metrics = evaluate_model(y_test, knn_pred, knn_proba)
    models_performance['KNN'] = knn_metrics
    print("KNN Performance:", knn_metrics)
    print_classification_report(y_test, knn_pred)
    if knn_proba is not None:
        plot_roc_curve(y_test, knn_proba, 'KNN')
    
    # Train Logistic Regression
    lr = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:,1]
    lr_metrics = evaluate_model(y_test, lr_pred, lr_proba)
    models_performance['Logistic Regression'] = lr_metrics
    print("Logistic Regression Performance:", lr_metrics)
    print_classification_report(y_test, lr_pred)
    plot_roc_curve(y_test, lr_proba, 'Logistic Regression')
    
    # Train Naïve Bayes
    nb = train_naive_bayes(X_train, y_train, X_val, y_val)
    nb_pred = nb.predict(X_test)
    nb_proba = nb.predict_proba(X_test)[:,1]
    nb_metrics = evaluate_model(y_test, nb_pred, nb_proba)
    models_performance['Naive Bayes'] = nb_metrics
    print("Naive Bayes Performance:", nb_metrics)
    print_classification_report(y_test, nb_pred)
    plot_roc_curve(y_test, nb_proba, 'Naive Bayes')
    
    # Train Decision Tree
    dt = train_decision_tree(X_train, y_train, X_val, y_val)
    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:,1]
    dt_metrics = evaluate_model(y_test, dt_pred, dt_proba)
    models_performance['Decision Tree'] = dt_metrics
    print("Decision Tree Performance:", dt_metrics)
    print_classification_report(y_test, dt_pred)
    plot_roc_curve(y_test, dt_proba, 'Decision Tree')
    
    # Train Support Vector Machine
    svm = train_svm(X_train, y_train, X_val, y_val)
    svm_pred = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)[:,1]
    svm_metrics = evaluate_model(y_test, svm_pred, svm_proba)
    models_performance['SVM'] = svm_metrics
    print("SVM Performance:", svm_metrics)
    print_classification_report(y_test, svm_pred)
    plot_roc_curve(y_test, svm_proba, 'SVM')
    
    # Compare Models
    performance_df = pd.DataFrame(models_performance).T
    print("Model Comparison:\n", performance_df)

if __name__ == "__main__":
    main()
