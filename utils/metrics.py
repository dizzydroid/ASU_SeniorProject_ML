# utils/metrics.py

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Evaluate the model using various metrics.
    """
    metrics = {}
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    return metrics

def print_classification_report(y_true, y_pred):
    """
    Print a detailed classification report.
    """
    report = classification_report(y_true, y_pred)
    print(report)

def get_confusion_matrix(y_true, y_pred):
    """
    Return the confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
