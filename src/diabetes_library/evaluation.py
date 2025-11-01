from sklearn.metrics import roc_auc_score

def compute_auc(df):
    """Compute ROC AUC metric."""
    try:
        return roc_auc_score(df["diabetes_mellitus"], df["predictions"])
    except KeyError:
        raise KeyError("Missing required columns: 'diabetes_mellitus' or 'predictions'.")
