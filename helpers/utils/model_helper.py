import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os, joblib 

def evaluate_multiclass_model(y_true, y_pred, columns=None):
    print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=columns, columns=columns)

    sns.heatmap(cm.div(cm.sum(axis=1), axis=0), annot=True, fmt='.2f')
    plt.title('Confusion Matrix Normalized for True State')
    plt.xlabel('Predicted State')
    plt.ylabel('True State')
    plt.show()

def save_or_load_model(model, X_train, y_train, filename, folder ='models'):
    """
    loads model if it exists, otherwise trains saves model based on a function.
    """
    
    path = os.path.join(folder, filename.split('.')[0])+ '.jbl'

    if not os.path.exists(path):
        print(f"Model not found at {path}. Training and saving model.")
        model.fit(X_train, y_train)
        with open(path, 'wb') as f:
            joblib.dump(model, f)
        print(f"Model saved to {path}")

    with open(path, 'rb') as f:
        loaded_model = joblib.load(f)
    print(f"Model loaded from {path}")
    return loaded_model