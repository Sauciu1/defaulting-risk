import joblib
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request

app = Flask("default")

@app.route("/predict", methods=["POST"])
def predict():
    # Check content type
    content_type = request.headers.get('Content-Type')
    if content_type != 'application/pickle':
        return pickle.dumps({"error": "Content-Type must be application/pickle"}), 400

    # Get pickle data
    if not request.data:
        return pickle.dumps({"error": "No input data provided"}), 400

    try:
        X = pickle.loads(request.data)
    except Exception as e:
        return pickle.dumps({"error": f"Invalid pickle format: {str(e)}"}), 400

    if not isinstance(X, pd.DataFrame):
        return pickle.dumps({"error": "Input must be a pandas DataFrame"}), 400

    if not check_required_columns(X):
        return pickle.dumps({"error": "Missing required input columns"}), 400

    print(f"Received DataFrame shape: {X.shape}")
    print(f"Data types: {X.dtypes.value_counts()}")

    results = make_prediction(X)
    
    # Return results as pickle
    return pickle.dumps(results)

def make_prediction(X: pd.DataFrame) -> dict:
    """
    Make a prediction using the pre-trained model.

    Args:
        X (pd.DataFrame): Input features for prediction.

    Returns:
        dict: Prediction results including probability and prediction.
    """
    index = X.pop("SK_ID_CURR") if "SK_ID_CURR" in X.columns else X.index
    probabilities = model.predict_proba(X)[:, 1]  # Get positive class probability
    prediction = model.predict(X)



    results = {
        "index": index,
        "probability": probabilities.tolist(),
        "prediction": prediction.tolist(),
    }

    return results

def check_required_columns(X: pd.DataFrame) -> bool:
    columns_dict = joblib.load("models/deployment_columns.jbl")
    expected_columns = columns_dict["X_cols"]

    # Remove ID columns from check since they're not needed for prediction
    id_cols = columns_dict.get("id_cols", ["SK_ID_CURR"])
    X_prediction_cols = X.drop(columns=[col for col in id_cols if col in X.columns], errors='ignore').columns
    
    missing_cols = set(expected_columns) - set(X_prediction_cols)
    if missing_cols:
        print(f"Missing columns: {len(missing_cols)}")
        return False
    return True

@app.route("/health", methods=["GET"])
def health():
    result = {
        "status": "healthy",
        "model_loaded": 'model' in globals(),
        "accepts": "application/pickle",
        "returns": "pickle"
    }
    return pickle.dumps(result)

if __name__ == "__main__":
    with open("models/deployment_model.jbl", "rb") as f:
        model = joblib.load(f)
    print("Model loaded successfully")
    print("Server accepts: application/pickle")
    print("Server returns: pickle data")
    app.run(debug=True, host="0.0.0.0", port=8080)