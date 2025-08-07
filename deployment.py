import joblib



import numpy as np
import io
import pandas as pd
from flask import Flask, jsonify, request
import pickle
from lightgbm import LGBMClassifier
from helpers.final_model_helper import preprocessor_for_sklearn




app = Flask("default")


required_columns = [
    ["SK_ID_CURR"]
]


@app.route("/predict", methods=["POST"])
def predict():
    # Get the provided JSON
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Handle both single and multiple records

    # Multiple records in 'split' orientation expected
    try:
        X = decode_json_to_df(data)
    except ValueError as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    # if not check_required_columns(X):
    #     return (
    #         jsonify(
    #             {
    #                 "error": f"Missing required input columns; \nsupplied:{X.columns};\nrequired {required_columns}"
    #             }
    #         ),
    #         400,
    #     )

    results = make_prediction(X)

    return jsonify(results)


def make_prediction(X: pd.DataFrame) -> dict:
    """
    Make a prediction using the pre-trained model.

    Args:
        X (pd.DataFrame): Input features for prediction.

    Returns:
        dict: Prediction results including id, probability, and prediction.
    """
    probabilities = model.predict_proba(X)[:, 1]  # Get positive class probability
    prediction = model.predict(X)

    results = {
        "id": X["id"].tolist(),
        "probability": probabilities.tolist(),
        "prediction": prediction.tolist(),
    }

    return results


def decode_json_to_df(json_data):
    """Convert JSON data to DataFrame and ensure required columns exist."""
    try:
        # Check if this is split-oriented format (columns, index, data)
        if isinstance(json_data, dict) and all(
            k in json_data for k in ["columns", "data"]
        ):
            # This is a split format - reconstruct directly
            df = pd.DataFrame(
                json_data["data"],
                columns=json_data["columns"],
                index=json_data.get("index", None),
            )

        # Column-oriented format with row indices as nested dictionaries
        elif isinstance(json_data, dict) and all(
            isinstance(v, dict) for v in json_data.values()
        ):
            df = pd.DataFrame.from_dict(json_data)

        # Single record
        elif isinstance(json_data, dict):
            df = pd.DataFrame([json_data])

        # List of records
        else:
            df = pd.DataFrame(json_data)


        return df

    except Exception as e:
        raise ValueError(f"Error parsing JSON: {str(e)}")


def check_required_columns(X: pd.DataFrame) -> bool:

    if not all(col in X.columns for col in required_columns):
        return False
    return True


if __name__ == "__main__":

    with open("models/voter_log_reg_gbdt.jbl", "rb") as f:
        model = joblib.load(f)
    app.run(debug=True, host="0.0.0.0", port=8989)
