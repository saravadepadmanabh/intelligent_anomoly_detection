import pickle
import pandas as pd
from utils import preprocess_input, categorize_risk

# Load artifacts
model = pickle.load(open("app/model.pkl", "rb"))
scaler = pickle.load(open("app/scaler.pkl", "rb"))
trained_features = pickle.load(open("app/features.pkl", "rb"))

def predict_transaction(input_dict):

    input_df = pd.DataFrame([input_dict])

    processed = preprocess_input(input_df, trained_features)
    scaled = scaler.transform(processed)

    risk_score = model.predict_proba(scaled)[0][1] * 100
    risk_label = categorize_risk(risk_score)

    return {
        "risk_score": round(risk_score, 2),
        "risk_label": risk_label
    }


# Example usage
if __name__ == "__main__":

    sample_transaction = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 200000,
        "oldbalanceOrg": 300000,
        "newbalanceOrig": 100000,
        "oldbalanceDest": 0,
        "newbalanceDest": 200000
    }

    result = predict_transaction(sample_transaction)
    print(result)
