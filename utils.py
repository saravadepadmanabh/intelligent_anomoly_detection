import pandas as pd

def preprocess_input(input_df, trained_features):

    df = input_df.copy()
    df = pd.get_dummies(df, columns=["type"])

    for col in trained_features:
        if col not in df.columns:
            df[col] = 0

    df = df[trained_features]

    return df


def categorize_risk(score):
    if score >= 80:
        return "HIGH RISK"
    elif score >= 50:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def recommended_action(score):
    if score >= 80:
        return "❌ Block Immediately"
    elif score >= 50:
        return "⚠️ Manual Review Required"
    else:
        return "✅ Safe Transaction"
