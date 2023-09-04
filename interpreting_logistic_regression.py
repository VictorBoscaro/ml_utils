import numpy as np
from sklearn.metrics import classification_report

def interpret_logistic_regression(lr_model, feature_names):
    # Coefficients
    coef = lr_model.coef_.flatten()
    print("Coefficients:")
    for feature, coef_ in zip(feature_names, coef):
        print(f"{feature}: {coef_}")
        
    # Odds Ratios
    odds_ratios = np.exp(coef)
    print("\nOdds Ratios:")
    for feature, odds in zip(feature_names, odds_ratios):
        print(f"{feature}: {odds}")

    # Intercept
    intercept = lr_model.intercept_[0]
    print(f"\nIntercept: {intercept}")

    # Decision Function example (assuming you have a test set X_test)
    # decision_scores = lr_model.decision_function(X_test)
    # print("\nDecision Function Scores for Test Set:")
    # print(decision_scores)

    # Model Metrics example (assuming you have a test set X_test and y_test)
    # print("\nModel Metrics:")
    # print(classification_report(y_test, lr_model.predict(X_test)))

# Example usage:
# Assuming `lr` is your trained LogisticRegression model and
# `feature_names` is a list of your feature names.
# interpret_logistic_regression(lr, feature_names)
