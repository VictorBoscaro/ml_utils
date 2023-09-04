import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# Check Linearity (Visual Inspection)
def check_linearity(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    plt.scatter(y, pred)
    plt.title("Check for Linearity")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.show()

# Check Homoscedasticity (Visual Inspection)
def check_homoscedasticity(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    residuals = y - pred
    plt.scatter(pred, residuals)
    plt.title("Check for Homoscedasticity")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.show()

# Check Normality of Errors (using Shapiro-Wilk Test)
def check_normality_of_errors(y, X):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    result = stats.shapiro(model.resid)
    return result  # result[1] <= 0.05 implies non-normal errors

# Check for No Endogeneity (Durbin-Watson Test)
def check_endogeneity(y, X):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return sm.stats.durbin_watson(model.resid)  # values close to 2 are better

# Check for Independence (this is typically done by examining the data collection process)
# No generic code can be written for this, as it's problem-specific

# Check for No Autocorrelation (Ljung-Box Test)
def check_autocorrelation(y, X):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    result = sm.stats.acorr_ljungbox(model.resid, lags=[1], return_df=True)
    return result  # p-value <= 0.05 implies autocorrelation

# Check for No Confounding (this typically requires domain expertise)
# No generic code can be written for this, as it's problem-specific

# Dummy dataset
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = 3 + 2*X[:, 0] + 4*X[:, 1] + np.random.randn(100)  # dependent variable

# Check assumptions
check_linearity(X, y)
check_homoscedasticity(X, y)
print("Normality of errors test:", check_normality_of_errors(y, X))
print("Endogeneity test:", check_endogeneity(y, X))
print("Autocorrelation test:", check_autocorrelation(y, X))

from sklearn.metrics import mean_squared_error

def interpret_linear_regression(lr_model, feature_names, X_test=None, y_test=None):
    # Coefficients
    coef = lr_model.coef_
    print("Coefficients:")
    for feature, coef_ in zip(feature_names, coef):
        print(f"{feature}: {coef_}")

    # Intercept
    intercept = lr_model.intercept_
    print(f"\nIntercept: {intercept}")

    # If test set is provided, calculate and display the Mean Squared Error (MSE)
    if X_test is not None and y_test is not None:
        y_pred = lr_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"\nMean Squared Error on Test Set: {mse}")

# Example usage:
# Assuming `lr` is your trained LinearRegression model and
# `feature_names` is a list of your feature names.
# interpret_linear_regression(lr, feature_names)

