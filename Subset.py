import pandas as pd
import numpy as np
import sklearn.linear_model as skl
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
from sklearn import feature_selection as fs


df = pd.read_csv(
    r"LC50_regression_models\qsar_aquatic_toxicity.csv",
    delimiter=";",
    names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040", "LC50"])



Linear_model = skl.LinearRegression()
X_train, X_test, Y_train, Y_test = skm.train_test_split(df[["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040"]],
                                                        df["LC50"],
                                                        random_state=0,
                                                        test_size=0.33)


def AIC(estimator, X, y):
    n_samples = X.shape[0]
    y_pred = estimator.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    k = X.shape[1] + 1
    aic = n_samples * np.log(rss / n_samples) + 2 * k
    return -aic

def BIC(estimator, X, y):
    n_samples = X.shape[0]
    y_pred = estimator.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    k = X.shape[1] + 1
    bic = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)
    return -bic


#Forward selection AIC
var_sel_for_AIC = fs.SequentialFeatureSelector(
        Linear_model,
        direction="forward",
        scoring = AIC,
        cv=5
        )

var_sel_for_AIC.fit(X_train, Y_train)

var_sel_for_AIC_support = []
for i in range(len(var_sel_for_AIC.get_support())):
    if var_sel_for_AIC.get_support()[i]:
        var_sel_for_AIC_support.append(X_train.columns[i])

#Backward selection AIC
var_sel_back_AIC = fs.SequentialFeatureSelector(
        Linear_model,
        direction="backward",
        scoring = AIC,
        cv=5
        )

var_sel_back_AIC.fit(X_train, Y_train)

var_sel_back_AIC_support = []
for i in range(len(var_sel_back_AIC.get_support())):
    if var_sel_back_AIC.get_support()[i]:
        var_sel_back_AIC_support.append(X_train.columns[i])

#Forward selection BIC
var_sel_for_BIC = fs.SequentialFeatureSelector(
        Linear_model,
        direction="forward",
        scoring = BIC,
        cv=5
        )

var_sel_for_BIC.fit(X_train, Y_train)

var_sel_for_BIC_support = []
for i in range(len(var_sel_for_BIC.get_support())):
    if var_sel_for_BIC.get_support()[i]:
        var_sel_for_BIC_support.append(X_train.columns[i])

#Backward selection BIC
var_sel_back_BIC = fs.SequentialFeatureSelector(
        Linear_model,
        direction="backward",
        scoring = BIC,
        cv=5
        )

var_sel_back_BIC.fit(X_train, Y_train)

var_sel_back_BIC_support = []
for i in range(len(var_sel_back_BIC.get_support())):
    if var_sel_back_BIC.get_support()[i]:
        var_sel_back_BIC_support.append(X_train.columns[i])

print("Forward AIC selected variables: ", var_sel_for_AIC_support)
print("Backward AIC selected variables: ", var_sel_back_AIC_support)
print("Forward BIC selected variables: ", var_sel_for_BIC_support)
print("Backward BIC selected variables: ", var_sel_back_BIC_support)

#error
print(X_train.shape)
full_model = skl.LinearRegression()
full_model.fit(X_train, Y_train)
Y_train_pred_full = full_model.predict(X_train)
Y_test_pred_full = full_model.predict(X_test)
train_mse_full = skmetrics.mean_squared_error(Y_train, Y_train_pred_full)
test_mse_full = skmetrics.mean_squared_error(Y_test, Y_test_pred_full)
print(f"Full model - Train MSE: {train_mse_full:.4f}, Test MSE: {test_mse_full:.4f}")

model_AIC_for = skl.LinearRegression()
model_AIC_for.fit(X_train[var_sel_for_AIC_support], Y_train)
Y_train_pred_AIC_for = model_AIC_for.predict(X_train[var_sel_for_AIC_support])
Y_test_pred_AIC_for = model_AIC_for.predict(X_test[var_sel_for_AIC_support])
train_mse_AIC_for = skmetrics.mean_squared_error(Y_train, Y_train_pred_AIC_for)
test_mse_AIC_for = skmetrics.mean_squared_error(Y_test, Y_test_pred_AIC_for) 
print(f"Forward AIC Train MSE: {train_mse_AIC_for:.4f}")
print(f"Forward AIC Test MSE: {test_mse_AIC_for:.4f}")

model_AIC_back = skl.LinearRegression()
model_AIC_back.fit(X_train[var_sel_back_AIC_support], Y_train)
Y_train_pred_AIC_back = model_AIC_back.predict(X_train[var_sel_back_AIC_support])
Y_test_pred_AIC_back = model_AIC_back.predict(X_test[var_sel_back_AIC_support])
train_mse_AIC_back = skmetrics.mean_squared_error(Y_train, Y_train_pred_AIC_back)
test_mse_AIC_back = skmetrics.mean_squared_error(Y_test, Y_test_pred_AIC_back) 
print(f"Backward AIC Train MSE: {train_mse_AIC_back:.4f}")
print(f"Backward AIC Test MSE: {test_mse_AIC_back:.4f}")

model_BIC_for = skl.LinearRegression()
model_BIC_for.fit(X_train[var_sel_for_BIC_support], Y_train)
Y_train_pred_BIC_for = model_BIC_for.predict(X_train[var_sel_for_BIC_support])
Y_test_pred_BIC_for = model_BIC_for.predict(X_test[var_sel_for_BIC_support])
train_mse_BIC_for = skmetrics.mean_squared_error(Y_train, Y_train_pred_BIC_for)
test_mse_BIC_for = skmetrics.mean_squared_error(Y_test, Y_test_pred_BIC_for) 
print(f"Forward BIC Train MSE: {train_mse_BIC_for:.4f}")
print(f"Forward BIC Test MSE: {test_mse_BIC_for:.4f}")

model_BIC_back = skl.LinearRegression()
model_BIC_back.fit(X_train[var_sel_back_BIC_support], Y_train)
Y_train_pred_BIC_back = model_BIC_back.predict(X_train[var_sel_back_BIC_support])
Y_test_pred_BIC_back = model_BIC_back.predict(X_test[var_sel_back_BIC_support])
train_mse_BIC_back = skmetrics.mean_squared_error(Y_train, Y_train_pred_BIC_back)
test_mse_BIC_back = skmetrics.mean_squared_error(Y_test, Y_test_pred_BIC_back) 
print(f"Backward BIC Train MSE: {train_mse_BIC_back:.4f}")
print(f"Backward BIC Test MSE: {test_mse_BIC_back:.4f}")