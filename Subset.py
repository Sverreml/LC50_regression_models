import pandas as pd
import numpy as np
import sklearn.linear_model as skl
import sklearn.model_selection as skm
from sklearn.metrics import make_scorer
from sklearn import feature_selection as fs


df = pd.read_csv(
    r"qsar_aquatic_toxicity.csv",
    delimiter=";",
    names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040", "LC50"])



Linear_model = skl.LinearRegression()
X_train, X_test, Y_train, Y_test = skm.train_test_split(df[["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040"]],
                                                        df["LC50"],
                                                        random_state=0,
                                                        test_size=0.33)

def AIC(estimator, X, y):
    n_samples, n_features = X.shape
    y_pred = estimator.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    k = n_features
    aic = 2*k - 2 * np.log(rss / n_samples)
    return -aic

def BIC(estimator, X, y):
    n_samples, n_features = X.shape
    y_pred = estimator.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    k = n_features
    bic = k * np.log(k) - 2 * np.log(rss / n_samples)
    return -bic


#Forward selection AIC
var_sel_for_AIC = fs.SequentialFeatureSelector(
        Linear_model,
        direction="forward",
        scoring = AIC
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
        scoring = AIC
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
        scoring = BIC
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
        scoring = BIC
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