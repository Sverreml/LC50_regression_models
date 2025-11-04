import pandas as pd
import sklearn.linear_model as skl
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
import statsmodels.api as sm
import statsmodels.formula.api as smf

#ready data
df = pd.read_csv(
    r"LC50_regression_models\qsar_aquatic_toxicity.csv",
    delimiter=";",
    names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040", "LC50"]
)

df.insert(3, "H050_d", (df["H050"] > 0).astype(int))
df.insert(7, "nN_d", (df["nN"] > 0).astype(int))
df.insert(10, "C040_d", (df["C040"] > 0).astype(int))


X = df[["TPSA", "SAacc", "H050","H050_d", "MLOGP", "RDCHI", "GATS1P", "nN", "nN_d", "C040", "C040_d"]]
Y = df["LC50"]

X_train, X_test, Y_train, Y_test = skm.train_test_split(X, Y, random_state=0, test_size=0.66)

#Linear effect model
X_train_lin = X_train.drop(columns = ["H050_d","C040_d","nN_d"])
X_test_lin = X_test.drop(columns = ["H050_d","C040_d","nN_d"])

Y_train_lin = Y_train.drop(columns = ["H050_d","C040_d","nN_d"])
Y_test_lin = Y_test.drop(columns = ["H050_d","C040_d","nN_d"])

lin_eff = smf.ols(formula='LC50 ~ TPSA + SAacc + H050 + MLOGP + RDCHI + GATS1P + nN + C040', data=pd.concat([X_train_lin, Y_train_lin], axis=1))
lin_eff_fit = lin_eff.fit()
Lin_eff_test = lin_eff_fit.predict(X_test_lin)
lin_eff_train = lin_eff_fit.predict(X_train_lin)

lin_eff_summary = lin_eff_fit.summary()
print(lin_eff_summary)



#Dummy encoding model
X_train_dum = X_train.drop(columns = ["H050","C040","nN"])
X_test_dum = X_test.drop(columns = ["H050","C040","nN"])

Y_train_dum = Y_train.drop(columns = ["H050","C040","nN"])
Y_test_dum = Y_test.drop(columns = ["H050","C040","nN"])

dum_enc = smf.ols(formula='LC50 ~ TPSA + SAacc + H050_d + MLOGP + RDCHI + GATS1P + nN_d + C040_d', data=pd.concat([X_train_dum, Y_train_dum], axis=1))
dum_enc_fit = dum_enc.fit()

lin_eff_summary = dum_enc_fit.summary()
print(lin_eff_summary)
