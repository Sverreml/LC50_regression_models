import pandas as pd
import sklearn.linear_model as skl
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics

#ready data
df = pd.read_csv(
    r"qsar_aquatic_toxicity.csv",
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

lin_eff = skl.LinearRegression()
lin_eff.fit(X_train_lin, Y_train_lin)

Lin_eff_test = lin_eff.predict(X_test_lin)
lin_eff_train = lin_eff.predict(X_train_lin)

print("Linear effects coefs: ", lin_eff.coef_)
print("Linear effects Training MSE: {:.4}".format(skmetrics.mean_squared_error(lin_eff_train, Y_train_lin)))
print("Linear effects Test MSE: {:.4}".format(skmetrics.mean_squared_error(Y_test_lin, Lin_eff_test)))


#Dummy encoding model
X_train_dum = X_train.drop(columns = ["H050","C040","nN"])
X_test_dum = X_test.drop(columns = ["H050","C040","nN"])

Y_train_dum = Y_train.drop(columns = ["H050","C040","nN"])
Y_test_dum = Y_test.drop(columns = ["H050","C040","nN"])

dum_enc = skl.LinearRegression()
dum_enc.fit(X_train_dum, Y_train_dum)

dum_enc_test = dum_enc.predict(X_test_dum)
dum_emc_train = dum_enc.predict(X_train_dum)

print("Dummy enconding coefs: ", dum_enc.coef_)
print("Dummy enconding Training MSE: {:.4}".format(skmetrics.mean_squared_error(dum_emc_train, Y_train_dum)))
print("Dummy enconding Test MSE: {:.4}".format(skmetrics.mean_squared_error(Y_test_dum, dum_enc_test)))