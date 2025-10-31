import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as skl
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics

#ready data
df = pd.read_csv(
    r"C:\Users\Sverr\Desktop\Datasets\LC50\qsar_aquatic_toxicity.csv",
    delimiter=";",
    names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040", "LC50"]
)

df.insert(3, "H050_d", (df["H050"] > 0).astype(int))
df.insert(7, "nN_d", (df["nN"] > 0).astype(int))
df.insert(10, "C040_d", (df["C040"] > 0).astype(int))


X = df[["TPSA", "SAacc", "H050","H050_d", "MLOGP", "RDCHI", "GATS1P", "nN", "nN_d", "C040", "C040_d"]]
Y = df["LC50"]

Lin_train_error = []
Lin_test_error = []
dum_train_error = []
dum_test_error = []



for i in range(200):

    X_train, X_test, Y_train, Y_test = skm.train_test_split(X, Y, random_state=i, test_size=0.66)

    #Linear effect model
    X_train_lin = X_train.drop(columns = ["H050_d","C040_d","nN_d"])
    X_test_lin = X_test.drop(columns = ["H050_d","C040_d","nN_d"])

    lin_eff = skl.LinearRegression()
    lin_eff.fit(X_train_lin, Y_train)

    lin_eff_test = lin_eff.predict(X_test_lin)
    lin_eff_train = lin_eff.predict(X_train_lin)

    Lin_train_error.append(skmetrics.mean_squared_error(lin_eff_train, Y_train))
    Lin_test_error.append(skmetrics.mean_squared_error(Y_test, lin_eff_test))


    #Dummy encoding model
    X_train_dum = X_train.drop(columns = ["H050","C040","nN"])
    X_test_dum = X_test.drop(columns = ["H050","C040","nN"])

    dum_enc = skl.LinearRegression()
    dum_enc.fit(X_train_dum, Y_train)

    dum_enc_test = dum_enc.predict(X_test_dum)
    dum_enc_train = dum_enc.predict(X_train_dum)

    dum_train_error.append(skmetrics.mean_squared_error(dum_enc_train, Y_train))
    dum_test_error.append(skmetrics.mean_squared_error(Y_test, dum_enc_test))

lower = min(min(Lin_train_error), min(Lin_test_error), min(dum_train_error), min(dum_test_error))
upper = max(max(Lin_train_error), max(Lin_test_error), max(dum_train_error), max(dum_test_error))
bins = np.linspace(lower, upper, 100)

fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
#TODO Add means (and std?) for each plot
axs[0,0].set_title("Linear Effects Training MSE")
axs[0,1].set_title("Linear Effects Test MSE")
axs[1,0].set_title("Dummy Encoding Training MSE")
axs[1,1].set_title("Dummy Encoding Test MSE")
axs[0,0].hist(Lin_train_error, bins)
axs[0,1].hist(Lin_test_error, bins)
axs[1,0].hist(dum_train_error, bins)
axs[1,1].hist(dum_test_error, bins)
plt.show()