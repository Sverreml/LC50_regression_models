from pygam import GAM, s
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skm

# ready data
df = pd.read_csv(
    r"LC50_regression_models\qsar_aquatic_toxicity.csv",
    delimiter=";",
    names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040", "LC50"])


X_train, X_test, Y_train, Y_test = skm.train_test_split(df[["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040"]],
                                                        df["LC50"],
                                                        random_state=0,
                                                        test_size=0.33)
# fit GAM model
gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7))
gam.fit(X_train, Y_train)

gam.summary()