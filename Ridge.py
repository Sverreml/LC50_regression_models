import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sklearn.linear_model as skl
import sklearn.model_selection as skm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    r"C:\Users\Sverr\Desktop\Datasets\LC50\qsar_aquatic_toxicity.csv",
    delimiter=";",
    names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040", "LC50"])


X = df[["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040"]]
Y = df["LC50"]



#Standardize and split
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = skm.train_test_split(df[["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040"]],
                                                        df["LC50"],
                                                        random_state=0)


alphas = np.logspace(-4, 3, 500)

#cross-validation
ridge_error = []
for a in alphas:
    ridge = skl.Ridge(alpha=a, max_iter=10000)
    scores = skm.cross_val_score(ridge, X_train, Y_train, cv=5, scoring="neg_mean_squared_error")
    ridge_error.append(float(-np.mean(scores)))

min_index = ridge_error.index(min(ridge_error))
best_alpha = alphas[min_index]

#Bootstrap
ridge_bootstrap_errors = []
n_bootstraps = 100
counter = 0
for i in range(alphas.shape[0]):
    print(f"Bootstrap Ridge progress: {counter}/{alphas.shape[0]}")
    counter += 1
    ridge = skl.Ridge(alpha=alphas[i], max_iter=10000)
    bootstrap_errors = []
    for _ in range(n_bootstraps):
        X_resampled, Y_resampled = resample(X_train, Y_train)
        ridge.fit(X_resampled, Y_resampled)
        Y_pred = ridge.predict(X_test)
        mse = np.mean((Y_test - Y_pred) ** 2)
        bootstrap_errors.append(mse)
    ridge_bootstrap_errors.append(np.mean(bootstrap_errors))

min_bootstrap_index = ridge_bootstrap_errors.index(min(ridge_bootstrap_errors))
best_bootstrap_alpha = alphas[min_bootstrap_index] 

#Plotting
fig, ax = plt.subplots()
ax.plot(alphas, ridge_error, label="Cross-validation MSE")
ax.plot(alphas, ridge_bootstrap_errors, label="Bootstrap MSE")
ax.axvline(best_alpha, color='r', linestyle='--', label=f'Best CV alpha: {best_alpha:.4f}')
ax.axvline(best_bootstrap_alpha, color='g', linestyle='--', label=f'Best Bootstrap alpha: {best_bootstrap_alpha:.4f}')
ax.set_xscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Ridge Regression: Alpha vs MSE')
ax.legend()
plt.show()