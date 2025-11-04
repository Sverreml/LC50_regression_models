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

penalties = np.logspace(-4, 1, 100)
test_error = []
train_error = []
# fit GAM model
for i in penalties:
    gam = GAM(s(0, lam=i) + s(1, lam=i) + s(2, lam=i) + s(3, lam=i) + s(4, lam=i) + s(5, lam=i) + s(6, lam=i) + s(7, lam=i))
    gam.fit(X_train, Y_train)
    Y_pred = gam.predict(X_test)
    test_error.append(np.mean((Y_test - Y_pred) ** 2))
    train_error.append(np.mean((Y_train - gam.predict(X_train))**2))

min_index = test_error.index(min(test_error))
best_penalty = penalties[min_index]
plt.plot(penalties, test_error, label = "Test error")
plt.plot(penalties, train_error, label = "Train error")
plt.xscale('log')
plt.xlabel('Penalty (lambda)')
plt.ylabel('Mean Squared Error')
plt.axvline(x=best_penalty, color='r', linestyle='--', label=f'Best Penalty: {best_penalty:.4f}')
plt.title('GAM Penalty vs MSE')
plt.legend()
plt.show()

print(f"Gam train MSE with best penalty ({best_penalty:.4f}): {train_error[min_index]:.4f}")
print(f"Gam test MSE with best penalty ({best_penalty:.4f}): {test_error[min_index]:.4f}")