import sklearn.tree as skt
import numpy as np
import pandas as pd
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt

# ready data
df = pd.read_csv(
    r"LC50_regression_models\qsar_aquatic_toxicity.csv",
    delimiter=";",
    names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040", "LC50"])


X_train, X_test, Y_train, Y_test = skm.train_test_split(df[["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1P", "nN", "C040"]],
                                                        df["LC50"],
                                                        random_state=0,
                                                        test_size=0.33)
# Decision Tree Regressor
reg_tree = skt.DecisionTreeRegressor(random_state=0)
reg_tree.fit(X_train, Y_train)

path = reg_tree.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

trees = []
for ccp_alpha in ccp_alphas:
    clf = skt.DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    trees.append(clf)


print(f"Number of trees built: {len(trees)}")


train_scores = [t.score(X_train, Y_train) for t in trees]
test_scores = [t.score(X_test, Y_test) for t in trees]

plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, train_scores, marker='o', label='train')
plt.plot(ccp_alphas, test_scores, marker='o', label='test')
plt.xlabel("ccp_alpha (pruning strength)")
plt.ylabel("R2 - score")
plt.title("Cost-Complexity Pruning Path")
plt.legend()
plt.show()


best_alpha = ccp_alphas[np.argmax(test_scores)]
best_tree = skt.DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
best_tree.fit(X_train, Y_train)

print(f"Best alpha: {best_alpha:.5f}")
print(f"Best test R²: {max(test_scores):.3f}")

plt.figure(figsize=(10, 6))
skt.plot_tree(best_tree, filled=True, feature_names=X_train.columns)
plt.title(f"Pruned tree (ccp_alpha={best_alpha:.5f})")
plt.show()