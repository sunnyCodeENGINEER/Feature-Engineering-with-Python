import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

data = pd.read_csv("housing.csv")
df = pd.read_csv("ames.csv")
print(data.head())
print(data.columns)
X = data.loc[:, ["median_income", "latitude", "longitude"]]
print(X.head())
# data might need rescaling

# Create cluster feature
kmeans = KMeans(n_clusters=6, n_init=10, random_state=1)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

print(X.head())
print(X.dtypes)

sns.relplot(x="longitude", y="latitude", hue="Cluster", data=X, height=6)

X["MedHouseVal"] = data["median_house_value"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6)

# SCALING FEATURES
X2 = df.copy()
y = X2.pop("SalePrice")

X = df.copy()
y = X.pop("SalePrice")

features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF","GrLivArea"]

# Standardize
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)

kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)

Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["SalePrice"] = y
sns.relplot(
    x="value", y="SalePrice", hue="Cluster", col="variable", height=4, aspect=1, facet_kws={'sharex': False},
    col_wrap=3, data=Xy.melt(
        value_vars=features, id_vars=["SalePrice", "Cluster"],
    ),
)


plt.show()





