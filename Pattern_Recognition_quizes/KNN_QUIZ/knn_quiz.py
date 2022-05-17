import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

X1 = [
    -2.0,
    -2.0,
    -1.8,
    -1.4,
    -1.2,
    1.2,
    1.3,
    1.3,
    2.0,
    2.0,
    -0.9,
    -0.5,
    -0.2,
    0.0,
    0.0,
    0.3,
    0.4,
    0.5,
    0.8,
    1.0,
]
X2 = [
    -2.0,
    1.0,
    -1.0,
    2.0,
    1.2,
    1.0,
    -1.0,
    2.0,
    0.0,
    -2.0,
    0.0,
    -1.0,
    1.5,
    0.0,
    -0.5,
    1.0,
    0.0,
    -1.5,
    1.5,
    0.0,
]

Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

data = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})
xtrain = data.loc[:, ["X1", "X2"]].values
ytrain = data.loc[:, "Y"].values

# Question 1

clf = KNeighborsClassifier(n_neighbors=3).fit(xtrain, ytrain)
pred = clf.predict([[1.5, -0.5]])
print(f"Prediction: {pred[0]}")

# Question 2
clf = KNeighborsClassifier(n_neighbors=5).fit(xtrain, ytrain)
pred_prob = clf.predict_proba([[-1, 1]])
print(f"Probability of being predicted as class 1: {pred_prob[0][0]*100}%")
