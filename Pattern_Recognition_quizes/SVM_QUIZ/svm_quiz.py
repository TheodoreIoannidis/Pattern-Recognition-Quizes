import pandas as pd
from sklearn.metrics import accuracy_score

X1 = [2, 2, -2, -2, 1, 1, -1, -1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]

Y = [1, 1, 1, 1, 2, 2, 2, 2]

alldata = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})
xtrain = alldata.loc[:, ["X1", "X2"]].values
ytrain = alldata.loc[:, "Y"].values
# print("ytrain: \n",ytrain)
# print("xtrain: \n",xtrain)

from sklearn import svm

# Question 1

clf = svm.SVC(kernel="rbf", gamma=1)
clf.fit(xtrain, ytrain)
pred = clf.predict(xtrain)
acc = accuracy_score(pred, ytrain)
print(f"Accuracy in the training set: {acc*100} %")

# Question 2

clf = svm.SVC(kernel="rbf", gamma=1000000)
clf.fit(xtrain, ytrain)
pred = clf.predict([[-2, -1.9]])
print(f"Prediction for the observation: {pred[0]}")
