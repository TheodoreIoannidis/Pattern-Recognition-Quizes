import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score

data = pd.read_csv("./PCA_quiz.csv")
# print(data.head())

# Spliting data into train and test sets
trainingRange = list(range(0, 50)) + list(range(90, 146))
training = data.loc[trainingRange, :]
trainingType = training.loc[:, "Type"]  # Train set Labels
training = training.drop(["Type"], axis=1)

testingRange = list(range(50, 90))
testing = data.loc[testingRange, :]
testingType = testing.loc[:, "Type"]  # Test set Labels
testing = testing.drop(["Type"], axis=1)

# -----------------------------------------------------------------

# Question 1:
scaler = StandardScaler()
scaler = scaler.fit(training)
transformed = pd.DataFrame(scaler.transform(training), columns=training.columns)
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_
# print(eigenvalues)
print("PC1 info: ", round(eigenvalues[0] / sum(eigenvalues), 4))

# Question 2:
infoloss = sum(eigenvalues[4:9]) / (sum(eigenvalues))
print("info loss: ", round(infoloss, 4))


# --------------knn questions-----------------

# Question 3:
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(training, trainingType)
pred = clf.predict(testing)

accuracy = accuracy_score(testingType, pred)
recall = recall_score(testingType, pred, pos_label=2)
print("kNN Accuracy: ", round(accuracy, 3))
print("kNN Recall: ", round(recall, 1))


# Question 4:
transformedtrain = pd.DataFrame(scaler.transform(training), columns=training.columns)
transformedtest = scaler.transform(testing)
accuracies = []
for i in range(1, 10):
    pca = PCA(n_components=i)
    pca = pca.fit(transformedtrain.values)
    pca_transformedtrain = pca.transform(transformedtrain.values)
    pca_transformedtest = pca.transform(transformedtest)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(pca_transformedtrain, trainingType)
    pred = clf.predict(pca_transformedtest)
    accuracy = accuracy_score(pred, testingType)
    accuracies.append(accuracy)

best_score = max(accuracies)
best_n_components = accuracies.index(best_score) + 1
print(f"We get the best Accuracy ({best_score}) for {best_n_components} components.")
