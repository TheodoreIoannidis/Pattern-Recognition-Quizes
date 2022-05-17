### PATTERN RECOGNITION SVM QUIZ:
Difficulty: Easy.

This is a quiz on Support Vector Machines (SVMs).

We are given the following data 

    X1 = [2, 2, -2, -2, 1, 1, -1, -1]
    X2 = [2, -2, -2, 2, 1, -1, -1, 1]
    Y = [1, 1, 1, 1, 2, 2, 2, 2]

and we are asked to answer these questions:

### Question 1:
Build a Support Vector classifier with RBF kernel and gamma= 1. Calculate the accuracy that the model achieves on the training set.

### Question 2: 
Build a Support Vector classifier with RBF kernel and gamma=1000000. What will a new observation (X1, X2)= (-2, 1.9) be classified as (class 1 or 2)? 

### My Answers: 

I used scikit-learn's Support Vector Classifier (SVC).

For the first question:

    clf = svm.SVC(kernel="rbf", gamma=1)
    clf.fit(xtrain, ytrain)
    pred = clf.predict(xtrain)
    acc = accuracy_score(pred, ytrain)

* Accuracy in the training set: 100.0 %

For the second question:

    clf = svm.SVC(kernel="rbf", gamma=1000000)
    clf.fit(xtrain, ytrain)
    pred = clf.predict([[-2, -1.9]])

* Prediction for the observation: 2

The accuracy achieved in the training set is 100% and the new observation will be classified as class 2.

SEE:

 https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html