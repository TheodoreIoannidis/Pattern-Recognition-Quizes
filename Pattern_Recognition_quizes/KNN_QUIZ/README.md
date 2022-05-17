### PATTERN RECOGNITION k-NEAREST NEIGHBOURS QUIZ:
Difficulty: Easy.

This is a quiz on the k-Nearest Neighbors (kNN) algorithm.

We are given the following data 

X1 = [-2.0, -2.0, -1.8, -1.4, -1.2, 1.2, 1.3, 1.3, 2.0, 2.0, -0.9, -0.5, -0.2, 0.0, 0.0, 0.3, 0.4, 0.5, 0.8, 1.0,]

X2 = [-2.0, 1.0, -1.0, 2.0, 1.2, 1.0, -1.0, 2.0, 0.0, -2.0, 0.0, -1.0, 1.5, 0.0, -0.5, 1.0, 0.0, -1.5, 1.5, 0.0,]

Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

and we are asked to answer these questions:

### Question 1:
Build a kNN Classifier with k= 3. Which class will a new observation (X1, X2)= (1.5, -0.5) be classified as (1 or 2) ?

### Question 2: 
Build a kNN Classifier with k=5. What is the probability of a new observation (X1, X2)= (-1, 1) being classified as class 1. 

### My Answers: 

I used scikit-learn's KNeighborsClassifier.

After building the models and fitting them to the data, we can easily answer the questions. 
print(clf.predict([[1.5, -0.5]])) answers the first question (class 1).

For the second model, print(clf.predict_proba([[-1, 1]])) answers the respective question (prob = 0.6).


SEE:

 https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
