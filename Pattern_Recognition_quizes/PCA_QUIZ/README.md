### PATTERN RECOGNITION PRINCIPAL COMPONENT ANALYSIS (PCA) QUIZ:
Difficulty: Easy

We are given some data in the form of a csv file ("PCA_quiz.csv") and we are instructed to use the samples in rows 50-89 for testing and the rest for training (0-49, 90-145).

We are asked to answer the following questions:

### Question 1:
 
Perform scaling and PCA on the training data. Calculate the information percentage in the first Primary Component (round to 4 decimals).

### Question 2: 

Perform scaling and PCA on the testing data. Calculate the information loss we will have if we only keep first 4 Primary Components. (round to 4 decimals).

### Question 3:

Train a kNN model with k=3 in order to classify the testing data  (without using PCA). Calculate the accuracy the model achieves on the test set (round to 3 decimals).

### Question 4:

Calculate the value of the Recall metric for the testing data, taking class 2 as the positive class (round to 1 decimal).

### Question 5: 

Perform scaling and PCA on the training data. Train a kNN model with k= 3 in order to classify the testing data. What is the otpimal number of Primary Components we should keep in order to maximize the Accuracy?

### My Answers: 

We have 9 primary components (as many as as the features of the data) and therefore, we also have 9 eigenvalues.

First, we get the eigenvalues: 

    eigenvalues = pca.explained_variance_

Then we can calculate the info of PC1
by dividing the corresponding eigenvalue (first) by the sum of the eigenvalues.

    eigenvalues[0] / sum(eigenvalues)

* PC1 info:  0.4064 (40.64%).

With similar logic, we understand that calculating the information loss that keeping the first 4 Primary Components causes, is equivalent to calculating the info of the 5 other Primary Components.  

    sum(eigenvalues[4:9]) / (sum(eigenvalues))

* info loss:  0.128 (12.8%).

Calculating the metrics required in the next 2 questions is pretty straightforward. I used accuracy_score and recall_score from sklearn.metrics.

    accuracy = accuracy_score(testingType, pred)
    recall = recall_score(testingType, pred, pos_label=2)

* kNN Accuracy:  0.775
* kNN Recall:  0.8

With a simple for loop we can train see how the kNN model performs for different n.o. components (1-9).

* We get the best Accuracy (0.85) for 6 components.