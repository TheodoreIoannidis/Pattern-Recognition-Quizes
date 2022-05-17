### PATTERN RECOGNITION k-MEANS QUIZ:
Difficulty: Easy.

This is a quiz on the k-Means clustering algorithm.

First, we are given the data in a csv file ("kmeans_quiz_data.csv").

Then, we are asked to perform k-Means clustering on the data , using the starting points:

    (-4, 10), (0, 0), (4, 10)
and answer the following questions:

### Question 1: 
Calculate the cohesion (inertia) metric. (round to 2 decimals).

### Question 2:
Calculate the separation metric. (round to 2 decimals).

### Question 3:
Calculate the sihlouette score. (round to 2 decimals).


Then we are asked to perform k-Means clustering on the same data, this time with different starting points: (-2, 0), (2, 0), (0, 10).

### Question 4:
Which of the two models produces better clustering of the data?


### My Answers:

I used the scikit-learn library to implement my solution. 

Using the starting points: 

    (-4, 10), (0, 0), (4, 10)

we get:

* Cohesion (Inertia):  637.87
* Separation:  9496.21
* Silhouette:  0.78

Using the starting points:

    (-2, 0), (2, 0), (0, 10)

we get:

* Cohesion (Inertia):  3667.77
* Separation:  6466.3
* Silhouette:  0.46

Comparing the values of these metrics, we can conclude that the first set of starting points results in better clustering, since the silhouette and the separation metrics have a greater value, and also the cohesion has a smaller value.

* Cluster Cohesion measures how close the objects within the same cluster are. The closer the objects within the same cluster are, the smaller the value of the cohesion.

* Cluster Separation measures how well-separated a cluster is from other clusters. The better the clusters are separated, the greater the value of the separation. 

* As far as the silhouette score goes, the best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.

SEE:

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html ,

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html