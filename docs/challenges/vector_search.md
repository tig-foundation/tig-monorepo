# Vector Range Search

[Vector range search (or vector search engine)](https://en.wikipedia.org/wiki/Vector_database) is the task where, given 2 sets of vectors with the same number of dimensions, a database set and a query set, find for each query vector a nearby vector in the database set, such that the mean distance between the query vectors and their corresponding vector in the database is within a threshold value.

# Example 

* 10 vectors in the database set.
* 2-dimensional space.
* 3 vectors in the query set.
* Mean distance threshold is $0.2$.
* Distance is Euclidean distance

```
vector_database = [
    [0.05, 0.16],
    [0.31, 0.74],
    [0.32, 0.8 ],
    [0.03, 0.25],
    [0.33, 0.07],
    [0.88, 0.77],
    [0.91, 0.29],
    [0.7 , 0.02],
    [0.53, 0.04],
    [0.72, 0.38]
]

query_vectors = [
    [0.89, 0.86],
    [0.26, 0.88],
    [0.17, 0.41]
]
```

The Euclidean distance from each query vector to the database set is approximately:
```
distances = [
    [1.09, 0.59, 0.57, 1.05, 0.97, 0.09, 0.57, 0.86, 0.9 , 0.51],
    [0.75, 0.15, 0.1 , 0.67, 0.81, 0.63, 0.88, 0.97, 0.88, 0.68],
    [0.28, 0.36, 0.42, 0.21, 0.38, 0.8 , 0.75, 0.66, 0.52, 0.55]
]
```

It can be seen that, for each query vector, if we select the following vectors in the database, the mean Euclidean distance will be below 0.2:
```
indexes = [
    5, // select vector 5 in database as "nearby" to query vector 0
    1, // select vector 1 in database as "nearby" to query vector 1
    0, // select vector 0 in database as "nearby" to query vector 2
]
total_distance = 0.09 + 0.1 + 0.28 = 0.47
mean_distance = 0.47 / 3 = 0.16
```

# Our Challenge

In TIG, the vector search challenge features vectors with 250 dimensions, 100000 vectors in the vector database, and uses Euclidean distance. There are two parameters can be adjusted in order to vary the difficulty of the challenge instance:

- Parameter 1: $num\textunderscore{ }queries$ = **The number of queries**.  
- Parameter 2: $better\textunderscore{ }than\textunderscore{ }baseline$ = **The mean Euclidean distance of query vectors to selected nearby vectors in the database have to be below `threshold = 6 - better_than_baseline / 1000`**. 

All vectors in the query and database sets are generated uniformly at random within a 250-dimensional hypercube; that is, each component in a vector is drawn from a uniform distribution over the interval $[0, 1]$.

# Application

Vector search has a wide range of applications an example of which is Threshold-Based Anomaly Detection, where the vector database represents operational data in a high-dimensional space, and query vectors represent new incoming data points to be monitored for anomalies. If the average distance exceeds a predefined threshold, the query vectors are flagged as anomalies. 

See also for example Outlier detection for high dimensional data: https://dl.acm.org/doi/abs/10.1145/375663.375668

Another example application of vector search is in network security, where the vector database corresponds to historical traffic patterns, and query vectors are new traffic data. By tracking the mean distance between sets new data points and historic "regular" data, any deviation exceeding a threshold can indicate a potential intrusion.

