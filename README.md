# Massively Parallel Algorithms Assignments

This repo contains the code for both the MPA assignments.

## Assignment 1

Instead of just points, each point has a radius making them into balls. We then
check how the clustering is affected by using random radiusses for the balls.

### Generating data

Using `sklearn` we can generate data. This data then has to be converted to a
graph. For each pair of vertices we compute the euclidean distance. Furthermore,
we give each vertex a random radius.

### Computing MST

We use the edge sampling algorithm. For the balls we have to check whether it is
faster to first preprocess the graph or take the radius into consideration
during execution. Each machine has a copy of $V$ so it might be faster to not
pre-process.

~~The random sampling can be done by first prepending a random key to an edge
(i.e., `(534, e)` where $e \in E$), and then using `RDD.partitionBy`.~~ This is done internaly by `RDD.repartition`.

### Visualization

To find the clusters a user speciefies that there should be $k$ clusters in the
dataset. The algorithm finds the $k-1$ largest edges and removes them. Then the
tree becomes a forest and we draw each of the trees in the forest a different
colour.
