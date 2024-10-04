# Knapsack Problem

The quadratic knapsack problem is one of the most popular variants of the single knapsack problem with applications in many optimization problems. The aim is to maximise the value of individual items placed in the knapsack while satisfying a weight constraint. However, pairs of items also have interaction values which may be negative or positive that are added to the total value within the knapsack.

# Example

For our challenge, we use a version of the quadratic knapsack problem with configurable difficulty, where the following two parameters can be adjusted in order to vary the difficulty of the challenge:

- Parameter 1:  $num\textunderscore{ }items$ is the number of items from which you need to select a subset to put in the knapsack. 
- Parameter 2: $better\textunderscore{ }than\textunderscore{ }baseline \geq 1$ is the factor by which a solution must be better than the baseline value [link TIG challenges for explanation of baseline value].


The larger the $num\textunderscore{ }items$, the more number of possible $S_{knapsack}$, making the challenge more difficult. Also, the higher $better\textunderscore{ }than\textunderscore{ }baseline$, the less likely a given $S_{knapsack}$ will be a solution, making the challenge more difficult.

The weight $w_i$ of each of the $num\textunderscore{ }items$ is an integer, chosen independently, uniformly at random, and such that each of the item weights $1 <= w_i <= 50$, for $i=1,2,...,num\textunderscore{ }items$. The individual values of the items $v_i$ are selected by random from the range $50 <= v_i <= 100$, and the interaction values of pairs of items $V_{ij}$ are selected by random from the range $-50 <= V_{ij} <= 50$.  

The total value of a knapsack is determined by summing up the individual values of items in the knapsack, as well as the interaction values of every pair of items $(i,j)$ where $i > j$ in the knapsack:

$$V_{knapsack} = \sum_{i \in knapsack}{v_i} + \sum_{(i,j)\in knapsack}{V_{ij}}$$

We impose a weight constraint $W(S_{knapsack}) <= 0.5 \cdot W(S_{all})$, where the knapsack can hold at most half the total weight of all items.


Consider an example of a challenge instance with `num_items=4` and `better_than_baseline = 1.10`. Let the baseline value be 150:

```
weights = [26, 20, 39, 13]
individual_values = [63, 87, 52, 97]
interaction_values = [  0,  23, -18, -37
                       23,   0,  42, -28
                      -18,  42,   0,  32
                      -37, -28,  32,   0]
max_weight = 60
min_value = baseline*better_than_baseline = 165
```
The objective is to find a set of items where the total weight is at most 60 but has a total value of at least 165.

Now consider the following selection:

```
selected_items =  [0, 1, 3]
```

When evaluating this selection, we can confirm that the total weight is less than 60, and the total value is more than 165, thereby this selection of items is a solution:

* Total weight = 26 + 20 + 13 = 59
* Total value = 63 + 52 + 97 + 23 - 37 - 28 = 170

# Our Challenge 
In TIG, the baseline value is determined by a greedy algorithm that simply iterates through items sorted by potential value to weight ratio, adding them if knapsack is still below the weight constraint.  
