# Knapsack Problem

The quadratic knapsack problem is one of the most popular variants of the single knapsack problem, with applications in many optimization contexts. The aim is to maximize the value of individual items placed in the knapsack while satisfying a weight constraint. However, pairs of items also have positive interaction values, contributing to the total value within the knapsack.


## Challenge Overview

For our challenge, we use a version of the quadratic knapsack problem with configurable difficulty, where the following two parameters can be adjusted in order to vary the difficulty of the challenge:

- Parameter 1:  $num\textunderscore{ }items$ is the number of items from which you need to select a subset to put in the knapsack. 
- Parameter 2: $better\textunderscore{ }than\textunderscore{ }baseline \geq 1$ is the factor by which a solution must be better than the baseline value.

The larger the $num\textunderscore{ }items$, the more number of possible $S_{knapsack}$, making the challenge more difficult. Also, the higher $better\textunderscore{ }than\textunderscore{ }baseline$, the less likely a given $S_{knapsack}$ will be a solution, making the challenge more difficult.

The weight $w_i$ of each of the $num\textunderscore{ }items$ is an integer, chosen independently, uniformly at random, and such that each of the item weights $1 <= w_i <= 50$, for $i=1,2,...,num\textunderscore{ }items$. The values of the items are nonzero  with a density of 25%, meaning they have a 25% probability of being nonzero. The nonzero individual values of the item, $v_i$, and the nonzero interaction values of pairs of items,  $V_{ij}$, are selected at random from the range $[1,100]$.

The total value of a knapsack is determined by summing up the individual values of items in the knapsack, as well as the interaction values of every pair of items \((i,j)\), where \( i > j \), in the knapsack:

$$
V_{knapsack} = \sum_{i \in knapsack}{v_i} + \sum_{(i,j)\in knapsack}{V_{ij}}
$$

We impose a weight constraint $W(S_{knapsack}) <= 0.5 \cdot W(S_{all})$, where the knapsack can hold at most half the total weight of all items.


# Example

Consider an example of a challenge instance with `num_items=4` and `better_than_baseline = 1.50`. Let the baseline value be 46:

```
weights = [39, 29, 15, 43]
individual_values = [0, 14, 0, 75]
interaction_values = [ 0,  0,  0,  0
                       0,  0, 32,  0
                       0, 32,  0,  0
                       0,  0,  0,  0]
max_weight = 63
min_value = baseline*better_than_baseline = 69
```
The objective is to find a set of items where the total weight is at most 63 but has a total value of at least 69.

Now consider the following selection:

```
selected_items =  [2, 3]
```

When evaluating this selection, we can confirm that the total weight is less than 63, and the total value is more than 69, thereby this selection of items is a solution:

* Total weight = 15 + 43 = 58
* Total value = 0 + 75 + 0 = 75

# Our Challenge 
In TIG, the baseline value is determined by a two-stage approach. First, items are selected based on their value-to-weight ratio, including interaction values, until the capacity is reached. Then, a tabu-based local search refines the solution by swapping items to improve value while avoiding reversals, with early termination for unpromising swaps.
