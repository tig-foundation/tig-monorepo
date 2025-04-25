# Knapsack Problem

The quadratic knapsack problem is one of the most popular variants of the single knapsack problem, with applications in many optimization contexts. The aim is to maximize the value of individual items placed in the knapsack while satisfying a weight constraint. However, pairs of items also have positive interaction values, contributing to the total value within the knapsack.


## Challenge Overview

For our challenge, we use a version of the quadratic knapsack problem with configurable difficulty, where the following two parameters can be adjusted in order to vary the difficulty of the challenge:

- Parameter 1:  $num\textunderscore{ }items$ is the number of items from which you need to select a subset to put in the knapsack. 
- Parameter 2: $better\textunderscore{ }than\textunderscore{ }baseline \geq 1$ (see Our Challenge)

The larger the $num\textunderscore{ }items$, the more number of possible $S_{knapsack}$, making the challenge more difficult. Also, the higher $better\textunderscore{ }than\textunderscore{ }baseline$, the less likely a given $S_{knapsack}$ will be a solution, making the challenge more difficult.

The weight $w_i$ of each of the $num\textunderscore{ }items$ is an integer, chosen independently, uniformly at random, and such that each of the item weights $1 <= w_i <= 50$, for $i=1,2,...,num\textunderscore{ }items$. The values of the items are nonzero  with a density of 25%, meaning they have a 25% probability of being nonzero. The nonzero individual values of the item, $v_i$, and the nonzero interaction values of pairs of items,  $V_{ij}$, are selected at random from the range $[1,100]$.

The total value of a knapsack is determined by summing up the individual values of items in the knapsack, as well as the interaction values of every pair of items \((i,j)\), where \( i > j \), in the knapsack:

$$
V_{knapsack} = \sum_{i \in knapsack}{v_i} + \sum_{(i,j)\in knapsack}{V_{ij}}
$$

We impose a weight constraint $W(S_{knapsack}) <= 0.5 \cdot W(S_{all})$, where the knapsack can hold at most half the total weight of all items.


# Example

Consider an example of a challenge instance with `num_items=4`:

```
weights = [39, 29, 15, 43, 3]
individual_values = [0, 14, 0, 75, 10]
interaction_values = [ 0,  0,  0,  0,  5
                       0,  0, 32,  0, 10
                       0, 32,  0,  0,  0
                       0,  0,  0,  0,  0
                       5, 10,  0,  0,  0 ]
max_weight = 63
baseline_value = 100
```

Now consider the following selection:

```
selected_items =  [2, 3, 4]
```

When evaluating this selection, we can confirm that the total weight is less than 63, and the total value is 127:

* Total weight = 15 + 43 + 3 = 61
* Interaction values = (2,3) + (2,4) + (3,4) = 32 + 10 + 0 = 42
* Individual values = 0 + 75 + 10 = 85
* Total value = 42 + 85 = 127

This selection is 27% better than the baseline: 
```
better_than_baseline = total_value / baseline_value - 1
                     = 127 / 100 - 1
                     = 0.27
```

# Our Challenge 
In TIG, the baseline value is determined by a two-stage approach. First, items are selected based on their value-to-weight ratio, including interaction values, until the capacity is reached. Then, a tabu-based local search refines the solution by swapping items to improve value while avoiding reversals, with early termination for unpromising swaps.

Each instance of TIG's knapsack problem contains 16 random sub-instances, each with its own baseline selection and baseline value. For each sub-instance, we calculate how much your selection's total value exceeds the baseline value, expressed as a percentage improvement. This improvement percentage is called `better_than_baseline`. Your overall performance is measured by taking the root mean square of these 16 `better_than_baseline` percentages. To pass a difficulty level, this overall score must meet or exceed the specified difficulty target.

For precision, `better_than_baseline` is stored as an integer where each unit represents 0.01%. For example, a `better_than_baseline` value of 150 corresponds to 150/10000 = 1.5%.