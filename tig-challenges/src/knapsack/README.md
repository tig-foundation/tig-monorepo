# Knapsack Problem

The quadratic knapsack problem is one of the most popular variants of the single knapsack problem, with applications in many optimization contexts. The aim is to select items to maximize the value of the knapsack while satisfying a weight constraint. Pairs of items also have positive interaction values, contributing to the total value within the knapsack.


## Challenge Overview

For our challenge, we use a version of the quadratic knapsack problem with configurable difficulty, framed as **team formation**. Each "item" is a **participant**; you select a subset of participants (a team) subject to a weight (budget) constraint. Value comes from how well participants work together, based on shared projects.

- Parameter 1: $num\textunderscore{ }items$ is the number of participants (items) from which you select a subset.
- Parameter 2: quality target (see Our Challenge).

The larger $num\textunderscore{ }items$, the larger the search space. The generation method is as follows (see the challenge code for full detail):

- **Participants and projects:** There is a large pool of projects. Each participant is assigned a set of projects (cardinality and assignment follow a lognormal-based process so that participants often share projects with others in the same "region" of the project space).
- **Weights:** Each participant has an integer weight in $[1, 10]$, chosen uniformly at random. The knapsack capacity (max weight) is a percentage of the total weight of all participants.
- **Individual values:** $v_i = 0$ for all $i$ (no linear term).
- **Interaction values:** For $i \neq j$, $V_{ij}$ is based on participants $i$ and $j$ being in the same projects: it is the **Jaccard similarity** of their project sets (intersection size / union size), scaled to an integer (e.g. multiplied by 1000). If they share no projects or the union is empty, $V_{ij} = 0$. The matrix is symmetric: $V_{ij} = V_{ji}$.

The total value of a knapsack (team) is the sum of individual values plus the sum of interaction values for every pair in the selection:

$$
V_{knapsack} = \sum_{i \in knapsack}{v_i} + \sum_{(i,j)\in knapsack,\, i < j}{V_{ij}}
$$

A valid solution must use unique participant indices and have total weight at most the given capacity.


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

Your algorithm does not return a solution; it calls `save_solution` as it runs. The **last** saved solution is evaluated. A valid solution must meet the constraints: only **unique** item indices may be selected, and total weight must **not exceed** the knapsack capacity. Invalid solutions are not scored.

The evaluated metric is **quality** (a fixed-point integer with 6 decimal places). For knapsack, quality functions as improvement over the baseline: `quality = (total_value / baseline_value) − 1` (expressed in the fixed-point format). Higher quality is better. See the challenge code for the precise definition.