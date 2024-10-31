 # Hypergraph Partitioning Problem

 The hypergraph partitioning problem is the task of splitting a given hypergraph into 64 parts in a way that minimises the cut-cost of the partition. The hypergraph is defined by: $\mathcal{H} = (V, N)$ where $V = {v_1, v_2, v_3, ..., v_{num\_vertices}}$ is the set of  vertices, and $N = {n_1, n_2, n_3, ..., n_{num\_hyperedges}}$. Each hyperedge contains an equal number of vertices, e.g. $n_1 = {v_2, v_6, v_3}$. The cut-cost of a partition is defined as:

 $$ C(\Pi) = \sum_{n \in N_E} \lambda_n - 1 $$

 where $\lambda_n$ is the the number of parts the hyperedge is connected to, and $N_E$ is the set of hyperedges that are cut (connected to more than one part). The only restriction on the parts is that they must have a difference in size of $ <= 1$.

 # Example
 
 For our challenge, the two difficulty parameters are:
 - Parameter 1: $num\_vertices$ = the number of vertices in the hypergraph that need to be partitioned
 - Parameter 2: $better\_than\_baseline$ = the amount the solution must be better than the baseline solution to qualify as a solution

 The number of parts is always 64, the number of hyperedges is set constant at 1000, and the number of vertices in each hyperedge (size) is derived from:

 $$ size\_of\_hyperedges = num\_vertices  // 64 $$

 The maximum cut-cost of the partition that allows it to qualify as a solution is found from:

 $$ maximum\_value = floor(\frac{baseline\_value * (1000 - better\_than\_baseline)}{1000}) $$

 Consider an example with $num\_vertices$ = 20, $better\_than\_baseline = 200$. Let the baseline be 26. To make the example smaller and easier to understand, the $size\_of\_hyperedges = 5$, $num\_hyperedges = 10$ and $num\_parts = 5$ in this example **ONLY** (size of hyperedges will be 64, number of vertices 1000 and num_parts 64 in **ALL** challenge instances):

```
hyperedges = [
    [10, 7, 14, 17, 3],
    [19, 1, 8, 10, 2],
    [0, 11, 16, 4, 10],
    [9, 0, 18, 2, 1],
    [4, 13, 15, 17, 1],
    [17, 8, 12, 16, 6],
    [6, 3, 19, 0, 13],
    [2, 15, 18, 13, 0],
    [15, 14, 16, 17, 13],
    [19, 1, 2, 16, 5]
]
maximum_value = (baseline_value * (1000 - better_than_baseline)) // 1000 = 20
```

The objective is to find a partition where the cut-cost of the partition is less than or equal to 20

Now consider the following partition:

```
partition = [[0, 1, 2, 18],
             [19, 5, 13, 6],
             [10, 7, 14, 3],
             [17, 15, 16, 4],
             [8, 9, 12, 11]]
```

To work out the total cut-cost of the partition, the contribution of each hyperedge must be calculated:

 - Hyperedge 1 [10, 7, 14, 17, 3] in  $P_2, P_3$ 
	=> 	 $\lambda_n = 2 - 1 = 1$ 
 - Hyperedge 2 [19, 1, 8, 10, 2] in  $P_0, P_1, P_2, P_4$ 
	=> $\lambda_n = 4 - 1 = 3$ 
 -	Hyperedge 3 [0, 11, 16, 4, 10] in  $P_0, P_2, P_3, P_4$
	=>  $\lambda_n = 4 - 1 = 3$ 
 - Hyperedge 4 [9, 0, 18, 2, 1] in  $P_0, P_4$ 
	=> $\lambda_n = 2 - 1 = 1$ 
 -	Hyperedge 5 [4, 13, 15, 17, 1] in  $P_0, P_1, P_3$ 
	=>  $\lambda_n = 3 - 1 = 2$ 
 - Hyperedge 6 [17, 8, 12, 16, 6] in  $P_1, P_3, P_4$ 
	=>  $\lambda_n = 3 - 1 = 2$ 
 - Hyperedge 7 [6, 3, 19, 0, 13] in  $P_0, P_1, P_2$ 
    =>  $\lambda_n = 3 - 1 = 2$ 
 -	Hyperedge 8 [2, 15, 18, 13, 0] in  $P_0, P_1, P_3$ 
	=> $\lambda_n = 3 - 1 = 2$ 
 -	Hyperedge 9 [15, 14, 16, 17, 13] in  $P_1, P_2, P_3$ 
	=> $\lambda_n = 3 - 1 = 2$ 
 - Hyperedge 10 [19, 1, 2, 16, 5] in  $P_0, P_1, P_3$ 
	=> $\lambda_n = 3 - 1 = 2$ 

Summing together to get the total cut-cost:

- Cut-cost = 1 + 3 + 3 + 1 + 2 + 2 + 2 + 2 + 2 + 2 = 20

whicih is less than or equal to 20, and thus the partitino is a valid solution.

# Our Challenge

In TIG, the baseline value is determined by a greedy algorithm that recursively bipartitions the original hypergraph and subsequent separate parts, each time selecting vertices based on common hyperedges already in both of the available parts. 