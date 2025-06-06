## Overview
[A hypergraph is a generalization of a graph where edges can connect more than just two nodes. Hypergraph partitioning is a technique used to assign the nodes of a hypergraph into separate groups (parts) with the goal of minimizing the connections (hyperedges) linking these parts. This is important for various applications, including parallel computing, VLSI design, and data analysis.](https://en.wikipedia.org/wiki/Hypergraph)  

## Challenge Overview

For our challenge, we use a version of the hypergraph partitioning problem with configurable difficulty, where the following two parameters can be adjusted in order to vary the difficulty of the challenge:

- Parameter 1:  $num\textunderscore{ }hyperedges$ is the number of hyperedges in the hypergraph. 
- Parameter 2: $better\textunderscore{ }than\textunderscore{ }baseline \geq 1$ (see Our Challenge)

A hypergraph is a structure made up of:
* Nodes, each belonging to one or more hyperedges.
* Hyperedges, each containing two or more nodes.

TIG's generation method is such that:
* The weight/cost of nodes and hyperedges are fixed at 1 (in some variants costs can be different)
* The number of nodes is around 92% the number of hypedges (i.e. if there are 100 hyperedges, there are around 92 nodes).
* The number of hyperedges that a node belongs to follows a [power law distribution](https://en.wikipedia.org/wiki/Power_law)
* The number of nodes contained by a hyperedge follows a [power law distribution](https://en.wikipedia.org/wiki/Power_law)

**Objective:**

The goal is deceptively simple: each node must be assigned to one of 64 parts (i.e. 64-way partition).

A partition is scored by the connectivity metric, where the connectivity of each hyperedge is the number of parts it connects:

```
connectivity_metric = 0
for each hyperedge:
    connected = set(
        partition[node] # contains the id of the part a node is assigned to
        for node in hyperedge
    )
    connectivity = len(connected)
    connectivity_metric += connectivity - 1
```

The lower the connectivity metric, the better the partition.

**Constraints:**
1. Each node must be assigned to one part.
2. Every part must contain at least one node.
3. The number of nodes assigned to each part cannot be larger than 1.03x the average part size:
```
average_size = num_nodes / num_parts
max_size = ceil(average_size * 1.03)
for part in partition:
    len(part) <= max_size
```

## Example

Consider an example instance with `num_hyperedges = 16` and `num_nodes = 14`:

```
Edge ID: SIZE: NODES:
      0     2      8, 11
      1    12      0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13
      2     2      8, 9
      3     8      0, 1, 2, 3, 4, 7, 8, 11
      4     4      8, 9, 10, 11
      5     1      13
      6     4      4, 5, 6, 7
      7     1      12
      8     9      1, 2, 4, 6, 7, 8, 9, 10, 11
      9     2      12, 13
     10     2      12, 13
     11     2      1, 2
     12     4      8, 12, 13
     13    10      0, 1, 2, 3, 4, 7, 8, 9, 10, 11
     14     4      0, 1, 2, 3
     15     3      8, 9, 10

baseline_connectivity_metric = 26
```

Now consider the following partition:
```
partition = [1, 3, 3, 0, 2, 0, 0, 2, 3, 2, 1, 2, 1, 1]
# nodes in part 0: [3, 5, 6]
# nodes in part 1: [0, 10, 12, 13] 
# nodes in part 2: [4, 7, 9, 11] 
# nodes in part 3: [1, 2, 8]
```

Evaluating the connectivity of each hyperedge:
```
Hyperedge ID:      0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
Connectivity - 1:  1   3   1   3   2   0   1   0   3   0    2   0   1   3   2   2

connectivity_metric = 24

# explanation: 
hyperedge 4 contains nodes [8, 9, 10, 11], overlapping with 3 parts (connectivity = 3)
   node 8 is in part 3
   node 9 and 11 are in part 2
   node 10 is in part 1
```

This partition is ~7.7% better than the baseline:
```
better_than_baseline = 1 - connectivity_metric / baseline_connectivity_metric
                     = 1 - 24 / 26
                     = 0.077
```

 ## Our Challenge

At TIG, the baseline connectivity is determined using a greedy bipartition approach. The nodes are ordered by degree, then at each bipartition, nodes are assigned to the left or right part based on the number of hyperedges in common with the nodes already in each part. This process is repeated until the desired number of partitions is reached (eg: 64). 

Each instance of TIG's hypergraph partitioning problem contains 4 random sub-instances, each with its own baseline connectivity metric. For each sub-instance, we calculate how much your connectivity metric is better than the baseline connectivity metric, expressed as a percentage improvement. This improvement percentage is called `better_than_baseline`. Your overall performance is measured by taking the root mean square of these 4 `better_than_baseline` percentages. To pass a difficulty level, this overall score must meet or exceed the specified difficulty target.

For precision, `better_than_baseline` is stored as an integer where each unit represents 0.1%. For example, a `better_than_baseline` value of 22 corresponds to 22/1000 = 2.2%.

## Applications

Hypergraphs are a powerful tool for representing complex networks in which relationships may involve more than two elements simultaneously. Hypergraph partitioning refers to dividing such a network into a specified number of groups that are roughly equal in size while keeping as many related items together as possible. Although the problem is computationally challenging (NP-hard), it has broad applications across numerous fields:

* **Parallel Computing & Load Balancing:** By intelligently distributing tasks across processors, hypergraph partitioning minimizes communication overhead and enhances overall computational efficiency [^1][^2][^3][^4][^5].
* **Distributed Neural Network Training:** It enables the partitioning of compute graphs across multiple GPUs or servers, significantly accelerating the training of deep learning models [^6][^7].
* **VLSI & Circuit Design:** By effectively grouping circuit components, it optimizes chip layouts and reduces interconnect complexity, leading to faster and more efficient designs [^8][^9].
* **Social Networks & Community Detection:** Capturing multi-way relationships, hypergraph partitioning reveals hidden community structures and provides deeper insights into group dynamics [^10].
* **Bioinformatics & Computational Biology:** It facilitates the clustering of proteins, genes, and genomic regions to identify functional modules, thereby aiding discovery in biological research [^11].
* **Machine Learning & Data Mining:** By effectively modeling higher-order interactions, it improves data clustering and feature selection, enhancing analytical outcomes [^12].
* **Other Applications:** From optimizing database sharding and segmenting GIS regions to modularizing software systems, hypergraph partitioning transforms large-scale challenges into more tractable problems [^1][^7][^4].

In the rapidly evolving field of Decentralized Physical Infrastructure Networks (DePIN) — which leverage blockchain technology and distributed nodes to manage physical assets — hypergraph partitioning plays an especially important role. By accurately modeling complex interactions, it can effectively group related tasks and resources across scenarios such as decentralized compute/storage, blockchain data sharding, IoT networks, or supply chain logistics [^16]. This grouping helps minimize cross-node communication and balances workloads, ultimately enhancing the scalability and performance of these decentralized systems [^15].

[^1]: Devine, K.D., Boman, E.G., Heaphy, R.T., Bisseling, R.H., & Catalyurek, U.V. (2006). *Parallel hypergraph partitioning for scientific computing*. Proceedings 20th IEEE International Parallel & Distributed Processing Symposium.
[^2]:  Aykanat, C., Cambazoglu, B., & Uçar, B. (2008). *Multi-level direct K-way hypergraph partitioning with multiple constraints and fixed vertices*. Journal of Parallel and Distributed Computing, 68, 609–625.
[^3]: Trifunovic, A., & Knottenbelt, W. (2008). *Parallel multilevel algorithms for hypergraph partitioning*. J. Parallel Distrib. Comput., 68, 563–581.
[^4]: Gottesbüren, L., & Hamann, M. (2022). *Deterministic Parallel Hypergraph Partitioning*. In Euro-Par 2022: Parallel Processing (pp. 301–316). Springer International Publishing.
[^5]: Schlag, S., Heuer, T., Gottesbüren, L., Akhremtsev, Y., Schulz, C., & Sanders, P. (2023). *High-Quality Hypergraph Partitioning*. ACM J. Exp. Algorithmics, 27(1.9), 39. 
[^6]: Zheng, D., Song, X., Yang, C., LaSalle, D., & Karypis, G. (2022). *Distributed Hybrid CPU and GPU Training for Graph Neural Networks on Billion-Scale Heterogeneous Graphs*. In Proceedings (pp. 4582–4591). [↩](https://chatgpt.com/c/67b36128-2270-8009-a6b5-411cb01de345#user-content-fnref-6)
[^7]: Catalyurek, U., Devine, K., Fonseca Faraj, M., Gottesbüren, L., Heuer, T., Meyerhenke, H., Sanders, P., Schlag, S., Schulz, C., & Seemaier, D. (2022). *More Recent Advances in (Hyper)Graph Partitioning*. 
[^8]: Papa, D., & Markov, I. (2007). *Hypergraph Partitioning and Clustering*. In Handbook of Approximation Algorithms and Metaheuristics. 
[^9]: Karypis, G., Aggarwal, R., Kumar, V., & Shekhar, S. (1999). *Multilevel hypergraph partitioning: applications in VLSI domain*. IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 7(1), 69–79. 
[^10]: Zhang, C., Cheng, W., Li, F., & Wang, X. (2024). *Hypergraph-Based Influence Maximization in Online Social Networks*. Mathematics, 12(17), 2769. 
[^11]: Wang, S., Cui, H., Qu, Y., & Yijia, Z. (2025). *Multi-source biological knowledge-guided hypergraph spatiotemporal subnetwork embedding for protein complex identification*. Briefings in Bioinformatics, 26. 
[^12]: Zhou, D., Huang, J., & Schölkopf, B. (2006). *Learning with Hypergraphs: Clustering, Classification, and Embedding*. In Advances in Neural Information Processing Systems 19 (2006), 1601–1608. 
[^13]: Chodrow, P.S., Veldt, N., & Benson, A.R. (2021). *Generative hypergraph clustering: From blockmodels to modularity*. Science Advances, 7.
[^14]: Kolodziej, S., Mahmoudi Aznaveh, M., Bullock, M., David, J., Davis, T., Henderson, M., Hu, Y., & Sandstrom, R. (2019). *The SuiteSparse Matrix Collection Website Interface*. Journal of Open Source Software, 4, 1244.
[^15]: K. Kumar et al. “SWORD: workload-aware data placement and replica selection for cloud data management systems”. In: The VLDB Journal 23 (Dec. 2014), pp. 845–870. doi: 10.1007/s00778-014-0362-1. 
[^16]: Qu C, Tao M, Yuan R. A Hypergraph-Based Blockchain Model and Application in Internet of Things-Enabled Smart Homes. Sensors (Basel). 2018 Aug 24;18(9):2784. doi: 10.3390/s18092784. PMID: 30149523; PMCID: PMC6164253.