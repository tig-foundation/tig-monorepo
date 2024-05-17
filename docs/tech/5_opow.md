# 5. Optimisable Proof-of-Work

Optimisable proof-of-work (OPoW) uniquely can integrate multiple proof-of-works, “binding” them in such a way that optimisations to the proof-of-work algorithms do not cause instability/centralisation. This binding is embodied in the calculation of influence for Benchmarkers. The adoption of an algorithm is then calculated using each Benchmarker’s influence and the fraction of qualifiers they computed using that algorithm.

## 5.1. Rewards for Benchmarkers

OPoW introduces a novel metric, imbalance, aimed at quantifying the degree to which a Benchmarker spreads their computational work between challenges unevenly. This is only possible when there are multiple proof-of-works.

The metric is defined as:‍

$$imbalance = \frac{C_v(\\%qualifiers)^2}{N-1}$$

‍where \(C_V\) is coefficient of variation, %qualifiers is the fraction of qualifiers found by a Benchmarker across challenges, and N is the number of active challenges. This metric ranges from 0 to 1, where lower values signify less centralisation.

Penalising imbalance is achieved through:

$$imbalance\textunderscore{ }penalty = 1 - exp(- k \cdot imbalance)$$

where k is a coefficient (currently set to 1.5). The imbalance penalty ranges from 0 to 1, where 0 signifies no penalty.

When block rewards are distributed pro-rata amongst Benchmarkers after applying their imbalance penalty, the result is that Benchmarkers are incentivised to minimise their imbalance as to maximise their reward:

$$benchmarker\textunderscore{ }reward \propto mean(\\%qualifiers) \cdot (1 - imbalance\textunderscore{ }penalty)$$

where $\\%qualifiers$ is the fraction of qualifiers found by a Benchmarker for a particular challenge

Notes:

- A Benchmarker focusing solely on a single challenge will exhibit a maximum imbalance and therefore maximum penalty.
- Conversely, a Benchmarker with an equal fraction of qualifiers across all challenges will demonstrate a minimum imbalance value of 0.

## 5.2. Rewards for Innovators

In order to guard against potential manipulation of algorithm adoption by Benchmarkers, Innovator rewards are linked to Benchmarker rewards (where imbalance is heavily penalised):

$$innovator\textunderscore{ }reward \propto \sum_{benchmarkers} benchmarker\textunderscore{ }reward \cdot algorithm\textunderscore{ }\\%qualifiers$$

Where $algorithm\textunderscore{ }\\%qualifiers$ is the fraction of qualifiers found by a Benchmarker using a particular algorithm (the algorithm submitted by the Innovator)
