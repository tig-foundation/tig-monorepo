# 5. Optimisable Proof-of-Work

Optimisable proof-of-work (OPoW) distinctively requires multiple proof-of-works to be featured, “binding” them in such a way that optimisations to the proof-of-work algorithms do not cause instability/centralisation. This binding is embodied in the calculation of influence for Benchmarkers. The adoption of an algorithm is then calculated using each Benchmarker’s influence and the fraction of qualifiers they computed using that algorithm.

## 5.1. Influence

OPoW introduces a novel metric, imbalance, aimed at quantifying the degree to which a Benchmarker spreads their computational work between challenges unevenly. This is only possible when there are multiple proof-of-works.

The metric is defined as $imbalance = \frac{CV(\bf{\%qualifiers})^2}{N-1}$ where CV is coefficient of variation, %qualifiers is the fraction of qualifiers found by a Benchmarker across challenges, and N is the number of active challenges. This metric ranges from 0 to 1, where lower values signify less centralisation.

Penalising imbalance is achieved through $imbalance\textunderscore{ }penalty = 1 - exp(-k \cdot imbalance)$, where k is a coefficient (currently set to 1.5). The modifier ranges from 1 to 0, where 0 signifies no penalty.

When block rewards are distributed pro-rata amongst Benchmarkers based on influence, where influence has imbalance penalty applied, the result is that Benchmarkers are incentivised to minimise their imbalance as to maximise their earnings:

$$weight = mean(\bf{\%qualifiers}) \cdot (1 - imbalance\textunderscore{ }penalty)$$

$$influence = \frac{\bf{weights}}{sum(\bf{weights})}$$

Where:
* $\bf{\%qualifiers}$ is particular to a Benchmarker, where elements correspond to the fraction of qualifiers found by a Benchmarker for a challenge
* $\bf{weights}$ is a set, where elements correspond to the weight for a particular Benchmarker

Notes:

- A Benchmarker focusing solely on a single challenge will exhibit a maximum imbalance and therefore maximum penalty.
- Conversely, a Benchmarker with an equal fraction of qualifiers across all challenges will demonstrate a minimum imbalance value of 0.

## 5.2. Adoption

Any active solution can be assumed to already have undergone verification of the algorithm used. This allows the straightforward use of Benchmarkers' influence for calculating an algorithm's weight:

$$weight = sum(\bf{influences} \cdot \bf{algorithm\textunderscore{ }\%qualifiers})$$

Where:
* $\bf{influences}$ is a set, where elements correspond to the influence for a particular Benchmarker
* $\bf{algorithm\textunderscore{ }\%qualifiers}$ is a set, where elements correspond to the fraction of qualifiers found by a Benchmarker using a particular algorithm

Then, for each challenge, adoption is calculated: 

$$adoption = \frac{\bf{weights}}{sum(\bf{weights})}$$

Where:
* $\bf{\%weights}$ is a set, where elements correspond to the weight for a particular algorithm

By integrating influence into the adoption calculation, TIG effectively guards against potential manipulation by Benchmarkers.
