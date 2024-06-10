# 3. Innovators

Innovators are players in TIG who optimise existing proof-of-work algorithms and/or invent new ones, contributing them to TIG in the hope of earning token rewards.

This chapter covers the following topics:

1. The two types of algorithm submissions
2. Mechanisms for maintaining a decentralised repository
3. How algorithms are executed by Benchmarkers
4. How algorithms earn token rewards

## 3.1. Types of Algorithm Submissions

There are two types of algorithm submissions in TIG:

1. Code submissions
2. Breakthrough submissions

**Code submissions** encompass porting an existing algorithm for use in TIG, optimising the performance of an algorithm previously submitted by another Innovator, or an implementation of an entirely new algorithm. Code submissions must implement a solve_challenge function.

Presently, code submissions are restricted to Rust, automatically compiled into WebAssembly (WASM) for execution by Benchmarkers. Rust was chosen for its performance advantages over other languages, enhancing commercial viability of algorithms contributed to TIG, particularly in high-performance computing. Future iterations of TIG will support additional languages compilable to WASM.

**Breakthrough submissions** involve the introduction of novel algorithms tailored to solve TIG's proof-of-work challenges. A breakthrough submission will often yield such a significant performance enhancement that even unoptimised code of the new algorithm outpaces the most optimised code of an existing one.

Note: Support for breakthrough submissions is not currently in place but will be available in the coming months (pending a sufficiently wide token distribution).

## 3.2. Decentralised Repository

Algorithms are contributed to a repository without a centralised gatekeeper. TIG addresses crucial issues such as spam and piracy to ensure fair rewards for Innovators based on performance, maintaining a strong incentive for innovation.

To combat spam, Innovators must pay a submission fee of 0.001 ETH, burnt by sending it to the null address (0x0000000000000000000000000000000000000000). In the future, this fee will be denominated in TIG tokens.

To address the possibility of piracy and to provide an opportunity for IP protection, TIG implements a “push delay” and “merge points” mechanism:

### 3.2.1. Push Delay Mechanism

Upon submission, algorithms are committed to their own branch and pushed to a private repository. Following successful compilation into WebAssembly (WASM), a delay of 3 rounds ensues before the algorithm is made public where the branch is pushed to TIG’s public repository. This delay safeguards Innovators' contributions, allowing them time to benefit before others can optimise upon or pirate their work.

Notes:

- Confirmation of an algorithm's submission occurs in the next block, determining the submission round.
- An algorithm submitted in round X is made public at the onset of round X + 2.

### 3.2.2. Merge Points Mechanism

This mechanism aims to deter algorithm piracy. For every block in which an algorithm achieves at least 25% adoption, it earns a merge point alongside a share of the block reward based on its adoption.

At the end of each round, the algorithm from each challenge with the most merge points (exceeding a minimum threshold of 5,040) is merged into the repository's main branch. Merge points reset each round.

Merged algorithms, as long as their adoption is above 0%, share in block rewards every block.

The barrier for an Innovator contribution to be merged is intentionally chosen to be relatively high to minimise the likely payoff for pirating algorithms.

For algorithmic breakthrough submissions, the vote for recognising the algorithm as a breakthrough starts only when its code gets merged (details to come). This barrier is based on TIG’s expectation that breakthroughs will demonstrate distinct performance improvements, ensuring high adoption even in unoptimised code.

## 3.3. Deterministic Execution

Algorithms in TIG are compiled into WebAssembly (WASM), facilitating execution by a corresponding WASM Virtual Machine. This environment, based on wasmi developed by Parity Technologies for blockchain applications, enables tracking of fuel consumption, imposition of memory limits, and has tools for deterministic compilation.

Benchmarkers must download the WASM blob for their selected algorithm from TIG's repository before executing it using TIG's WASM Virtual Machine.

Notes:

- The WASM Virtual Machine functions as a sandbox environment, safeguarding against excessive runtime, memory usage, and malicious actions.
- Advanced Benchmarkers may opt to compile algorithms into binary executables for more efficient nonce searches, following thorough vetting of the code.

### 3.3.1. Runtime Signature

As an algorithm is executed by TIG's WASM Virtual Machine, a "runtime signature" is updated every opcode using the stack variables. This runtime signature is unique to the algorithm and instance of challenge, and is used to verify the algorithm used by Benchmarkers in their settings.

## 3.4. Sharing in Block Rewards

TIG incentivises algorithm contributions through block rewards:

- 15% of block rewards are allocated evenly across challenges with at least one "pushed" algorithm before distributing pro-rata based on adoption rates.
- In the future, a fixed percentage (we intend 15% of block rewards, see below) will be assigned to the latest algorithmic breakthrough for each challenge. In the absence of a breakthrough, this percentage reverts back to the Benchmarkers' pool. Given the expected relative rarity of algorithmic breakthroughs (compared to code optimisations), this represents a significant reward, reflecting TIG's emphasis on breakthrough innovations.
- When the rewards stream for algorithmic breakthroughs is introduced, there will be a total of 30% of block rewards for Innovators and 70% for Benchmarkers. Over time, we intend for the percentage of block rewards for Innovators to approach 50%.
