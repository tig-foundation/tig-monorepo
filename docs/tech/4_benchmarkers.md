# 4. Benchmarkers

Benchmarkers are players in TIG who continuously select algorithms to compute solutions for challenges and submit them to TIG through benchmarks and proofs to earn block rewards.

This chapter covers the following topics:

1. How solutions are computed
2. How solutions are submitted
3. How solutions are verified
4. How solutions earn block rewards

## 4.1. Computing Solutions

The process of benchmarking comprises 3 steps:

1. Selecting benchmark settings
2. Generate challenge instances
3. Execute algorithm on instances and record solutions

Apart from algorithm selection, this process is entirely automated by the browser benchmarker.

### 4.1.1. Benchmark Settings

A Benchmarker must select their settings, comprising 5 fields, before benchmarking can begin:

1. Player Id
2. Challenge Id
3. Algorithm Id
4. Block Id
5. Difficulty

**Player Id** is the address of the Benchmarker. This prevents fraudulent re-use of solutions computed by another Benchmarker.

**Challenge Id** identifies the proof-of-work challenge for which the Benchmarker is attempting to compute solutions. The challenge must be flagged as active in the referenced block. Benchmarkers are incentivised to make their selection based on minimising their imbalance. Note: Imbalance minimisation is the default strategy for the browser benchmarker.

**Algorithm Id** is the proof-of-work algorithm that the Benchmarker wants to use to compute solutions. The algorithm must be flagged as active in the referenced block. Benchmarkers are incentivised to make their selection based on the algorithm’s performance in computing solutions.

**Block Id** is a reference block from which the lifespan of the solutions begins counting down. Benchmarkers are incentivised to reference the latest block as to maximise the remaining lifespan of any computed solutions.

**Difficulty** is the difficulty of the challenge instances for which the Benchmarker is attempting to compute solutions. The difficulty must lie within the valid range of the challenge for the referenced block. Benchmarkers are incentivised to make their selection to strike a balance between the number of blocks for which their solution will remain a qualifier, and the number of solutions they can compute. (e.g. lower difficulty may mean more solutions, but may lower the number of blocks that the solutions remain qualifiers)

### 4.1.2. Unpredictable Challenge Instances

TIG makes it intractable for Benchmarkers to attempt to re-use solutions by:

1. Challenge instances are deterministically pseudo-randomly generated, with at least $10^{15}$ unique instances even at minimum difficulty.
2. Instance seeds are computed by hashing benchmark settings and XOR-ing with a nonce, ensuring randomness.

During benchmarking, Benchmarkers iterate over nonces for seed and instance generation.

### 4.1.3. Algorithm Execution

Active algorithms reside as compiled WebAssembly (WASM) blobs in TIG's open repository.

https://raw.githubusercontent.com/tig-foundation/tig-monorepo/**_&lt;branch&gt;_**/tig-algorithms/wasm/**_&lt;branch&gt;_**.wasm

where &lt;branch&gt; is &lt;challenge_name&gt;/&lt;algorithm_name&gt;

Benchmarkers download the relevant WASM blob for their chosen algorithm, execute it using TIG's WASM Virtual Machine with specified seed and difficulty inputs.

If a solution is found, the following data is outputted:

1. Nonce
2. Runtime signature
3. Fuel consumed
4. Serialised solution

From this data, Benchmarkers compute the solution signature and retain the solution only if it meets the challenge's threshold.

## 4.2. Submitting Solutions

The process of submitting solutions comprises 4 steps:

1. Submit the benchmark
2. Await probabilistic verification
3. Submit the proof
4. Await submission delay

### 4.2.1. Submitting Benchmark

A benchmark, a lightweight batch of valid solutions found using identical settings, includes:

- Benchmark settings
- Metadata for solutions
- Data for a single solution

**Benchmark settings** must be unique, i.e. the same settings can only be submitted once.

**Metadata** for a solution consists of its nonce and solution signature. Nonces must be unique and all solution signatures must be under the threshold for the referenced challenge & block.

**Data** for a solution consists of its nonce, runtime signature, fuel consumed and the serialised solution. The solution for which data must be submitted is randomly sampled. TIG requires this data as Sybil-defence against fraudulent benchmarks.

### 4.2.2. Probabilistic Verification

Upon benchmark submission, it enters the mempool for inclusion in the next block. When the benchmark is confirmed into a block, up to three unique nonces are sampled from the metadata, and corresponding solution data must be submitted by Benchmarkers.

TIG refers to this sampling as probabilistic verification, and ensures its unpredictability by using both the new block id and benchmark id in seeding the pseudo-random number generator. Probabilistic verification not only drastically reduces the amount of solution data that gets submitted to TIG, but also renders it irrational to fraudulently "pad" a benchmark with fake solutions:

If a Benchmarker computes N solutions, and pads M fake solutions to the benchmark for a total of N + M solutions, then the chance of getting away with this is $\left(\frac{N}{N+M}\right)^3$. The expected payoff for honesty (N solutions always accepted) is always greater than the payoff for fraudulence (N+M solutions sometimes accepted):

$$N > (N+M) \cdot \left(\frac{N}{N+M}\right)^3$$

$$1 > \left(\frac{N}{N+M}\right)^2$$

Note that N is always smaller than N + M.

### 4.2.3. Submitting Proof

A proof includes the following fields:

- Benchmark id
- Array of solution data

**Benchmark id** refers to the benchmark for which a proof is being submitted. Only one proof can be submitted per benchmark.

**Array of solution data** must correspond to the nonces sampled from the benchmark’s solutions metadata.

### 4.2.4. Submission Delay & Lifespan mechanism

Upon confirmation of a proof submission, a submission delay is determined based on the block gap between when the benchmark started and when its proof was confirmed.

A submission delay penalty is calculated by multiplying the submission delay by a multiplier (currently set to 3). If the penalty is X and the proof was confirmed at block Y, then the benchmark’s solutions only become “active” (eligible to potentially be qualifiers and share in block rewards) from block X + Y onwards.

As TIG imposes a lifespan, the maximum number of blocks that a solution can be active (currently set to 120 blocks), there is a strong incentive for Benchmarkers to submit solutions as soon as possible.

## 4.3. Verification of Solutions

Two types of verification are performed on solutions submitted to TIG to safeguard algorithm adoption against manipulation:

1. Verification of serialised solutions against challenge instances, triggered during benchmark and proof submission.
2. Verification of the algorithm that the Benchmarker claims to have used, involving re-running the algorithm against the challenge instance before checking that the same solution data is reproduced.

If verification fails, the benchmark is flagged as fraudulent, disqualifying its solutions. In the future (when Benchmarker deposits are introduced) a slashing penalty will be applied.

## 4.4. Sharing in Block Rewards

Every block, 85% of block rewards are distributed pro-rata amongst Benchmarkers based on influence. A Benchmarker’s influence is based on their fraction of qualifying solutions across challenges with only active solutions eligible.

### 4.4.1. Cutoff Mechanism

To strongly disincentivise Benchmarkers from focusing only on a single challenge (e.g. benchmarking their own algorithm), TIG employs a cutoff mechanism. This mechanism limits the maximum qualifiers per challenge based on an average number of solutions multiplied by a multiplier (currently set to 1.5).

The multiplier is such that the cutoff mechanism will not affect normal benchmarking in 99.9% of cases.
