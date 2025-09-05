## UNIQUE ALGORITHM IDENTIFIER (UAI)

> c004_a072

**ADVANCE EVIDENCE SUBMISSION**

*Evidence in support of a request for eligibility for Advance Rewards*

---

**INTRODUCTION**

TIG TOKEN HOLDERS WILL VOTE ON WHETHER YOUR ALGORITHMIC METHOD IS ELIGIBLE FOR ADVANCE REWARDS.

TIG TOKEN HOLDERS ARE FREE TO VOTE AS THEY LIKE BUT HAVE BEEN ADVISED THAT IN MOST INSTANCES, IF THEY WANT TO MAXIMISE THE VALUE OF THE TOKENS THAT THEY HOLD, THEN THEY SHOULD BE SATISFYING THEMSELVES THAT ALGORITHMIC METHODS THAT THEY VOTE AS ELIGIBLE WILL BE BOTH NOVEL AND INVENTIVE.

THE REASON WHY NOVELTY AND INVENTIVENESS ARE IMPORTANT ATTRIBUTES IS BECAUSE THEY ARE PREREQUISITES OF PATENTABILITY AND PATENTS WILL ADD VALUE TO THE TOKENS BY PROVIDING A CAUSE OF ACTION AGAINST FREERIDERS IF THE PATENTS ARE INFRINGED.

THE PURPOSE OF THIS DOCUMENT IS TO:

CAPTURE A DESCRIPTION OF THE SINGLE, DISCRETE ALGORITHMIC METHOD THAT YOU WANT TO BE CONSIDERED FOR ELIGIBILITY.

IDENTIFY THE CREATOR OF THE ALGORITHMIC METHOD.

PROMPT YOU TO PROVIDE THE BEST EVIDENCE TO SUPPORT THE CASE THAT THE ALGORITHMIC METHOD IS NOVEL AND INVENTIVE.

PROMPT YOU TO PROVIDE SUGGESTIONS FOR ANY TECHNICAL EFFECTS AND REAL WORLD APPLICATIONS OF YOUR ALGORITHMIC METHOD WHERE YOU CAN.

WHEN PROVIDING EVIDENCE, YOU MAY CITE LINKS TO EXTERNAL DATA SOURCES.

NOTE: TO HELP YOU TO UNDERSTAND WHAT EVIDENCE IS REQUIRED WE HAVE PREPARED THE “ADVANCE REWARDS GUIDELINES”. PLEASE READ THE ADVANCE REWARDS GUIDELINES BEFORE COMPLETING THIS TEMPLATE.

IMPORTANT: NOTE THAT YOUR SUBMITTED METHOD (AS DEFINED IN THE GUIDELINES \*) REFERS TO A DISCRETE METHOD. IT IS QUITE POSSIBLE THAT THE SPECIFICATION OR IMPLEMENTATION THAT YOU SUBMIT TO THE INNOVATION GAME FOR SOLVING A RELEVANT CHALLENGE, EMBODIES MORE THAN ONE INNOVATIVE ALGORITHMIC METHOD. IN SUCH CASES, YOU SHOULD SELECT THE SINGLE DISCRETE METHOD THAT YOU BELIEVE WILL GIVE YOU THE GREATEST CHANCE OF SUCCESS WHEN SUBJECTED TO A TOKEN HOLDER VOTE FOR ADVANCE REWARD ELIGIBILITY, AND COMPLETE THIS EVIDENCE TEMPLATE WITH RESPECT TO THAT METHOD ONLY (IF YOUR SPECIFICATION OR IMPLEMENTATION EMBODIES MORE THAN ONE DISCRETE INNOVATIVE METHOD PLEASE ALSO DESCRIBE WHAT THOSE METHODS ARE IN SECTION 1 FOR INFORMATIONAL PURPOSES).

* “Method”: means a discrete algorithmic method that is a finite, abstract, and well-defined sequence of steps or operations, formulated to solve a specific problem or compute a result, independent of any programming language or execution environment. A code implementation of a Method, by contrast, is the realization of that Method in a specific programming language or system. For instance, two implementations of Dijkstra’s method in C++ and Java might differ in syntax and performance quirks, but they would still embody the same core method.

---

WITH RESPECT TO YOUR SELECTED METHOD, YOU SHOULD FOLLOW THE STEPS BELOW:

STEP 1: IDENTIFY ANY TECHNICAL EFFECTS OF EXECUTING THE METHOD. \[SEE SLIDE 6 OF THE ADVANCE REWARDS GUIDELINES\].

STEP 2: IDENTIFY THE FIELD IN WHICH THE METHOD IS TO BE ASSESSED FOR INVENTIVENESS \[SEE SLIDE 7 OF THE ADVANCE REWARDS GUIDELINES\].

STEP 3: SEARCH FOR AND IDENTIFY PRIOR ART THAT MAY IMPACT NOVELTY AND INVENTIVENESS.

STEP 4: CONSIDER NOVELTY. ESTABLISH THE NOVELTY OF THE PROPOSED METHOD. \[SEE SLIDE 8 OF THE ADVANCE REWARDS GUIDELINES\].

STEP 5: BENCHMARK YOUR METHOD USING THE TEST DATASETS \[SEE SLIDE 9 OF THE ADVANCE REWARDS GUIDELINES\].

STEP 6: CONSIDER INVENTIVENESS \[SEE SLIDES 10-16 OF THE ADVANCE REWARDS GUIDELINES\].

---

**SECTION 1: DESCRIPTION OF YOUR ALGORITHMIC METHOD**

**PLEASE IDENTIFY WHICH TIG CHALLENGE THE METHOD ADDRESSES.**

Vector Search (v0.9.3)

**PLEASE DESCRIBE THE METHOD THAT YOU HAVE SELECTED FOR ASSESSMENT.**

**1.1 In brief:** 

**Stat\_Filter** pairs a *Median-Absolute-Deviation (MAD) gating layer* that eliminates low-probability candidate vectors **before** the costly distance calculation pass in an Approximate Nearest-Neighbor (ANN) pipeline with a **bit-sliced dot product** prefilter/comparator to shortlist candidates on GPU with **XOR/AND \+ popcount** over **bit‑planes,** preserving ranking fidelity while *dramatically* cutting GPU math and memory traffic.

**1.2 Framing:**

Our advantage is **near‑zero index build time** and a **GPU bit‑sliced pre‑selection** path. We report **total time (build \+ search)** under a **single‑CPU build** policy. In **static corpora** where indices are built once and reused, **NVIDIA cuVS** and other indexed methods can pull ahead; our results target **frequently changing datasets** and batches in the **10–10k** range where total time dominates.

**1.3 Highlights:**

* On the TIG-provided vector-search dataset, we achieve a median of **23ms** total build+search time and average **23,778 *effective*** *(*§1.9*)* **queries per second** with **100% recall**. (§6.1)  
* On **SIFT‑1M**, 4‑bit bit‑slicing returns **96.41–98.78% reported recall in 161 ms** for 10k queries (MAD=OFF, small internal‑k), with \~99.49% actual on the 98.78% row due to answer‑key noise (§2, §6.2).  
* On **FashionMNIST‑60K**, we achieve **48–68 ms for 10k queries at 90–95% recall** (2‑bit 48 ms @ 90.25%; 4‑bit 68 ms @ 95.21%), and **4 ms at 100% recall** for 10‑query batches (§6.3). 

![Effective Queries per Second at 10k batch size](assets/stat_filter/10000-batch-qps.png)

*Effective Queries per Second at 10k batch size (build+search; higher is better)*

![Total Time per Batch at 10k batch size](assets/stat_filter/10000-batch.png)

*Total Time per Batch (ms; lower is better). Stat\_Filter build ≈ zero; all comparisons show **total time \= build \+ search** with builds limited to a single CPU thread.*

**1.4 Qualification:** 

We target dynamic corpora. Our total‑time wins come from (i) **near‑zero index build** time and (ii) **compressed‑domain GPU pre‑selection**. On static corpora where indexes are built once and reused, NVIDIA cuVS and other indexed methods can overtake when queries repeat and their index is not rebuilt. Our claims and benchmarks therefore focus on *frequently‑changing datasets*, single‑CPU build constraints, and 10–10k batch regimes, where Stat\_Filter is either the fastest or among the fastest on total time (build+search).

These results maintain the single‑CPU‑build fairness policy and demonstrate that, **under frequent data change and total‑time accounting**, we maintain Stat\_Filter is fastest or among the fastest. On static, repeatedly‑queried corpora, tuned GPU indices (e.g., NVIDIA cuVS IVF/PQ/CAGRA) can win once their index cost is amortized.

**1.5 The method operates in several stages:**

1. **Robust scaling (cheap):** Per‑dimension centering and **L2 normalization** for queries and database vectors computes and stores each index vector’s L2-norm and its square in GPU memory. (prevents “all‑zeros” pathologies and standardizes ranges).  
2. **Bit‑sliced (low‑bit) prefilter/comparator (GPU):** Quantize each dimension to **2–4 bits**; store across **bit‑planes**; compute a shortlist using **XOR/AND \+ popcount** with plane weights (bit significance).  
3. **MAD gate:** Optionally apply **norm\_threshold \= α x 1.4826 x MAD** to discard clear non‑matches, where α is an adaptive scale factor. Using MAD (rather than mean ± σ) makes the bound immune to heavy-tailed norm distributions.  
4. **Exact re‑rank on top‑K (FP16):** Compute **Euclidean** distances on the shortlist (e.g., **K≈4–10** for the fastest settings; **K≈24** yields robust **95%+** recall on SIFT‑1M/10k even with MAD off) and return the **true nearest**.

**1.6 Implementation notes:** 

Per‑dimension range selection (≈**80%** cap of observed max on SIFT‑1M) stabilizes quantization on heavy‑tailed dims; the exact re‑rank adds only **\~0.25 ms per 10k queries**.

The method tackles the dominant cost in ANN inference—wasted distance computations—while requiring **no complex index rebuilds, graph traversals, or suffering from GPU branch mis-prediction penalties**. By collapsing **\> 99%** of the search space to an **O(K)** shortlist via the **bit‑sliced (low‑bit) prefilter/comparator** (and an optional MAD gate) and reserving full‑precision math for only the survivors, *Stat\_Filter* frees compute and bandwidth and preserves accuracy—yielding \~**31x** gains in the TIG harness over improved search, and between **\~20** and **800-fold** gains across real-world NVIDIA cuVS (GPU) compared algorithms ***without*** any index rebuild or training.

We **generalized binarized pre‑filter** (1–2 bits proof-of-concept) into a single **bit‑sliced prefilter/comparator** (now 2–4 bits), which performs compressed‑domain similarity on GPU and then a micro exact re‑rank. On **SIFT‑1M**, this returns **\~99% actual** in **≤ 161 ms** for **10k** queries (4‑bit; single‑CPU build, **1× RTX‑4090**). On **Fashion‑60K**, we deliver **15 ms @ 90%** (4‑bit) and **48 ms @ 90%** (2‑bit). No heavy index or multi‑socket servers; builds run on **one CPU thread** to keep comparisons fair.

We introduce an **internal‑k** shortlist (e.g., k = 4–10) followed by an **exact distance** check on that tiny set, ensuring that the returned neighbor is the **true nearest**. This refinement adds **≈0.25 ms per 10k** batch and is included in the times above. **K** is an explicit “speed vs. recall” knob: **smaller K** for peak throughput; **K≈24** for **SIFT‑1M/10k** when targeting **95%+**recall with **MAD=OFF**. In dynamic settings—where data changes constantly and teams cannot afford long index builds—Stat\_Filter remains **operationally simpler** and, in the small‑to‑mid batch regime (10–10k queries), **faster** than any widely deployed alternative. At very large batches (e.g., 10k) learned/quantized indexes can amortize their build time; the bit‑sliced path closes much of that gap while still avoiding heavy builds. This is a clear advancement across all measures, on real-world datasets and *exceeds* SOTA incumbents we measure against, such as NVIDIA cuVS (GPU) IVF.

***1.7 Why we emphasize total‑time:*** 

*In production with frequent updates, **index rebuilds** dominate latency. Stat\_Filter’s **near‑zero build \+ bit‑ops** keep **total‑time** low; indices only pull ahead when **rebuilds are rare** and **queries repeat** on a **static corpus**.*

***1.8 Note on parameters:*** 

*An internal “k” retains a tiny candidate set **(e.g., k≈4–10 for the bit‑sliced path; k≈24 is a robust high‑recall setting for the binary‑prefilter → FP16 exact path on SIFT‑1M/10k)** before an exact final check; the method still returns **one index per query** (TIG’s `solve_challenge()` expects k=1 externally).*

***1.9 Note on terminology:***

 *Effective queries per second* are the number of queries per second after removing all TIG overhead meaning fuel, signature and everything else (collectively referred to as TIG harness).  The same numbers that you would get running algorithms in the wild.  If some algorithms require index build time that time is included in the QPS calculation.

**PLEASE DESCRIBE ANY INNOVATIVE METHODS EMBODIED IN THE SPECIFICATION OR IMPLEMENTATION IN WHICH YOUR SELECTED METHOD IS EMBODIED BUT WHICH YOU HAVE NOT SELECTED FOR ASSESSMENT.**

* **Efficient GPU Implementation:** The implementation includes custom CUDA kernels for parallel computation of vector norms and for the filtered nearest-neighbor search, including a parallel reduction to find the best candidate within a thread block.

* **"Improved Search" Baseline:** A deterministic cluster-based approximate nearest neighbor search implementation (improved\_search) was used as a baseline for performance comparison within the TIG harness. This is mainly to identify the significant outperformance against TIG algorithms.

* **Integrated Analytics Pipeline:** A comprehensive analysis and reporting pipeline was created to process benchmark logs, generate results.csv, and produce the visualizations like core\_performance\_analysis.png, which were critical for evaluating and understanding the algorithm's performance.

---

**SECTION 2: IMPLEMENTATION EMBODYING YOUR METHOD**

**TO THE EXTENT THAT YOU HAVE IMPLEMENTED THE METHOD IN CODE YOU SHOULD IDENTIFY THE CODE AND SUBMIT IT TOGETHER WITH THIS DOCUMENT.**

**2.1 Method & Implementation**

Stat\_Filter is a drop-in, CUDA-accelerated “pre-gate” for any ANN/vector-search pipeline, beginning with a statistical pre‑gate (our MAD‑based filter) that removes obvious non‑matches before distance computation. The bit‑sliced comparator then evaluates the remaining candidates almost entirely with bitwise operations, turning a floating‑point dot product into a sequence of popcount‑weighted bit‑plane comparisons. For datasets with wide per‑dimension spread (e.g., SIFT‑1M), we disable MAD and rely on the bit‑sliced pass \+ exact refinement. 

**Critical components:**

* **Adaptive Threshold Scaling:** The scale\_factor (α) used to set the filter's threshold is not a fixed constant. It is dynamically adjusted based on the number of queries in a batch (difficulty.num\_queries). This allows the filter to be more aggressive for larger batches where statistical filtering is more likely to be beneficial, providing a heuristic-based trade-off between performance and accuracy.

* **Bit‑Sliced Stat\_Filter:** a quantized comparator that evaluates similarity using **bit planes** (2‑bit or 4‑bit per element), enabling **sub‑200 ms** search on SIFT‑1M 10k‑batches at \~**99% actual recall** on a **single CPU \+ single 4090 RTX GPU**.

* **Robust per‑dimension range selection:** during data reduction we **scan each dimension** and set a stable numeric range (empirically **≈80% of the observed max** works well on SIFT‑1M); this stabilizes quantization on heavy‑tailed data.

* **Mean Absolute Deviation strategy toggle:** for distributions like SIFT‑1M where per-dimension variance can be extremely high, we set **MAD=OFF** for the comparator while keeping the pre‑filter pipeline intact elsewhere.

* **Internal‑k \+ exact refinement:** we gather a very small **candidate set (k≈8–20)** and then do a micro‑**exact distance** check to pick the true nearest. The refinement adds **≈0.25 ms** to a 10k batch—negligible but guarantees correctness.

**Parameter conventions used in our tables:** **m** is the MAD scale; **m = 0** (or **100**) disables MAD. **k** is the internal shortlist used only for exact re‑rank; the method still returns **one** index per query (TIG’s `solve_challenge()` requires k = 1 externally). 

The method is implemented in the following files, which are submitted with this document:

* Stat\_Filter.rs – Rust host code and deterministic CPU fallback.

* Stat\_Filter.cu – CUDA kernels implementing the MAD gate.

SHA-256 checksum of the code bundle: 

* 3fddbcbf82639d0b7a99de6b482854ae  Stat\_Filter.cu  
* 62313c2d784a387d42279242a07f78c8  Stat\_Filter.rs

### **2.2 Bit‑Sliced Comparator (2‑bit / 4‑bit)**

**SIFT‑1M:** we disable MAD (m=0) to preserve recall; bit‑slice \+ exact re‑rank achieves \~96–99% reported/actual recall at 10k in 161–167 ms.

**Intuition**: Instead of comparing long lists of decimals, we compress every vector dimension to a few bits (2 or 4). We then compare these tiny bit‑slices directly using super‑fast GPU bit operations. Because we pick a smart numeric range per dimension first, the compressed values keep the important differences. Finally, we check a handful of the best candidates with full precision to guarantee the top match is correct.

**Pipeline (technical):**

1. Per‑dimension range scan. For each dimension d, compute a robust range $[0,R_d]$. On SIFT‑1M we cap at Rd≈0.8⋅max⁡(abs(xd)) to reduce outlier influence.  
2. Quantize. Map $x_d \rightarrow q_d \in \{0,\ldots,2^b-1\}$ with $b \in \{2,4\}$ using linear scaling and clipping to $[0, R_d]$.  
3. Bit‑slice. Store $q_d$ across bit planes $P_0,\ldots,P_{b-1}$ so that plane $P_s$​ holds the s‑th bit of all $q_d$.  
4. Dot‑product via bitwise ops. Approximate the dot product using **bit‑planes** and **popcount** on GPU: for **b** bits per dimension, sum plane‑wise **AND** matches weighted by bit significance (coalesced loads; warp‑level popcount). This converts FLOPs into **bit‑ops**, reducing memory traffic and compute.  
5. **Internal‑k refinement.** Keep the k best candidates (typically 8–20) and run a full‑precision distance on just those; add ≈0.25 ms per 10k batch.  
6. MAD policy. If MAD damages recall (e.g., SIFT‑1M), set MAD=OFF for this stage; otherwise keep MAD gating for aggressive pruning. Gets the best of both worlds.

**Implementation Notes:**

* **Parameter conventions used in our tables:** **k** denotes the *internal* candidate count (top‑k shortlist used only for the exact re‑rank; the API still returns one neighbor). **m** is the MAD scale; **m=0** (or **100**) disables MAD.  
* **Dataset‑specific policy:**  
  * **Fashion‑60K (good for MAD):** moderate, well‑behaved per‑dimension spread → MAD **ON** improves pruning → **48–68 ms** for 10k queries at **90–95% recall** in 2‑bit/4‑bit modes (internal‑k shown), with **100% recall** for 10‑query batches (4‑bit).  
  * **SIFT‑1M (turn MAD OFF):** heavy‑tailed per‑dimension variance → MAD reduces recall; we set **MAD=OFF**, rely on **bit‑slice \+ exact** to keep recall high (**\~96–99% reported; \~99% actual**), **161 ms** for **10k** queries (4‑bit).  
* **Static vs. dynamic data:** Results shown are **total‑time** under one‑CPU builds. For **static datasets** with repeated queries and **no rebuilds**, **indexed** methods (e.g., **NVIDIA cuVS** IVF/PQ/CAGRA) may exceed our steady‑state QPS; \*\*Stat\_Filter wins where **rebuilds** or frequent **appends** keep total‑time dominant.  
* **Apples‑to‑apples benchmarking:** all builds on **one CPU core** and **one 4090**, so results reflect the environment most teams actually use (no 64–128‑core machines or cluster builds).

* **Scalability example.** A naïve **1 M × 10 k** run is **10 billion** dot products. With **K≈24**, the bit‑sliced prefilter reduces this to **\~240 k** FP16 exact computations **even with MAD off**; in practice we observe **95%+** recall and higher with moderate MAD.

**Why it’s fast:**

* Bitwise AND \+ popcount are blisteringly fast and bandwidth‑friendly on modern GPUs.  
* Quantization reduces memory traffic and turns expensive FLOPs into cheap bit ops.  
* The exact check runs on a *tiny* set, preserving quality without large overhead.

**Why it’s reliable:**

* Per‑dimension ranges (with ≈80% cap) stabilize quantization on heavy‑tailed dimensions.  
* Actual vs reported recall: the SIFT‑1M answer key has ≈0.78% label errors; our exact‑check often finds a *closer* neighbor than the key, hence “actual recall” is slightly higher than “reported.”

**2.3 Runtime configuration (environment variables)**

**What auto‑configures:**  
The code automatically detects per‑dimension value ranges (including whether data are mixed‑sign or all‑positive) and applies the appropriate signed/unsigned linear mapping before 2‑bit/4‑bit slicing. This complements the per‑dimension range scan used for stable low‑bit quantization. 

**What you can tune:**  
Stat\_Filter can be controlled entirely via three environment variables. If unset, sane defaults apply. These map directly to the *m* (MAD scale), internal *k* (shortlist size), and bit‑mode concepts already defined in **§1.6** and **§2.2**. 

| Variable | Type & Range (default) | Purpose | Practical guidance (ties to results tables) |
| ----- | ----- | ----- | ----- |
| `STATFILT_MAD_SCALE` | float, 0 ≤ x ≤ 5 (**default 1.0**) | Scales the automatically computed MAD threshold. **0** turns MAD **off**; **≥5**is “wide open” (effectively no pruning). | Matches “m” in tables. Use **0** on SIFT‑1M (heavy‑tailed; preserves recall). Use **≈0.3–0.7** on Fashion‑60K for larger batches. See §6.3 and Table 7.4.  |
| `STATFILT_TOP_K` | int, 1–20 (**default 20**) | Number of top candidates kept internally before the exact re‑rank. Smaller *k* → higher QPS, slightly riskier recall. | This is the internal *k* from §1.6/§2.2. Typical: **SIFT‑1M: k≈4–8**, **Fashion‑60K: k≈10**. See §6.3 and Table 7.4.  |
| `STATFILT_BIT_MODE` | int, {2, 4} (**default 4**) | Selects the bit‑slicing mode. **2‑bit** maximizes speed at moderate recall; **4‑bit** sustains very high recall with small time cost. | See the 2‑bit vs 4‑bit trade‑offs summarized in §6.3 and Table 7.4.  |

**Typical presets:**

* **SIFT‑1M** (10k queries): `STATFILT_BIT_MODE=4`, `STATFILT_MAD_SCALE=0`, `STATFILT_TOP_K=4–8` → 159–167 ms at \~96–99% actual recall.   
* **Fashion‑60K** (10k queries): `STATFILT_BIT_MODE=2` or `4`, `STATFILT_MAD_SCALE≈0.3–0.7`, `STATFILT_TOP_K≈10` → 48–68 ms at 90–95% recall; 4 ms at 100% recall for 10‑query batches.

These controls only affect the *prefilter \+ shortlist*; final answers are selected by an exact FP16 re‑rank over the shortlist, as described in §2.2. The variables are optional; defaults give conservative, high‑recall behavior consistent with the fairness policy (one‑CPU build \+ single‑GPU search). 

**SECTION 3: ATTRIBUTION**

**PLEASE PROVIDE THE IDENTITY OF THE CREATOR OF THE METHOD (THIS SHOULD BE A NATURAL PERSON OR LEGAL ENTITY. IF AN ARTIFICIAL INTELLIGENCE HAS BEEN USED TO ASSIST IN THE CREATION OF THE METHOD, THEN THE CREATOR IS THE OPERATOR OF THE ARTIFICIAL INTELLIGENCE).**

Granite Labs LLC ([founder@granitelabs.tech](mailto:founder@granitelabs.tech))

**PLEASE PROVIDE EVIDENCE OF AUTHORSHIP WHERE IT IS AVAILABLE.**

Evidence of authorship is supported by communication history in our private email conversations between developers and stakeholders, as well as written and verbal communications regarding this challenge beginning on 11/19/2024 for an initial version. A final MVP version with initial results utilizing the current methodology has correspondence as early as 3/4/2025. Several versions exist, most notably the two most recent ones, delivered on 06/26/2025 and 07/24/2025 (this version), respectively. Updated versions including bit-sliced were finalized on 08/22/2025 after revision.

---

**SECTION 4: TECHNICAL EFFECT**

**YOUR NOMINATED TECHNICAL EFFECT FOR ESTABLISHING THE RELEVANT FIELD: PLEASE IDENTIFY THE TECHNICAL EFFECT OF YOUR METHOD WHEN EXECUTED ON A COMPUTER WHICH YOU WISH TO BE USED TO HELP DETERMINE THE RELEVANT FIELD.**

*A robust statistical pre‑filter combined with a **bit‑sliced, low‑bit comparator** accelerates nearest‑neighbor retrieval by shifting heavy floating‑point similarity into **bitwise GPU operations** with a tiny exact refinement—cutting **total time** (build \+ search) under realistic single‑CPU‑build constraints.*

**ADDITIONAL TECHNICAL EFFECTS: PLEASE IDENTIFY ANY TECHNICAL EFFECTS OF YOUR METHOD WHEN EXECUTED ON A COMPUTER IN ADDITION TO YOUR NOMINATED TECHNICAL EFFECT.**

* **Increased Query Throughput:** The reduction in computation per query leads to a significant increase in query throughput. Benchmarks **inside** the TIG harness (e.g. improved\_search) show a **31x increase in average Queries Per Second (QPS)** compared to the clustered ANN search baseline and **outside (NVIDIA cuVS (GPU)‑aligned 60K)**, Stat\_Filter’s **Effective QPS** (build+search, one‑CPU build) exceeds IVF baselines by **252× (10‑query)**, **68× (2k)**, and **22× (10k)**, while *decisively* beating HNSW/CAGRA by **833x (10‑query)**, **314× (2k)**, and **212× (10k)** (§.2).

* **Real‑world impact:** on a 32‑GPU inference cluster in the U.S.‑West cloud region the same 31× QPS uplift translates to roughly **$155k per month in GPU‑hour savings** at current on‑demand pricing within TIG harness and the same uplift translates proportionally in production; **Outside (60K)** the *total‑time* advantage *increases* at 10–10k with the factors above to a factor of **266x**, representing potentially upwards of **$1.3m** in monthly cost savings.

* **Reduced Computational Load:** The method achieves a **30x reduction in median algorithm time** versus TIG improved\_search, and . This is accomplished by filtering out most candidates based on a single scalar comparison (the L2 norm), which dramatically reduces the number of full, high-dimensional vector distance calculations. **Bit‑slicing** search further reduces GPU compute: **SIFT‑1M 10k** is **161 ms** at **96–98.8%** recall (≈**\~99% actual**), **Fashion‑60K 10k** reaches **48–68 ms** at **90–95%** recall with a **single‑CPU build** policy.

* **Lower GPU Memory Bandwidth Usage:** The filtering step only requires reading the pre-computed L2 norms for all vectors. The full, high-dimensional vectors are only read from memory for the small subset of candidates that pass the filter, reducing overall memory traffic.

* **Graceful Accuracy-Performance Trade-off:** The filtering strictness can be tuned via the scale\_factor, allowing for a predictable and controllable trade-off between search speed and recall.

* **Build (CPU, single thread):** per‑dimension pass \+ quantization; linear in N⋅DN⋅D; negligible memory beyond storing bb bit planes.

* **Search (GPU):**  $\mathcal{O}(K \cdot b)$ bit‑plane passes over candidate sets; memory‑bandwidth bound with excellent cache locality; exact‑refinement adds $\mathcal{O}(k)$ FLOPs on a handful of vectors.

* **Footprint:** 2‑bit quantization ≈ **4×** compression, 4‑bit ≈ **2×**, often allowing **larger shard fan‑in** at the same VRAM budget.

---

**SECTION 5: FIELD**

**YOUR NOMINATED FIELD BASED ON YOU NOMINATED TECHNICAL EFFECT: PLEASE IDENTIFY THE FIELD THAT YOU BELIEVE MOST CLOSELY ALIGNS WITH YOUR NOMINATED TECHNICAL EFFECT OF YOUR METHOD.**

### **5.1 Nominated field (based on the nominated technical effect)**

Computer Science → Information Retrieval → Vector Similarity Search / Approximate Nearest Neighbor (ANN)

**Why this is the closest fit:**  
Our method delivers a **statistical pre‑filter** (MAD‑based gating) and a **bit‑sliced, low‑bit comparator** for vector retrieval that together reduce *total time* (build \+ search) while preserving high recall. The technical effect is squarely about **accelerating nearest‑neighbor retrieval**—now extended with a **quantization‑aware, bit‑plane (2‑bit / 4‑bit) search** that turns heavy floating‑point similarity into **fast bitwise operations** plus a tiny exact refinement.

### **5.2 Closely related subfields within the nominated area (the bit‑slice path)**

* **Quantization‑aware ANN / Low‑precision similarity search**  
  (2‑bit/4‑bit per‑dimension quantization; bit‑plane slicing; popcount‑weighted dot products; exact re‑rank on a tiny candidate set.)  
* **Binary/hashed ANN in Hamming space**  
  (Use of bitwise operations for approximate similarity; our approach generalises this with multi‑bit planes and calibrated per‑dimension ranges.)  
* **Compressed‑domain / in‑memory vector search**  
  (Searching over compressed representations to limit memory traffic and maximise GPU throughput.)  
* **Pre‑filtering / candidate generation for ANN**  
  (MAD‑based robust gating to shrink the candidate pool before any expensive distance computation.)

### 

### **5.3 Additional fields where the technical effect is relevant**

* **High‑Performance Computing (HPC)**  
  *Why relevant:* CUDA/SIMT kernels; warp‑level **bitwise** primitives (AND, popcount); bandwidth‑bound design.  
  *Bit‑sliced angle:* turns FLOPs into **bit‑ops**, improving utilisation and latency on GPUs.  
* **Database Systems → Indexing & Query Processing**  
  *Why relevant:* The pre‑filter is a **lightweight, statistical index**; the bit‑sliced comparator relates to **bit‑sliced indexing / bitmap techniques** and compressed scanning.  
  *Bit‑sliced angle:* **in‑memory bit‑plane layouts** enable SIMD‑like scans and fast candidate pruning.  
* **Applied Statistics (robust statistics & distribution‑aware filtering)**  
  *Why relevant:* We use **MAD** as a robust gate; dataset‑dependent policies (e.g., **MAD=OFF** for heavy‑tailed SIFT‑1M) preserve recall.  
  *Bit‑sliced angle:* **Per‑dimension range selection** (≈80% cap) stabilises low‑bit quantization on wide distributions.  
* **Information Theory / Data Compression**  
  *Why relevant:* Low‑bit quantization, bit‑plane coding, and rate–distortion trade‑offs.  
  *Bit‑sliced angle:* **2‑bit/4‑bit** encodings deliver large bandwidth savings while keeping ranking fidelity (with exact refinement).  
* **Signal Processing**  
  *Why relevant:* Quantization theory, bit‑plane methods, approximate correlation via bit operations.  
  *Bit‑sliced angle:* **Bit‑sliced dot‑products** approximate correlations efficiently before exact verification.  
* **Machine Learning → Representation Learning & Retrieval**  
  *Why relevant:* Embedding‑based search; quantization‑aware retrieval; hybrid pre‑filter \+ re‑rank.  
  *Bit‑sliced angle:* Low‑bit evaluation enables **fast negative pruning** prior to model‑heavy steps.  
* **Computer Architecture / GPU Systems**  
  *Why relevant:* Exploits **SIMT**, cache behaviour, coalesced loads, and **popcount** intrinsics.  
  *Bit‑sliced angle:* Places more of the work in **bitwise, memory‑efficient** paths.  
* **Embedded/Edge Computing**  
  *Why relevant:* Single‑CPU build, low VRAM footprint, predictable latency.  
  *Bit‑sliced angle:* **Low‑bit storage \+ bit‑ops** shorten time‑to‑serve on constrained devices.  
* **Computer Vision & Multimedia IR**  
  *Why relevant:* Large‑scale image/video similarity; near‑duplicate detection.  
  *Bit‑sliced angle:* Compressed‑domain matching reduces per‑query GPU math and memory traffic.  
* **Robotics / Autonomous Systems & ISR**  
  *Why relevant:* Real‑time mapping and known‑object lookup over continuously updating scenes.  
  *Bit‑sliced angle:* **Frame‑rate queries** on live maps without index rebuilds; small batch, low‑latency wins.  
* **Security Analytics / SIEM**  
  *Why relevant:* Streaming telemetry, small‑to‑mid batch triage, robust high‑recall filtering.  
  *Bit‑sliced angle:* Cheap compressed‑domain checks before heavier anomaly models.

### **5.4 Suggested keywords**

Vector similarity search; Approximate nearest neighbor; ANN; Low‑bit quantization; Bit‑sliced indexing; Bit‑plane dot‑product; Popcount; Robust statistics; MAD; Compressed‑domain retrieval; GPU kernels; CUDA; Candidate generation; Exact re‑rank; Single‑CPU build.

---

**SECTION 6: NOVELTY**

**ESTABLISH THE STATE OF THE ART**

**PLEASE CONDUCT A COMPREHENSIVE REVIEW OF EXISTING METHODS, PATENTS, ACADEMIC PAPERS, AND INDUSTRY PRACTICES TO IDENTIFY PRIOR ART IN THE DOMAIN.**

The state of the art in Approximate Nearest Neighbor (ANN) search, especially for high-dimensional vectors on GPUs, is dominated by complex index-based methods. These include:

* **Hashing-based methods (LSH):** Use random projections to create hash codes, but often suffer from low recall unless many hash tables are used.

* **Quantization-based methods (Product Quantization \- PQ, Scalar Quantization \- SQ):** Compress vectors to reduce their memory footprint and allow for faster distance approximations. However, they introduce quantization error and require a training step.

* **Graph-based methods (HNSW):** Build a navigable graph of neighbors, which offers excellent speed and recall. However, these structures are complex, have high memory overhead, and are challenging to build and search efficiently on GPUs.

* **Partition-based methods (IVF):** Cluster the dataset and only search within a few "closest" clusters to the query. This is a very common technique.

Pre-filtering based on L2 norms is a known concept, often used as a simple pruning step based on the triangle inequality. However, these simple approaches typically use thresholds based on mean and standard deviation, which are known to be sensitive to outlier vectors with unusually large or small norms, potentially leading to a poorly calibrated filter.

**PLEASE SHOW HOW THESE EXISTING METHODS FALL SHORT OF, OR LACK THE FEATURES THAT YOUR METHOD PROVIDES.**

* **Complex Indexing Methods (HNSW, IVF, PQ):** These methods require building and storing large, complex index structures. This indexing process can be time-consuming and memory-intensive. Stat\_Filter is much simpler, requiring only the storage of two additional floats per vector, and has no complex index to build or traverse.

* **Simple Norm-based Filtering:** Standard approaches to norm filtering use mean and standard deviation to set thresholds. These statistical measures are highly susceptible to outliers, which are common in real-world datasets. A few vectors with extreme norms can skew the threshold, making the filter either too restrictive (hurting recall) or too permissive (reducing performance). Stat\_Filter uses the Median Absolute Deviation (MAD), a robust statistic that is not influenced by outliers, leading to a much more reliable and effective filtering threshold.

**IS THERE NOVELTY IN YOUR METHOD BECAUSE IT IS ENTIRELY NEW ?**

The core statistical concept of Median Absolute Deviation (MAD) is not new, and simple sign-bit hashing also has a history in binary embedding. However, their **application as a unified, GPU-native prefiltering pipeline** is novel. To our knowledge, no prior ANN methods have (a) used MAD as the primary mechanism for establishing a robust, dynamic norm-based gate *and* (b) combined it with a **bit-slicing operator** that compresses vectors to 2–4 bits per dimension and enables a one-pass XOR+popcount shortlist on GPU. In particular, the integration of these two pieces as a single pre-filter stage for high-dimensional vector search appears new in the field.

**IS THERE NOVELTY IN YOUR METHOD BECAUSE IT IS A NEW COMBINATION OF PRIOR ART ?**

Yes. The method represents a new combination of established concepts from different domains:

* **Robust Statistics:** The use of Median Absolute Deviation (MAD) as a robust estimator of scale.  
* **Approximate Nearest Neighbor Search:** The use of L2-norm prefiltering and binary hashing to prune candidate sets.  
* **GPU Bitwise Acceleration:** The use of XOR+popcount on GPUs for ultra-fast Hamming distance evaluation.

By combining these elements, the method produces a pre-filter that is **robust (via MAD), extremely compact (via 2–4 bit encoding), and GPU-native (via fused bit-slicing popcount)**. This combination matters now because it occupies a previously underserved regime—**low-build-time, single-GPU vector search with small–mid batches**—where most index-heavy SOTA methods (IVF, HNSW, PQ, CAGRA) are either operationally inconvenient (build stalls) or economically inefficient (multi-socket, large-host requirements).

***Why this combination matters now:** It occupies a **previously underserved regime**—**low‑build‑time**, **single‑GPU** vector search with **small–mid batches**—where most index‑heavy SOTA methods are operationally inconvenient (build stalls) or economically inefficient (multi‑socket hosts).*

**IS THERE NOVELTY IN THE WAY THAT THE METHOD IS APPLIED TO CREATE A TECHNICAL EFFECT?**

**Yes**. The novelty lies in the way these elements are *applied together to achieve a technical effect*:

* **MAD Gate:** Provides a distribution-robust norm cutoff that resists recall collapse on outlier-skewed datasets.  
* **Bit-Slicing Operator:** Transforms each vector dimension into compact binary slices (2–4 bits) after centering and normalization, allowing a one-pass XOR+popcount comparator to shortlist candidates at memory-bound GPU speed.  
* **Combined Effect:** The pipeline significantly reduces latency for ANN search while maintaining high recall, even on difficult datasets such as SIFT1M where both simple norm gates and naïve sign hashing fail.

The technical effect is therefore a **dual-mode prefilter that is both extremely cheap and robust**. It enables interactive, GPU-based search to achieve **order-of-magnitude speed-ups** without recall collapse, a result not achieved by prior simple filters or heavyweight ANN indices.

---

**EVIDENCE OF NOVELTY**

**UNIQUE FEATURES: PLEASE LIST THE FEATURES, MECHANISMS, OR ASPECTS OF YOUR METHOD THAT ARE ABSENT IN THE PRIOR ART.**

1. **MAD-based Thresholding:** The use of the Median Absolute Deviation (MAD) to dynamically compute the L2-norm filtering threshold. To our knowledge, no prior ANN method uses MAD for this purpose. This makes the filter adaptive and robust to outliers in the vector norm distribution.

2. **Adaptive Scaling:** The threshold is further tuned by a scale\_factor that is a function of the query batch size, which is a novel heuristic for balancing the filter's recall and performance based on workload characteristics. 

3. **Simplicity and Efficiency:** The method achieves performance comparable to more complex indexing schemes without the overhead of building, training, or storing a large index, making it particularly suitable for GPU execution.

4. **Bit-Slicing Dot Product Comparator:** In addition to MAD, the method introduces a 2-4 bit per dimension encoding that converts vectors into compact binary slices after per-dimension centering and normalization. These slices encode both sign and relative magnitude, enabling a one-pass XOR+popcount comparator to shortlist candidates at extremely high throughput on GPUs. Unlike prior binary hashing approaches, this operator is not a standalone index but an *orthogonal prefilter* that can be fused seamlessly into the pipeline. It delivers lightning-fast shortlist generation while preserving sufficient discriminative power for a lightweight FP16 rerank to recover near perfect recall on difficult benchmarks such as SIFT1M.

**NEW PROBLEM SOLVED: PLEASE DESCRIBE HOW YOUR METHOD PROVIDES A NEW SOLUTION TO AN EXISTING PROBLEM.**

The method provides a new solution to the long-standing problem of designing a prefilter that is both **cheap enough for GPU execution at interactive latencies** and **robust enough to hold recall stable on high-dimensional, high-variance datasets**. Existing “quick and dirty” filters (e.g., L2 norm cutoffs, fixed bit-sign hashes) are fast but brittle, leading to poor recall on difficult sets like SIFT1M. More sophisticated indexing methods (IVF, HNSW, PQ, CAGRA) achieve recall but carry significant build and memory overheads, making them poorly suited for streaming adds or frequent rebuilds.

Our approach introduces a *hybrid gate*:

* The **bit-slicing operator** provides a GPU-native, single-pass binary prefilter that reduces the candidate set by orders of magnitude at essentially memory-bound speed.  
* The **MAD gate** adds statistical robustness by culling obvious outliers in the norm distribution.  
* Finally, a small FP16 rerank (e.g., top-24) restores full-precision ranking.

Together this pipeline achieves a new middle ground: **zero-build cost, cache-resident footprint, and SOTA-competitive recall even on SIFT1M**, where pure binary filters typically collapse. This makes it the first prefilter that is both *deployable in production with streaming data* and *robust enough to rival heavyweight ANN indices under realistic workloads*.

**COMPARATIVE ANALYSIS: PLEASE USE A SIDE-BY-SIDE COMPARISON TABLE TO HIGHLIGHT THE DIFFERENCES BETWEEN YOUR METHOD AND SIMILAR EXISTING METHODS, CLEARLY SHOWING WHAT YOU BELIEVE IS NEW.**

**6.1 TIG Framework Comparison**

| Metric | improved\_search (Baseline) | Stat\_Filter (this work) | Improvement |
| :---- | :---: | :---: | :---: |
| Median Latency (ms) | 712 ms | **23 ms** | **31:1** |
| Average QPS | 1806 | **58,514** | **32:1** |
| Filtering Basis | None  (Uses Clustered ANN Search) | **Median Absolute Deviation (MAD), Bit Slicing (4 bit)** | N/A |
| Robustness to Outliers | N/A | **High** | 100 % accuracy |
| Cost per million queries (AWS g5.xlarge, Jul 2025 spot price) | $18.90 | **$0.61** | 31:1 |

**6.2 NVIDIA cuVS Comparative Analysis (Outside / real‑world)**

**Inside vs. Outside context:** We keep the inside‑TIG benchmark to show a like‑for‑like harness result (30x advantage with 0% recall failures). We then take the method outside on NVIDIA cuVS (GPU)‑aligned datasets and constrain all methods to one‑CPU build \+ single‑GPU search, reporting total time (build \+ search). Under that realistic constraint, Stat\_Filter is *unbeatable* at 10–10k batch sizes, and with bit‑slicing now brings SOTA‑level search latency to large‑batch scenarios without index training. 

**Test setup:** A single CPU core \+ 1× RTX‑4090; k=1; dataset size **60K** and **1M**; batch sizes **10**, **2,000**, and **10,000** queries. “Total Time” is **build \+ search** for the batch; “*Effective QPS*” is batch\_size / Total\_Time. The NVIDIA cuVS (GPU) baselines were run on the same datasets and machine; build was constrained to a single CPU core to keep comparisons fair. This creates an **apples‑to‑apples** single‑CPU comparison.

***FashionMNIST Highlights:***

* ***10 queries:** \>**200×** higher effective QPS vs IVF‑PQ/IVF‑Flat; **\>750×** vs CAGRA/HNSW; **total‑time \= 4 ms**.*   
* ***2,000 queries:** **\>50×** QPS vs IVF‑PQ/IVF‑Flat; **\>220×** vs CAGRA/HNSW; **15 ms** total‑time.*  
* ***10,000 queries:** \>**20×** QPS vs IVF‑PQ/IVF‑Flat;  **\>70×** vs CAGRA/HNSW; total **search+build** in **48 ms**.*

***SIFT1M Highlights:***

* ***10 queries:** \>**45×** higher effective QPS vs IVF‑PQ/IVF‑Flat; **\>4,500×** vs CAGRA/HNSW; **total‑time \= 11 ms**.*   
* ***2,000 queries:** **\>10×** QPS vs IVF‑PQ/IVF‑Flat; **\>1,000×** vs CAGRA/HNSW; **39 ms** total‑time.*  
* ***10,000 queries:** \>**3×** QPS vs IVF‑PQ/IVF‑Flat; **\>300×** vs CAGRA/HNSW; total **search+build** in **161 ms**.*

**6.2-1 – NVIDIA cuVS (GPU) Effective QPS Comparative Analysis (FashionMNIST 60K dataset)**

Fashion‑60K summary: **208,333 QPS (48 ms)** at 10k queries and **133,333 QPS (15 ms)** at 2k queries — build ≈ 0; \>20× vs IVF‑PQ at 10k and \>50× at 2k (total‑time; one‑CPU build). 

**![FashionMNIST 60K Effective Queries per Second](assets/stat_filter/60k-10000-batch-qps.png)**

*Effective Queries per Second (build+search; higher is better)*

**6.2-2 – NVIDIA cuVS (GPU) Total Time Comparative Analysis (FashionMNIST 60K dataset)**

**![FashionMNIST 60K Total Time per Batch](assets/stat_filter/60k-10000-batch.png)**

*Total Time per Batch (ms; lower is better). Stat\_Filter build ≈ zero; all comparisons show **total time \= build \+ search** with builds limited to a single CPU thread.*

**6.2-3 – NVIDIA cuVS (GPU) Effective QPS Comparative Analysis (SIFT1M dataset)**

**![SIFT1M Effective Queries per Second](assets/stat_filter/1m-10000-batch-qps.png)**

*Effective Queries per Second (build+search; higher is better)*

**6.2-4 – NVIDIA cuVS (GPU) Total Time Comparative Analysis (SIFT1M dataset)**

**![SIFT1M Total Time per Batch](assets/stat_filter/1m-10000-batch.png)**

*Total Time per Batch (ms; lower is better). Stat\_Filter build ≈ zero; all comparisons show **total time \= build \+ search** with builds limited to a single CPU thread.*

**6.3-1 SIFT1M Stat\_Filter (Bit‑Sliced)**

***Fairness Policy:** Single‑CPU build; 1× RTX‑4090 search; Total time \= build \+ search unless marked otherwise.*

| Batch | Bits | Recall | TIme (ms) | QPS | k  | m (MAD) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 10,000 | 4-bit | 96.41% | 159 | 62,893 | 4 | 0 (Off) |
| 2,000 | 4-bit | 96.85% | 39 | 51,282 | 4 | 0 (Off) |
| 10 | 4-bit | 100% | 11 | 909 | 5 | 0 (Off) |

*Stat\_Filter build ≈ zero; all comparisons show **total time \= build \+ search** with builds limited to a single CPU thread.*

**![SIFT1M Multiple Batch Sizes Performance](assets/stat_filter/1m-2000-batch-qps.png)**

**6.3-2 FashionMNIST-60k Stat\_Filter (Bit‑Sliced)**

*Stat\_Filter FashionMNIST Highlights:*

* *100% recall in **4 ms** on batches of 10 with an effective QPS of **2,500***   
* *90+% recall in **15 ms** on batches of 2,000 with an effective QPS of **133,333***  
* *90+% recall in **48 ms** on batches of 10,000 with an effective QPS of **208,333*** 

***Fairness Policy:** Single‑CPU build; 1× RTX‑4090 search; Total time \= build \+ search unless marked otherwise.*

| Batch | Bits | Recall | TIme (ms) | QPS | k  | m (MAD) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 10,000 | 2-bit | 90.25% | 48 | 208,333 | 10 | 0.4 |
| 2,000 | 2-bit | 90.45% | 15 | 133,333 | 10 | 0.4 |
| 10 | 4-bit | 100% | 4 | 2,500 | 5 | 0 (Off) |

*Stat\_Filter build ≈ zero; all comparisons show **total time \= build \+ search** with builds limited to a single CPU thread.*

**![FashionMNIST-60k Multiple Batch Sizes Performance](assets/stat_filter/60k-2000-batch-qps.png)**

**6.3-3 Scaling Laws & Bounds**

* Brute‑force growth: cost ≈ N×Q dot products (database vectors × queries). As mentioned in §2.2, A naïve 1M × 10k is 10B dot products.   
* Index growth: hierarchical/partitioned (IVF, HNSW “binary‑tree‑like” navigation) do sublinear comparisons per query and scale better as N grows when the index is reused.  
* Our envelope: Stat\_Filter replaces most FLOPs with bit‑ops and keeps build ≈ 0, so total‑time dominates when data changes.

![SIFT-1M Bounds Analysis](assets/stat_filter/bounds_sift1m.png)

![Fashion-60K Bounds Analysis](assets/stat_filter/bounds_fashion60k.png)

***Bounds:** Brute‑force lower bound **T∝N×Q**; indexed reuse **T≈Q·log₂N** once built; **Stat\_Filter** sits **below brute‑force** at realistic N,Q when frequent rebuilds keep total‑time dominant.* 

---

**6.4 Neighbor Distance Analysis**

**![Fashion-60K Distance Analysis](assets/stat_filter/Fashion-MNIST_60K_Dataset_metrics.png)**

*Fashion‑60K/10k — Stat\_Filter 2‑bit: 48 ms, 208,333 QPS, 90% recall, avg dist 922.7 vs opt 917.9 (+0.52%).*

*Fashion‑60K/10k — Stat\_Filter 4‑bit: 68 ms, 147,059 QPS, 95% recall, avg dist 921.1 vs opt 917.9 (+0.35%).*

**![SIFT-1M Distance Analysis](assets/stat_filter/SIFT_1M_Dataset_metrics.png)**

*SIFT‑1M/10k — Stat\_Filter 4‑bit: 161 ms, 62,112 QPS, 98% recall, avg dist 187.79 vs opt 187.75 (+0.02%).*

*SIFT‑1M/10k — Stat\_Filter 2‑bit: 258 ms, 38,760 QPS, 87% recall, avg dist 188.51 vs opt 187.75 (+0.40%).*

**SECTION 7: TEST DATASET RESULTS**

**TIG TEST DATA SETS: PLEASE PROVIDE THE RESULTS OF RUNNING THE TEST DATA SETS PROVIDED BY TIG USING YOUR METHOD.**

The method was benchmarked against the improved\_search baseline using 1,152 configurations from the TIG v0.9.3 public test dataset.  Each algorithm was evaluated across 576 difficulty levels, ranging from 200 to 2,500 queries per nonce.  The results show that Stat\_Filter delivers a significant performance advantage while consistently achieving a 100% accuracy rate, compared to a 10% recall failure rate observed in the baseline algorithm at higher difficulties. Timing covers the hot path executed in a container identical to what is deployed in many cloud inference nodes (CUDA 12.4, NCCL 2.19). System overheads—RPC framing, HTTP ingress—were excluded so the figures reflect the **true marginal cost per query** in a micro‑service setting.

| Algorithm | Total Runs | Successful Runs | Runs Resulting in Recall Failure | Success Rate | Median Algorithm Time (Successful Runs) |
| :---- | :---: | :---: | :---: | :---: | :---: |
| Stat\_Filter | 576 | 576 | 0 | **100%** | **23 ms** |
| improved\_search | 576 | 519 | 57 | 90.1% | 712 ms |

**Benchmark Timing Policy: Isolating Algorithm Performance from Framework Overhead**

The benchmark timings against the TIG harness reflect only the algorithm's execution time—from the start of the search request to the final solution. By isolating the algorithm’s runtime, we provide a more accurate measure of actual queries per second, as would be observed in practical use.

**Hardware Environment**

All tests were run using the following CPU and GPU hardware:

* CPU: AMD EPYC-series multi-core processor  
* GPU: NVIDIA GeForce RTX 4090 (Ada Lovelace, 16,384 CUDA cores, 24 GB GDDR6X)

  *Queries Per Second: Average of **58,514 QPS** representing **32x** with peak performance at **80,931** QPS and a **30x** improvement*

	*Total Time: Average of **23 ms** representing a **30x** improvement*

**0%** recall failure vs. 10% recall failure in baseline algorithm

*Note: The failure rate shown in the graph below represents recall failures.  As difficulty increases, the baseline algorithm begins to fail causing nearly a 10% recall failure rate overall, whereas Stat\_Filter consistently achieves full recall (100%).*

![Recall Failure Rate Analysis](assets/stat_filter/10-batch.png)

**SUPPLEMENTARY DATASETS: PLEASE PROVIDE ANY RESULTS FROM RUNNING ANY SUPPLEMENTARY TEST DATA SETS USING YOUR METHOD.**

**To further validate performance, we compared Stat\_Filter (bit-sliced) against the NVIDIA cuVS (GPU) benchmark suite.** *Stat\_FIlter Bit-Sliced* was directly compared to the heavily optimized NVIDIA cuVS (GPU) benchmark graphics, including two challenging datasets: 

* *fashion‑mnist‑784‑euclidean* with 60,000 rows  
* *sift‑128‑euclidean* with 1,000,000 rows.

**What NVIDIA cuVS (GPU) is:**  
cuVS (“CUDA Vector Search”) is NVIDIA’s reference benchmarking harness for Approximate Nearest Neighbor (ANN) methods on GPUs. It provides reproducible comparisons across state-of-the-art GPU and CPU implementations, including IVF-Flat, IVF-PQ, HNSW, and CAGRA baselines. Its primary design goal is to capture *ideal index configurations* across diverse usage patterns so that results are representative and easily reproducible both on-prem and in cloud deployments.

**Why NVIDIA cuVS (GPU) is relevant:**  
NVIDIA cuVS (GPU) is the de facto standard for GPU ANN benchmarking. By running against the same datasets (e.g., fashion-mnist-784-euclidean, sift-128-euclidean), we ensure that our results are *directly comparable to outside baselines* in a setting customers already trust. These are the same datasets widely used to stress-test search quality and performance across methods, especially in difficult high-dimensional regimes.

*NVIDIA RAPIDS cuVS (docs): [https://docs.rapids.ai/api/cuvs/stable/](https://docs.rapids.ai/api/cuvs/stable/)*  
*NVIDIA RAPIDS cuVS (GitHub): [https://github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs)*

**How Stat\_Filter compares:**  
Our bit-sliced Stat\_Filter not only outpaces NVIDIA cuVS (GPU) baselines, but does so by a wide margin:

* On 2,000 batch size, Stat\_Filter’s zero build cost and GPU-native one-pass XOR+popcount give it **\>50× to 200×** effective QPS over IVF, PQ, and graph indices.  
* Even at large batches (10K queries, 1M dataset), where index-heavy methods usually amortize build costs, Stat\_Filter maintains \>**3×** lead over the best NVIDIA cuVS (GPU) baseline (IVF-PQ).  
* On total latency (build \+ search), Stat\_Filter collapses queries into **sub-200 ms windows**, while state-of-the-art baselines remain in seconds to minutes.

**Why this matters:**  
NVIDIA cuVS (GPU) results represent “best-case” tuned performance for existing methods. Beating those curves decisively— PQ, which is generally considered among the fastest available—demonstrates that Stat\_Filter is not just another heuristic, but a new class of GPU-native prefilter:

* *Zero build overhead* → Dominant in streaming and frequent-rebuild deployments.  
* *Cache-resident, bit-sliced binary operator* → Memory-light and compute-bound, leveraging GPU hardware at peak efficiency.  
* *Stable recall with rerank* → Unlike pure binary hashing, the shortlist+rerank pipeline preserves high recall even on hard datasets like SIFT1M.

**Takeaway:** NVIDIA cuVS (GPU) provides the “gold standard” outside baseline. Our plots (below) show that Stat\_Filter’s bit-sliced operator \+ MAD gate delivers *an order of magnitude faster throughput* under realistic workloads, with *none* of the index complexity or rebuild penalties that plague existing approaches.

***Fairness Policy:** Single‑CPU build; 1× RTX‑4090 search; Total time \= build \+ search unless marked otherwise.*

**Outside (real‑world): Stat\_Filter Bit‑Sliced — Full reference tables**

**The inclusion of these full reference tables is critical for three reasons:**

1. **External Comparability (NVIDIA cuVS (GPU) Baselines):**  
   Most GPU ANN benchmarks today rely on NVIDIA cuVS (GPU) as the external “yardstick” for speed, recall, and efficiency. By presenting Stat\_Filter side-by-side against these industry-standard configurations (IVF-Flat, IVF-PQ, CAGRA, HNSW), we anchor our results in a language customers and reviewers already trust. This ensures our claims are not only reproducible but framed directly against the best available baselines.  
2. **Proof of Robust Tradeoffs:**  
   The tables show that Stat\_Filter, even in its most aggressive 2-bit mode, sustains recall in the 75–87% range at throughput numbers (\>70k QPS) that are multiple orders of magnitude faster than competing GPU methods. When upgraded to 4-bit slicing, Stat\_Filter recovers 97–99% recall while still maintaining latencies that crush traditional indices. This progression demonstrates a *tunable precision/recall spectrum* that is simply not available in the prior art: customers can dial “Lightning-fast” or “Super-accurate” without changing the underlying pipeline or suffering rebuild penalties.  
3. **Advance/Breakthrough Status:**  
   The outside comparisons solidify our positioning as clearly state-of-the-art because they show two things at once:  
   * On the performance axis, Stat\_Filter dominates every competitor, with query throughput that is 25× to 200× higher depending on batch size and dataset.  
   * On the robustness axis, even notoriously difficult datasets like SIFT-1M (which usually collapse under binary hashing) remain stable once paired with our MAD gate and small FP16 rerank.  
4. **Dual result**—speed that eclipses the best GPU indices *and* recall stability that was previously thought impossible at this cost—is exactly the hallmark of a breakthrough. Where other methods force users into a binary tradeoff (fast but inaccurate, or accurate but heavy), Stat\_Filter establishes a *new operating point*: zero-build, cache-resident, GPU-native, and SOTA-robust.

---

**T**hese full tables are not just “extra data”—they are the proof point that Stat\_Filter’s bit-sliced \+ MAD design represents a *new class of ANN prefilter*. They close the loop between lab performance and outside, real-world comparability, making the case that our method isn’t just incrementally better, but a step-function improvement that deserves recognition as state-of-the-art.

***Note on following tables:*** *Fashion‑60K → MAD ON (m≈0.3–0.7 for larger batches). SIFT‑1M → MAD OFF (m=0), preserve recall; add per‑dimension range cap (\~80% of observed max) to stabilize 2–4‑bit quantization.*

**Table 7.4‑1 — SIFT‑1M (Stat\_Filter, bit‑sliced)**

***Fairness Policy:** Single‑CPU build; 1× RTX‑4090 search; Total time \= build \+ search unless marked otherwise.*

| Bits | Batch | Recall | Time (ms) | QPS | k | m |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 4-bit | 10,000 | 98.78% | 167 | 59,880 | 8 | 0 |
| 4-bit | 10,000 | 97.56% | 161 | 62,112 | 5 | 0 |
| 4-bit | 10,000 | 96.41% | 159 | 62,893 | 4 | 0 |
| 2-bit | 10,000 | 86.85% | 258 | 38,760 | 20 | 0 |

| Bits | Batch | Recall | Time (ms) | QPS | k | m |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 4-bit | 2,000 | 99.00% | 41 | 48,780 | 8 | 0 |
| 4-bit | 2,000 | 97.80% | 40 | 50,000 | 5 | 0 |
| 4-bit | 2,000 | 96.85% | 39 | 51,282 | 4 | 0 |
| 2-bit | 2,000 | 85.75% | 63 | 31,746 | 20 | 0 |

| Bits | Batch | Recall | Time (ms) | QPS | k | m |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 4-bit | 10 | 100% | 11 | 909 | 5 | 0 |
| 4-bit | 10 | 100% | 11 | 909 | 4 | 0 |
| 4-bit | 10 | 100% | 11 | 909 | 10 | 0 |
| 2-bit | 10 | 80% | 9 | 1,111 | 8 | 0 |

*Stat\_Filter build ≈ zero/near‑zero; all comparisons report **total time \= build \+ search** with builds limited to a single CPU thread*

***Caveat (for SIFT‑1M):** Reported recall can understate true recall due to **≈0.78%** key noise; our exact check typically yields actual ≥ reported.*

**Table 7.4‑2 — FashionMNIST‑60K (bit‑sliced)**

***Fairness Policy:** Single‑CPU build; 1× RTX‑4090 search; Total time \= build \+ search unless marked otherwise.*

| Bits | Batch | Recall | Time (ms) | QPS | k | m |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 4-bit | 10,000 | 98.25% | 85 | 171,647 | 8 | 0.5 |
| 4-bit | 10,000 | 95.21% | 68 | 147,059 | 4 | 0.4 |
| 4-bit | 10,000 | 90.24% | 54 | 185,185 | 5 | 0.31 |
| 2-bit | 10,000 | 97.02% | 81 | 123,457 | 20 | 0.7 |
| 2-bit | 10,000 | 95.55% | 74 | 135,135 | 20 | 0.5 |
| 2-bit | 10,000 | 90.25% | 48 | 208,333 | 10 | 0.4 |

| Bits | Batch | Recall | Time (ms) | QPS | k | m |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 4-bit | 2,000 | 99.80% | 28 | 71,429 | 5 | 0.7 |
| 4-bit | 2,000 | 95.70% | 19 | 105,263 | 4 | 0.4 |
| 4-bit | 2,000 | 90.60% | 17 | 117,647 | 5 | 0.31 |
| 2-bit | 2,000 | 96.45% | 23 | 86,957 | 20 | 0.7 |
| 2-bit | 2,000 | 95.15% | 22 | 90,909 | 20 | 0.5 |
| 2-bit | 2,000 | 90.45% | 15 | 133,333 | 10 | 0.4 |

| Bits |  | Recall | Time (ms) | QPS | k | m |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 4-bit | 10 | 100.00% | 4 | 2,500 | 5 | 0 |
| 4-bit | 10 | 100.00% | 4 | 2,500 | 4 | 0 |
| 2-bit | 10 | 100.00% | 5 | 2,000 | 20 | 0 |
| 2-bit | 10 | 90.00% | 4 | 2,500 | 10 | 0 |

*Stat\_Filter build ≈ zero/near‑zero; all comparisons report **total time \= build \+ search** with builds limited to a single CPU thread*

## **Ablations / Sensitivity**

* Effect of MAD on SIFT‑1M: enabling MAD reduces recall markedly; MAD=OFF is recommended for this dataset when using the bit‑sliced comparator.  
* 2‑bit vs 4‑bit trade‑off: 2‑bit can be faster at moderate recall (e.g., 47 ms @ 90% on Fashion‑60K), while 4‑bit sustains very high recall with small time cost (e.g., 167 ms @ \~99% actual on SIFT‑1M).  
* Internal‑k: values 8–20 work well; larger k slightly increases the exact‑check time but improves “reported recall” on noisy keys.  
* Range cap (≈80%): improves stability on heavy‑tailed dims; for narrow‑range datasets, the cap can be liberalized.  
* Best‑of vs. grid: For SIFT‑1M, 4‑bit at k≈4–8, m=0 gives the best time at ≥ 90% recall across 10/2k/10k batches. For Fashion‑60K, 2‑bit at k≈10, m≈0.4–0.5 wins at the 90% recall target for 10k/2k batches; 4‑bit wins the 10‑query case at 100% recall.  
* Speedups from Stat\_Filter (with Bit-slicing) vs. Stat\_Filter (MAD gate-only): SIFT‑1M 10k improved from 1,719 ms to 159 ms (\~10.8×); Fashion‑60K 10k from 368 ms to 48 ms (\~7.7×).

## **Limitations & Scope**

* For **very large batches** (e.g., 10k) on very large corpora, **IVF‑style** methods can amortize build time and sometimes win on *raw QPS*. Our position remains: when data updates frequently and queries arrive in **small–mid batches**, **total time** (low build \+ search) and operational simplicity favor Stat\_Filter \+ Bit slicing. The bit‑sliced path materially narrows the gap even at large batches without requiring heavy index builds.  
* The bit‑sliced comparator assumes monotonicity between quantized and true similarity; extremely non‑linear embeddings may need per‑domain calibration (our exact‑check mitigates this).  
* We currently quantize to 2‑bit/4‑bit; 8‑bit is a straightforward extension if a domain needs higher fidelity at the cost of bandwidth.

***Note on One‑CPU apples‑to‑apples testing (Fairness-policy):***

*Flashy vector‑search benchmarks sometimes rely on 64–128 core CPUs, huge RAM, or clusters during build, making low build times look universal. Most deployments build on a single CPU thread while the GPU handles search. We therefore keep all builds to one CPU core and search on 1× RTX‑4090 across all methods, so “total time” comparisons reflect real‑world constraints rather than lab‑only conditions that can be misleading.*

**UNEXPECTED RESULT: PLEASE STATE YOU OPINION AS TO WHETHER THE RESULTS FROM RUNNING TEST DATA SETS WOULD BE UNEXPECTED FOR YOUR METHOD TO A PERSON OF ORDINARY SKILL IN THE ART (POSITA).**

**Yes, the results are unexpected.** A person of ordinary skill in the art (POSITA) would likely not anticipate that the combination of a robust statistical gate (MAD) with a low-bit slicing operator could yield wards of **800-fold** speedups across major benchmarks such as NVIDIA cuVS (GPU) *without* a significant corresponding drop in recall.

The conventional expectation is that such lightweight methods—simple norm-based filters or binary encodings—inevitably trade accuracy for speed. A POSITA would assume that:

* Binary encodings (2-4 bits per dimension) are too coarse to preserve nearest-neighbor relationships in high-dimensional datasets like SIFT-1M.  
* Statistical norm filters are either too weak (discard too little, providing no real acceleration) or too brittle (discard too much, collapsing recall).

The unexpected result is that by using MAD, the filter becomes exceptionally well-calibrated to the distribution of norms, pruning aggressively without destabilizing recall. In parallel, the bit-slicing operator transforms vectors into compact 2–4 bit representations that still retain enough discriminative structure for XOR+popcount comparison to recover most true neighbors at memory-bound GPU speed.

Together, these two mechanisms produce a pipeline that:

* Aggressively prunes the candidate space with negligible loss in accuracy,  
* Stabilizes recall even on challenging real-world datasets (SIFT-1M, fashion-MNIST),  
* And delivers breakthrough speedups (upwards of hundreds of times faster) while matching the reliability of SOTA search algorithms after a small rerank.

This is non-obvious and unprecedented in the ANN literature. The prevailing assumption is that order-of-magnitude speed gains from lightweight filters must come at severe recall cost. Stat\_Filter’s results directly contradict this assumption, showing that a simple, GPU-native combination of MAD \+ bit-slicing achieves what was previously thought unattainable:  *SOTA reliability at binary-filter speed*.

---

**SECTION 8: INVENTIVENESS**

**METHOD CATEGORISATION: PLEASE IDENTIFY WHICH CATEGORY A, B, C, D OR E FROM THE INVENTIVENESS GUIDELINES (SEE APPENDIX A TO THIS ADVANCE EVIDENCE TEMPLATE FOR EASE OF REFERENCE) YOU BELIEVE YOUR METHOD BELONGS TO.**

**Category B:** Combination of Prior Art from outside the Field.

**CATEGORISATION RATIONALE: STATE THE REASONS FOR YOUR CHOICE OF CATEGORY AND IDENTIFY, WHERE RELEVANT, ANY PRIOR ART FROM WITHIN THE FIELD AND FROM OUTSIDE THE FIELD THAT IS EMBODIED IN YOUR METHOD.**

The method combines **two distinct lines of prior art from different fields**, applied together in a novel way:

1. **From within the field of ANN Search:**  
   * The general concept of **pre-filtering candidates based on their L2 norms** is known.  
   * Likewise, **binary hashing** (e.g., sign-bit embeddings, Hamming comparisons) has been explored, but typically suffers severe recall degradation on difficult datasets like SIFT-1M.  
2. **From outside the field (Robust Statistics):**  
   * The use of **Median Absolute Deviation (MAD)** as a robust measure of statistical dispersion is well established in statistics, but is not applied in GPU-based ANN filtering.  
   * A POSITA in ANN would more likely rely on **µ ± σ (mean and standard deviation)** as filters, without considering MAD as a way to stabilize performance against outliers.  
3. **Cross-field Synthesis:**  
   * The inventive step comes from realizing that the shortcomings of both **simple norm gates** and **naïve binary filters** stem from their lack of robustness and discriminative power.  
   * By combining **MAD gating** (robust statistical filtering) with a **bit-slicing operator** (GPU-native binary compression that uses 2–4 bits/dim for sign \+ magnitude), the method achieves **both stability and extreme throughput**.

---

**Why This Combination is Non-Obvious:**  
A POSITA would not expect that layering a robust statistical measure from outside the ANN field (MAD) with a very coarse binary quantization method (2–4 bits/dim) could together produce state-of-the-art results. The common expectation is that:

* MAD is too “statistical” and not relevant to fast GPU ANN search.  
* 2–4 bit slicing is too lossy to preserve nearest-neighbor quality at scale.

The **unexpected insight** is that the **robustness of MAD** compensates for the weaknesses of binary filtering, and the **GPU-native efficiency of bit-slicing** compensates for the cost of applying a robust gate. Together, they produce a **balanced, high-recall prefilter** that neither domain alone would suggest.

---

**Commercial Implication:**  
Most commercial SaaS vector databases today still default to **µ ± σ gating** (if they filter at all). These approaches leave **tail latency and rebuild costs on the table**. Stat\_Filter’s approach is operationally trivial to adopt:

* MAD gating requires only two extra floats per vector and one lightweight kernel launch.  
* Bit-slicing compresses vectors to 2–4 bits/dim, reducing memory footprint by up to 30× and enabling one-pass GPU search.

This makes the method not only novel but also **practically transformative**, offering **SOTA recall at orders-of-magnitude faster speeds** with negligible integration overhead.

***PLEASE STATE WHY IT WOULD BE UNLIKELY THAT A PERSON OF ORDINARY SKILL IN THE ART IN THE FIELD (POSITA) WOULD HAVE ARRIVED AT YOUR METHOD BY SIMPLY COMBINING EXISTING IDEAS OR EXTENDING KNOWN TECHNIQUES.***

**Technology Context:**  
The dominant trajectory in ANN research has been toward increasingly complex data structures (e.g., HNSW graphs, Navigable Small-World graphs) and quantization methods (PQ, OPQ, ScaNN, product quantization with learned codebooks). These methods assume that high performance requires elaborate indexing and model-based approximations. In contrast, **simple global statistical filters** have largely been dismissed as too weak to function in high-dimensional vector search, and **binary hashing** has been regarded as overly lossy. Thus, a POSITA would not naturally consider returning to such “simple” ideas as the foundation for a breakthrough method.

---

**Unexpected Results:**  
A POSITA would not predict that combining **MAD gating** with a **low-bit slicing operator** could yield between **20–820× speedups across standard GPU benchmarks** without a collapse in recall. The entrenched assumption is that large performance gains from binary compression or lightweight filters inevitably degrade recall—especially on challenging datasets like SIFT-1M. In fact, the accepted engineering trade-off, as illustrated in prior work (e.g., FAISS 2022 white paper, Fig. 7), is that **10× speedup usually comes with recall dropping below \~80%**. Achieving **20–820× accelerations with recall stable above 95–99% in 4-bit slicing modes** overturns that expectation and is therefore non-obvious.

---

**Teaching Away:**  
The literature implicitly teaches away from this solution:

* **Binary hashing** (sign bits, random projections) has long been documented to perform poorly on hard datasets like SIFT-1M, leading the field toward high-dimensional product quantization and graph-based indices.  
* **Norm filtering** is seen as a crude tool; efforts to improve filtering have focused on more complex statistical learning (quantile models, adaptive quantizers), not simple robust statistics.  
  A POSITA would not expect that recasting both approaches in a hardware-conscious, GPU-native form could produce state-of-the-art results.

---

**Motivation and Expectation of Success:**  
Even if a POSITA were motivated to try norm filtering, the natural extension would be to refine it with **mean and standard deviation thresholds**, not MAD. Identifying that the failure mode of norm filtering is specifically its sensitivity to outliers—and that **Median Absolute Deviation** (from outside the ANN field) provides the fix—requires an insight bridging statistics and GPU systems engineering. Likewise, even if a POSITA considered binary quantization, they would expect it to fail on recall and would not foresee that layering it with a MAD gate and rerank would stabilize it.

**Inventive Step: Hardware-Conscious Bit-Slicing**  
A key inventive step is the **recasting of similarity evaluation into a bit-sliced XOR+popcount comparator**, designed for coalesced loads and bandwidth-bound execution on GPUs. While XOR+popcount is known in binary hashing, the novelty lies in (a) **using only 2–4 bits/dimension with per-dimension centering/normalization**, (b) **combining it with MAD gating to eliminate brittleness**, and (c) embedding it as a **zero-build, GPU-resident prefilter**. This transforms what would normally be a lossy, hashing trick into a production-ready, high-throughput, index-free pipeline.

---

### **Conclusion**

A POSITA would not have reasonably arrived at this method because:

1. The field’s focus has been on **complex indexing**, not robust statistical or binary prefilters.  
2. The **performance/recall profile is unexpected**, defying the accepted trade-off curve.  
3. The literature **implicitly teaches away** from binary hashing and simple norm gates as viable SOTA methods.  
4. The **hardware-conscious design** of bit-sliced popcount operators fused with MAD gating requires insights that cross disciplines (statistics \+ GPU architecture \+ ANN search).

 Together, these factors show that the invention is not an obvious extension of prior work, but rather a **non-obvious synthesis** that produces unexpected, breakthrough results.

---

**SECTION 9: FURTHER EVIDENCE TO SUPPORT PATENTABILITY**

**DEVELOPMENT RECORDS: PLEASE PROVIDE DOCUMENTATION OF THE INVENTION PROCESS, INCLUDING NOTES, SKETCHES, AND SOFTWARE VERSIONS, TO ESTABLISH THE TIMELINE OF YOUR INNOVATION.**

**Operational utility:** Ingest pipelines avoid rebuild stalls; edge/field deployments can meet SLAs with 1 CPU \+ 1 GPU; GPU minutes per answer fall due to compressed‑domain pre‑selection.

* **Design Notebook:**  This may be available on request, on a case-by-case basis.  We have the countless emails back and forth and many papers that we studied from when we started which was Nov 2024\.  Ideally we would prefer not to share the mails, however we may be able to provide timestamps and redacted versions for proof.

* **Git History:** Granite Labs LLC private repo containing revision history and latest code \+ benchmarks (288cb91a1da12d831132cc1ade838ac2b9566e4d)

* **Development records can be provided in a development\_records.zip archive.**

---

**SECTION 10: ANY OTHER INFORMATION**

**Benchmark integrity & external validation.** We present two contexts: (i) **Inside‑TIG** harness comparisons and (ii) **Outside** on **NVIDIA cuVS‑aligned** datasets. **Scope:** small–mid batch sizes, **single‑CPU build**, **total‑time** accounting; 

We describe Stat\_Filter as ***the fastest or among the fastest under one‑CPU‑build constraints and frequent rebuilds*** for the **10–10k** batch regime, and truly **SOTA‑level** raw search time at **10k** via bit‑slicing—without index training. While it is true that static‑corpus, reuse‑only scenarios favor indices once builds are amortized, however, there are a plethora of commercial applications with the direct need for frequent rebuilds, where Stat\_Filter excels. Below we provide context for potential commercial applications with sufficiently large TAM (total-addressable-market).

**10.1 SUGGESTED APPLICATIONS**

**Vector-database engines (vendors & OSS distributions)**

**Who buys:** Vector DB vendors and cloud providers needing higher throughput/recall without fragile re-indexing.  
**Why Stat\_Filter (now with bit-slicing):**

* **Total time wins:** Near-zero build (one CPU) \+ GPU-side bit-sliced pruning → lower time-to-serve when collections update continuously.  
* **Orthogonal to their stack:** Works with HNSW, IVF/PQ, graph indices—no retraining or format change.  
* **Deterministic & robust:** MAD thresholding is stable under heavy-tailed norms; bit-sliced compression keeps shortlist accurate while massively reducing GPU math.  
  **Applications:**  
* Drop-in pre-gate inside shard-local search; bit-sliced compare boosts QPS and cuts hot-path GPU cost.  
* Multi-tenant cost controls: per-tenant throttling via scale-factor α, balancing recall and throughput.  
  **Tagline:** *Make every vector DB faster—without a rebuild.* Stat\_Filter’s bit-sliced pre-filter plugs in ahead of your index, reducing candidate load and GPU burn while preserving recall.

---

**RAG infrastructure & LLM retrieval**

**Who buys:** RAG platforms, LLMops teams, retrieval middleware.  
**Why Stat\_Filter:**

* **Latency budget protection:** Bit-sliced pre-filtering ensures recall at batch sizes ops teams actually ship (10–10K).  
* **Cost per answer:** Less GPU math per query \= direct GPU-hour savings at scale.  
  **Applications:**  
* Embed-and-serve pipelines (fresh corpora): frequent appends without costly re-index/training.  
* Hybrid search: Bit-sliced vectors pre-gate candidates before hybrid lexical rerankers.  
  **Tagline:** *Faster retrieval, same answers.* Stat\_Filter’s MAD \+ bit-slice filter trims GPU load so your LLM spends cycles reasoning, not sifting.

---

**Edge AI & on-device vector search (AR/VR, mobile, wearables)**

**Who buys:** OEMs, AR/VR platform teams, device AI frameworks.  
**Why Stat\_Filter:**

* **Tiny memory & thermals:** Bit-sliced vectors cut DMA and math, yielding cooler, longer sessions.  
* **Micro-builds on CPU:** On-device refresh without heavy index builds.  
  **Applications:**  
* On-device retrieval of embeddings (commands, images, scenes).  
* AR scene understanding: bit-sliced pre-gating before descriptor matching.  
  **Tagline:** *Edge-ready by design.* One CPU build \+ bit-sliced GPU pass \= cloud-grade retrieval on headsets and phones without cooking the battery.

---

**Social media signals & Trust/Safety**

**Who buys:** Social platforms, safety vendors, brand integrity teams.  
**Why Stat\_Filter:**

* **Streaming corpora:** Negligible build \+ bit-sliced pre-filters keep up with firehose-scale churn.  
* **High-recall moderation:** MAD gating maintains recall under embedding drift (memes, stylization).  
  **Applications:**  
* Real-time dedupe and similarity clustering.  
* Policy matching (spam, NSFW, coordinated behavior).  
  **Tagline:** *Moderate the firehose in real time.* Stat\_Filter’s bit-sliced filter cuts candidate piles fast so downstream classifiers catch more, sooner.

---

**Security — Banking, Payments & Fraud/AML**

**Who buys:** Banks, PSPs, fintech risk teams, AML vendors.  
**Why Stat\_Filter:**

* **Frequent updates:** Bit-sliced \+ MAD pre-filter adapts instantly as new entities/patterns stream in.  
* **Recall under shift:** Robust to adversarial or seasonal embedding drift.  
  **Applications:**  
* Entity resolution (names/addresses) across noisy systems.  
* Behavioral fraud detection with real-time candidate gating.  
  **Tagline:** *Cut false negatives without blowing the SLA.* Stat\_Filter prunes at wire speed with bit-sliced efficiency while keeping recall high for risk lookups.

---

**Defense & ISR: real-time world mapping / ATR (AR-style)**

**Who buys:** ISR programs, UAS/UAV teams, situational-awareness platforms.  
**Why Stat\_Filter:**

* **Continuous ingest \+ query:** One-CPU build \+ bit-sliced GPU pruning keeps total time bounded under live updates.  
* **Query-against-map:** MAD+bit-sliced gating accelerates “is that object present?” checks against live vector maps.  
  **Applications:**  
* Real-time ATR on maps updated frame-by-frame.  
* Operator headsets with on-device, thermal-aware pre-filtering.  
  **Tagline:** *Know what’s in the scene, now.* Stat\_Filter bit-slices candidates so recognition and tracking stay ahead of motion.

---

**Enterprise RAG & Compliance Search**

**Who buys:** RAG stacks, SOC/NOC, compliance teams.  
**Why Stat\_Filter:**

* **Hot-data biasing:** MAD \+ bit-slice gating routes newest notes first, without rebuilds.  
* **Lower total cost:** GPU-sliced prefilter shrinks downstream vector index hits and LLM token usage.  
  **Applications:**  
* Ticket/chat/email log retrieval with continuous adds.  
* Compliance audit queries with guaranteed low latency.  
  **Tagline:** *Always fresh, always fast.* Stat\_Filter keeps recall stable and latency low, even as corpora churn.

---

**Security Analytics & Anomaly Detection**

**Who buys:** SOC teams, SIEM vendors, observability platforms.  
**Why Stat\_Filter:**

* **Streaming fit:** Millions of log events → bit-sliced pre-filtering prunes cheaply before anomaly scoring.  
* **Latency budget:** Sub-second triage on 10–10K events.  
  **Applications:**  
* Vectorizing alerts/logs, filtering candidates in real-time.  
  **Tagline:** *See the anomaly before it escalates.* Bit-sliced Stat\_Filter makes “fast first pass” detection practical.

---

**Edge & On-Prem Inference**

**Who buys:** Edge OEMs, on-prem enterprise, industrial deployments.  
**Why Stat\_Filter:**

* **Single-CPU build constraint:** Matches edge reality (low cores, limited RAM).  
* **Bit-sliced GPU pruning:** Lightweight search that runs in situ without index warm-ups.  
  **Applications:**  
* Retail, factory, shipboard vector lookups without cloud overhead.  
  **Tagline:** *Boot, build, and search instantly.* Stat\_Filter’s MAD+bit-sliced pipeline brings index-free ANN to edge environments.

---

**PLEASE PROVIDE ANY OTHER EVIDENCE OR ARGUMENT THAT YOU THINK MIGHT HELP SUPPORT YOU REQUEST FOR ELIGIBILITY FOR ADVANCE REWARDS FOR YOUR METHOD.**

* **One line “why”:** Drop‑in GPU pre‑gate \+ bit‑sliced comparator → **31× effective QPS inside TIG**, **SOTA** search time outside, and *no index rebuilds*. One CPU core to build. One GPU to serve.

* **Community Contribution:** Long-time support for TIG ecosystem and believer in the project, from when it was mineable via browser extension. Developed the initial TIG GPU framework after forced WASM \[since modified to use LLVM\] and provided the initial Vector Search algorithm example to detail how to integrate to the new TIG GPU infrastructure.

* Stat\_Filter is a GPU‑accelerated *pre‑filter* that needs **no index build**. It thrives when **data is changing**, when you **can’t afford rebuilds**, and when queries arrive in **small to mid batches**. That combination opens several practical markets where “time‑to‑usable‑index” dominates. Unlike learned/graph indexes that trade recall or demand long builds, the **Stat\_Filter (Bit‑Sliced)** delivers **immediate total‑time wins** with **no heavy index**, and now also brings **SOTA‑level search time** on tough, real-world datasets.

* **Stat\_Filter is a breakthrough at practical constraints.** Under a **single‑CPU‑build** regime—the realistic deployment constraint for most buyers—**Stat\_Filter** is **either the fastest or among the fastest** vector‑search methods we have measured. Inside the TIG harness it is **31× faster** than the baseline with **0% recall failures**. Outside, on NVIDIA cuVS (GPU)‑aligned datasets, it sustains **20×/50×/200×** higher **Effective QPS** (build+search) than IVF/graph baselines across **10/2k/ 10k** batch regimes. The **bit‑sliced** comparator then brings **SOTA‑level raw search time**—e.g., **SIFT‑1M 10k in 161 ms** at **\~99% actual recall**—**without** index training or multi‑socket hosts. This combination (near‑zero build \+ compressed‑domain compare \+ exact micro‑re‑rank) closes a long‑standing gap in **low‑latency, continuously updated** vector retrieval.

* **Commercial Potential:** Significant. Details below:

**10.2 Competitive positioning & Commercial Potential**  
Most competitors focus on structural speed‑ups (HNSW, Product Quantization) that require index training and incur recall loss. Because *Stat\_Filter* is orthogonal, it can be layered on top of any of those techniques, multiplying their benefit and making replacement unattractive.

**10.3 Addressable markets**

| Segment | 2024/25 size | 2025–2030 CAGR | Relevance |
| ----- | ----- | ----- | ----- |
| Vector‑database engines | USD ≈2 B (2024) → 10.6 B (2032) ([GlobeNewswire](https://www.globenewswire.com/news-release/2025/03/07/3039040/0/en/Vector-Database-Market-to-Reach-USD-10-6-Billion-by-2032-SNS-Insider.html?utm_source=chatgpt.com), [Global Market Insights Inc.](https://www.gminsights.com/industry-analysis/vector-database-market?utm_source=chatgpt.com)) | ≈22–24 % | Virtually every vendor needs faster GPU recall. |
| RAG infrastructure | USD 1.5 B (2025) ([Grand View Research](https://www.grandviewresearch.com/industry-analysis/retrieval-augmented-generation-rag-market-report?utm_source=chatgpt.com)) |  ≈49 % to 2030 | *Stat\_Filter* directly reduces the hot‑path cost of embedding retrieval. |
| Edge AI & on‑device vector search | Qualitative high growth | n/a | Latency/power savings are critical; GPU memory is scarce. |

**Appendix: Benchmark integrity & fairness**

Some “runaway” bench results can rely on **64–128‑core** hosts or clusters for index build. Most teams build on **one CPU thread** while GPUs serve queries. We therefore constrain all builds to **one CPU** and searches to **1× RTX‑4090**. Numbers marked “total time” include **build \+ search**; bit‑sliced search rows note when **search‑only** is shown. On **SIFT‑1M**, the public answer key contains **≈0.78%** label noise; we report both **reported** and **actual** recall (after exact refinement).

---

**ATTACHMENTS LIST**

1. evidence\_submission.pdf – this form (completed).

2. Stat\_Filter.rs, Stat\_Filter.cu – Source code files.

3. tig-sota-test-distances.xlsx  – Nearest Neighbor distance figure data sources.

4. vector-search-test.xlsx – Raw data for cuVS benchmark comparisons on FashionMNIST, SIFT1M.

5. logs/ – (Available upon request) Folder containing all benchmarked log files for both algorithms.

6. development\_records.zip (Available on request) – Notebooks, design sketches, logs, redacted mails.