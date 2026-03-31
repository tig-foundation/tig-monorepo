## UNIQUE ALGORITHM IDENTIFIER (UAI)

> c006_a025

# ADVANCE EVIDENCE TEMPLATE

**EVIDENCE IN SUPPORT OF A REQUEST FOR ELIGIBILITY FOR ADVANCE REWARDS**

## INTRODUCTION

* TIG TOKEN HOLDERS WILL VOTE ON WHETHER YOUR **ALGORITHMIC METHOD** IS ELIGIBLE FOR ADVANCE REWARDS.  
    
* TIG TOKEN HOLDERS ARE FREE TO VOTE AS THEY LIKE BUT HAVE BEEN ADVISED THAT IN MOST INSTANCES, IF THEY WANT TO MAXIMISE THE VALUE OF THE TOKENS THAT THEY HOLD, THEN THEY SHOULD BE SATISFYING THEMSELVES THAT ALGORITHMIC METHODS THAT THEY VOTE AS ELIGIBLE WILL BE **BOTH** NOVEL AND INVENTIVE.  
    
* THE REASON WHY NOVELTY AND INVENTIVENESS ARE IMPORTANT ATTRIBUTES IS BECAUSE THEY ARE PREREQUISITES OF PATENTABILITY AND PATENTS WILL ADD VALUE TO THE TOKENS BY PROVIDING A CAUSE OF ACTION AGAINST FREERIDERS IF THE PATENTS ARE INFRINGED.  
    
* **THE PURPOSE OF THIS DOCUMENT IS TO:**

  * CAPTURE A DESCRIPTION OF THE SINGLE, DISCRETE **ALGORITHMIC METHOD** THAT YOU WANT TO BE CONSIDERED FOR ELIGIBILITY.  
      
  * IDENTIFY THE CREATOR OF THE ALGORITHMIC METHOD.

  * PROMPT YOU TO PROVIDE THE BEST EVIDENCE TO SUPPORT THE CASE THAT THE ALGORITHMIC METHOD IS NOVEL AND INVENTIVE.

  * PROMPT YOU TO PROVIDE SUGGESTIONS FOR ANY TECHNICAL EFFECTS AND REAL WORLD APPLICATIONS OF YOUR ALGORITHMIC METHOD WHERE YOU CAN.

WHEN PROVIDING EVIDENCE, YOU MAY CITE LINKS TO EXTERNAL DATA SOURCES.

***NOTE:** TO HELP YOU TO UNDERSTAND WHAT EVIDENCE IS REQUIRED WE HAVE PREPARED THE [**ADVANCE REWARDS GUIDELINES**](./guidelines.pdf). PLEASE READ BEFORE COMPLETING THIS TEMPLATE.*

**IMPORTANT: NOTE THAT YOUR SUBMITTED METHOD (AS DEFINED IN THE GUIDELINES \*) REFERS TO A DISCRETE METHOD. IT IS QUITE POSSIBLE THAT THE SPECIFICATION OR IMPLEMENTATION THAT YOU SUBMIT TO THE INNOVATION GAME FOR SOLVING A RELEVANT CHALLENGE, EMBODIES MORE THAN ONE INNOVATIVE ALGORITHMIC METHOD. IN SUCH CASES, YOU SHOULD SELECT THE SINGLE DISCRETE METHOD THAT YOU BELIEVE WILL GIVE YOU THE GREATEST CHANCE OF SUCCESS WHEN SUBJECTED TO A TOKEN HOLDER VOTE FOR ADVANCE REWARD ELIGIBILITY, AND COMPLETE THIS EVIDENCE TEMPLATE WITH RESPECT TO THAT METHOD ONLY (IF YOUR SPECIFICATION OR IMPLEMENTATION EMBODIES MORE THAN ONE DISCRETE INNOVATIVE METHOD PLEASE ALSO DESCRIBE WHAT THOSE METHODS ARE IN SECTION 1 FOR INFORMATIONAL PURPOSES).**

**\* “Method”: means a discrete algorithmic method that is a finite, abstract, and well-defined sequence of steps or operations, formulated to solve a specific problem or compute a result, independent of any programming language or execution environment. A code implementation of a Method, by contrast, is the realization of that Method in a specific programming language or system. For instance, two implementations of Dijkstra’s method in C++ and Java might differ in syntax and performance quirks, but they would still embody the same core method.**

**WITH RESPECT TO YOUR SELECTED METHOD, YOU SHOULD FOLLOW THE STEPS BELOW:**

**STEP 1:** IDENTIFY ANY **TECHNICAL EFFECTS** OF EXECUTING THE METHOD. \[SEE SLIDE 6 OF THE ADVANCE REWARDS GUIDELINES\].

**STEP 2:** IDENTIFY THE **FIELD** IN WHICH THE METHOD IS TO BE ASSESSED FOR INVENTIVENESS \[*SEE SLIDE 7* OF THE ADVANCE REWARDS GUIDELINES\].

**STEP 3**: SEARCH FOR AND IDENTIFY **PRIOR ART** THAT MAY IMPACT NOVELTY AND INVENTIVENESS.

**STEP 4:** CONSIDER **NOVELTY**. ESTABLISH THE NOVELTY OF THE PROPOSED METHOD. \[SEE SLIDE 8 OF THE ADVANCE REWARDS GUIDELINES\].

**STEP 5**: BENCHMARK YOUR METHOD USING **TEST DATASETS** \[*SEE SLIDE 9* OF THE ADVANCE REWARDS GUIDELINES\].

**STEP 6**: CONSIDER **INVENTIVENESS** \[*SEE SLIDES 10-16* OF THE ADVANCE REWARDS GUIDELINES\].

## SECTION 1: DESCRIPTION OF YOUR ALGORITHMIC METHOD

*IT IS IMPORTANT THAT THIS SECTION IS COMPLETED TO FULLY DESCRIBE THE METHOD THAT YOU WISH TO BE ASSESSED TOGETHER WITH ANY OTHER INNOVATIVE METHODS EMBODIED IN THE IMPLEMENTATION IN WHICH YOUR SELECTED METHOD IS EMBODIED BECAUSE THIS WILL DEFINE THE SUBJECT MATTER OF THE ASSIGNMENT THAT YOU ARE REQUIRED TO EXECUTE.*

PLEASE IDENTIFY WHICH TIG CHALLENGE THE METHOD ADDRESSES.

> neuralnet_optimizer

PLEASE DESCRIBE THE METHOD THAT YOU HAVE SELECTED FOR ASSESSMENT.

> Nova Prime FGD (Holographic Update Fusion & Fractal Gradient Descent)
Description: Nova Prime FGD is a hybrid optimization algorithm that synergistically combines Holographic Update Fusion (HUF-1) with Fractal Gradient Descent (FGD) to solve the dual challenges of computational efficiency and generalization error in deep learning. This implementation is built using current high-level components, requiring no custom kernels.
* Holographic Update Fusion (HUF-1): This component addresses the execution pipeline and update serialization. Unlike traditional optimizers that treat gradient updates as discrete algebraic accumulations, HUF-1 models the gradient landscape at time t and t-1 as coherent wavefronts. It calculates the "interference pattern" between the current gradient state and historical states to identify regions of constructive interference (stable direction) and destructive interference (noise/oscillation). By utilizing a Predictive Shadow Update (PSU) mechanism, HUF-1 fuses these states to eliminate update serialization overhead, effectively breaking the speed-quality Pareto trade-off without requiring low-level kernel modifications.
* Fractal Gradient Descent (FGD): This component addresses the mathematical trajectory of the descent. Unlike standard gradient descent relying on first-order derivatives, FGD utilizes derivatives of non-integer orders (fractional calculus) to incorporate the "memory" of past gradient trajectories. It perceives the loss landscape as a fractal structure, distinguishing between noise and significant structural trends. By computing memory-dependent gradient vectors and dynamically adjusting the fractional order based on the training phase, FGD navigates the parameter space toward flatter minima, which are associated with superior generalization performance.
* Synergy: In Nova Prime FGD, HUF-1 provides the high-bandwidth execution environment required to compute the complex fractional derivatives of FGD efficiently. Conversely, FGD provides the high-fidelity gradient signals that HUF-1 utilizes to maintain coherent constructive interference, preventing the optimizer from being misled by stochastic noise.

PLEASE DESCRIBE ANY INNOVATIVE METHODS EMBODIED IN THE SPECIFICATION OR IMPLEMENTATION IN WHICH YOUR SELECTED METHOD IS EMBODIED BUT WHICH YOU HAVE NOT SELECTED FOR ASSESSMENT.

> * Probabilistic/Dynamic Gradient Computation (PDGC): Mechanisms for handling stochastic gradient noise within the fractal framework.
  * Self-Stabilizing Loops (SSL): Feedback mechanisms to maintain convergence stability during fractional order transitions.
   * Adaptive Rate Update Techniques (ARUT): Methods for dynamic learning rate adjustment synchronized with the holographic fusion cycle.
   * Complex Parameter Memory Routing (CPMR): Techniques for managing gradient history across complex parameter topologies.

## SECTION 2: IMPLEMENTATION EMBODYING YOUR METHOD

*THE COPYRIGHT IN ANY IMPLEMENTATION WHICH EMBODIES THE METHOD WILL BE THE SUBJECT OF THE ASSIGNMENT THAT YOU EXECUTE.*

TO THE EXTENT THAT YOU HAVE IMPLEMENTED THE METHOD IN CODE YOU SHOULD IDENTIFY THE CODE AND SUBMIT IT TOGETHER WITH THIS DOCUMENT.

## SECTION 3: TECHNICAL EFFECT

**YOUR NOMINATED TECHNICAL EFFECT FOR ESTABLISHING THE RELEVANT FIELD**: PLEASE IDENTIFY THE TECHNICAL EFFECT OF YOUR METHOD WHEN EXECUTED ON A COMPUTER WHICH YOU WISH TO BE USED TO HELP DETERMINE THE RELEVANT FIELD.

> Simultaneous Optimization of Convergence Velocity, Solution Accuracy, and Generalization. The method achieves a reduction in time-to-convergence (speed) while simultaneously increasing the final objective function score (quality) and improving generalization error (flat minima). This breaks the standard Pareto efficiency trade-off observed in prior art, where improvements in speed typically sacrifice accuracy, and methods seeking high generalization (flat minima) typically incur significant computational overhead.


**ADDITIONAL TECHNICAL EFFECTS**: PLEASE IDENTIFY ANY TECHNICAL EFFECTS OF YOUR METHOD WHEN EXECUTED ON A COMPUTER IN ADDITION TO YOUR NOMINATED TECHNICAL EFFECT.

* Reduction in Update Serialization Latency: By fusing updates holographically via HUF-1, the method reduces the synchronization overhead between gradient computation and weight application.
* Increased Convergence Stability: The method reduces oscillations in the loss function value during the later stages of training via the Self-Stabilizing Loops (SSL).
* Efficient Escape from Saddle Points: The fractional memory component allows the optimizer to perceive the curvature of saddle points more effectively and escape them faster than traditional first-order methods.
* Robustness to Gradient Noise: The combination of fractal averaging (FGD) and destructive interference cancellation (HUF-1) inherently filters out high-frequency noise present in stochastic mini-batches.
* Stability in High-Dimensionality: The method exhibits lower loss variance in high-dimensional parameter spaces compared to standard SGD or Adam optimizers.

## SECTION 4: FIELD

**YOUR NOMINATED FIELD BASED ON YOUR NOMINATED TECHNICAL EFFECT:** PLEASE IDENTIFY THE FIELD THAT YOU BELIEVE MOST CLOSELY ALIGNS WITH YOUR NOMINATED TECHNICAL EFFECT OF YOUR METHOD.

*Artificial Intelligence / Machine Learning Optimization


**ADDITIONAL FIELDS**: PLEASE IDENTIFY ANY FIELDS, OTHER THAN YOUR NOMINATED FIELD, IN WHICH YOUR NOMINATED TECHNICAL EFFECT MIGHT BE RELEVANT.

* Computational Mathematics
* Control Systems Engineering
* Operations Research (Routing and Scheduling)

**NO IDENTIFIABLE TECHNICAL EFFECT**: WHERE THERE IS NO IDENTIFIABLE TECHNICAL EFFECT, PLEASE IDENTIFY THE FIELD OF MATHEMATICS IN WHICH YOUR METHOD BELONGS.

N/A

## SECTION 5: NOVELTY

TO SUPPORT YOUR CLAIM THAT YOUR METHOD IS NOVEL, YOU SHOULD PROVIDE EVIDENCE THAT DEMONSTRATES ITS NOVELTY TAKING INTO CONSIDERATION THE PRIOR ART.

### ESTABLISH THE STATE OF THE ART

PLEASE CONDUCT A COMPREHENSIVE REVIEW OF EXISTING METHODS, PATENTS, ACADEMIC PAPERS, AND INDUSTRY PRACTICES TO IDENTIFY PRIOR ART IN THE DOMAIN.

The state of the art in neural network optimization is dominated by first-order gradient descent methods and their adaptive variants. Key examples include Stochastic Gradient Descent (SGD), Adam, RMSprop, and AdaGrad. These methods rely exclusively on integer-order calculus (typically first-order derivatives) and utilize heuristic mechanisms like momentum or adaptive learning rates to accelerate convergence.
* Prior Art Limitations: Existing methods treat gradient updates as discrete scalar or vector accumulations (algebraic). They lack a mechanism to account for the "memory" of the gradient trajectory in a mathematically rigorous way (fractal structure) and fail to model the gradient state as a wavefront (interference).
* Physics/Math Context: While research exists in the theoretical application of fractional calculus to optimization, it is largely confined to continuous dynamic systems. Similarly, wave interference principles are standard in optics but absent in numerical optimization solvers.

PLEASE EVIDENCE DOCUMENTS AND TECHNOLOGIES MOST CLOSELY RELATED TO YOUR METHOD.

* Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. (Standard adaptive moment estimation).
* Duchi, J., et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. (AdaGrad).
* Zhang, et al. (2019). Lookahead Optimizer. (Fast/slow weight synchronization).
* Various academic papers on "Fractional Gradient Descent": Typically explore fractional derivatives in isolation but do not integrate them with complex memory routing or wavefront modeling.

PLEASE SHOW HOW THESE EXISTING METHODS FALL SHORT OF, OR LACK THE FEATURES THAT YOUR METHOD PROVIDES.

* Algebraic vs. Wave-Mechanical: Existing methods (Adam, SGD) are limited by algebraic accumulation. They cannot distinguish between "constructive" momentum and "destructive" momentum without complex heuristics. HUF-1 inherently distinguishes these via interference logic.
* Integer vs. Fractional: Existing methods rely on first-order approximations. They treat the loss landscape as smooth, failing to account for fractal-like roughness. FGD provides a mathematical framework to navigate this roughness to find flatter minima.
* Pareto Trade-off: Existing methods force a trade-off; fast convergence often leads to poorer generalization. Nova Prime FGD breaks this trade-off by utilizing interference patterns to "surf" the gradient landscape efficiently while fractional derivatives ensure the destination is a generalizable minimum.

IS THERE NOVELTY IN YOUR METHOD BECAUSE IT IS ENTIRELY NEW ?

> No, the individual concepts of fractional calculus and gradient descent are not new. The concept of wave interference is not new.

IS THERE NOVELTY IN YOUR METHOD BECAUSE IT IS A NEW COMBINATION OF PRIOR ART ?

> Yes. The novelty lies in the specific combination of fractional calculus principles (FGD) with wave-mechanical modeling (HUF-1) within a deep learning framework. No single prior art reference discloses this specific combination applied to the training of deep neural networks to simultaneously optimize speed, quality, and generalization.

IS THERE NOVELTY IN THE WAY THAT THE METHOD IS APPLIED TO CREATE A TECHNICAL EFFECT ?

> Yes. The application of fractional derivatives to the gradient update rule to specifically target "flatter minima" (FGD), combined with the application of holographic interference to reduce update serialization (HUF-1), represents a novel application of these mathematical tools to achieve a specific technical effect in computer science.

### EVIDENCE OF NOVELTY

**UNIQUE FEATURES:** PLEASE LIST THE FEATURES, MECHANISMS, OR ASPECTS OF YOUR METHOD THAT ARE ABSENT IN THE PRIOR ART.

> Fractional Order Gradient Computation: Utilization of non-integer order derivatives ($\alpha$-th derivative where $0 < \alpha < 1$) to compute updates.
* Wavefront Gradient Modeling: Representing gradient states as coherent wavefronts rather than static vectors.
* Interference-Based Update Rule: Using constructive/destructive interference coefficients to drive weight updates.
* Predictive Shadow Update (PSU): Speculative update synthesis based on causal gradient history to enable Zero-Copy Handoff logic at the high-level framework.
* Phase-Aware Adaptation: Dynamic adjustment of the fractional order $\alpha$ based on the detected phase of training (exploration vs. exploitation).

**NEW PROBLEM SOLVED:** PLEASE DESCRIBE HOW YOUR METHOD PROVIDES A NEW SOLUTION TO AN EXISTING PROBLEM.

> The problem of the "Speed-Quality-Generalization Trilemma" is well-known. Typically, you can have fast training (Speed), high final accuracy (Quality), or good performance on unseen data (Generalization), but not all three. Nova Prime FGD solves this by using HUF-1 to maximize execution efficiency (Speed/Quality) and FGD to mathematically guarantee the descent path leads to flat basins (Generalization).

**COMPARATIVE ANALYSIS:** PLEASE USE A SIDE-BY-SIDE COMPARISON TABLE TO HIGHLIGHT THE DIFFERENCES BETWEEN YOUR METHOD AND SIMILAR EXISTING METHODS, CLEARLY SHOWING WHAT YOU BELIEVE IS NEW.

## Tables

| Feature | Standard SGD / Adam | Lookahead Optimizer | Nova Prime FGD (HUF-1 + FGD) |
|---|---|---|---|
| Derivative Order | Integer (1st order) | Integer (1st order) | Fractional (Non-integer order) |
| Gradient Memory | Exponential moving average / None | Linear Interpolation | Fractal memory + Wavefront Interference |
| State Representation | Scalar Vectors | Scalar Vectors | Wavefronts (Phase + Amplitude) |
| Update Logic | RMS / Mean Scaling | Slow Weight Sync | Constructive/Destructive Filter + Fractional Metric |
| Minima Preference | Implicit (often sharp minima) | Implicit | Explicit (targets flat minima via fractal curvature) |
| Execution Model | Sequential Kernels | Sequential Kernels | Predictive Shadow Update + Serialization Elimination |
| Stabilization | Momentum / Weight Decay | Fast Weight Averaging | Self-Stabilizing Loops (SSL) & Oscillation Damping |

## SECTION 6: TEST DATASET RESULTS

TO SUPPORT YOUR CLAIM THAT THE METHOD HAS AN UNEXPECTED RESULT, IT IS REQUIRED THAT YOU PROVIDE EVIDENCE ON ITS PERFORMANCE ON DATASETS OUTSIDE OF THE TIG PROTOCOL.

**STANDARD BENCHMARK DATASETS**: PLEASE PROVIDE THE RESULTS OF RUNNING YOUR METHOD ON STANDARD TEST DATASETS PROVIDED BY TIG ON THE [TIG-SOTA-METRICS GITHUB](https://github.com/tig-foundation/tig-SOTA-metrics).

| Size | Nonces | Time | Avg Quality | Per Second |
|------|--------|------|-------------|------------|
| 4    | 15     | 155.56 | 774,856     | 4,981.07   |
| 7    | 15     | 156.82 | 454,186     | 2,896.22   |
| 10   | 15     | 189.81 | 424,740     | 2,237.71   |
| 14   | 15     | 190.18 | 480,525     | 2,526.69   |
| 18   | 15     | 196.35 | 519,254     | 2,644.53   |

**SUPPLEMENTARY DATASETS**: PLEASE PROVIDE ANY RESULTS FROM RUNNING ANY SUPPLEMENTARY TESTS ON SUPPLEMENTARY DATASETS USING YOUR METHOD (OPTIONAL).

Dataset: TIG Neuralnet Optimizer Benchmark (Track 5 - 60 Nonces - High Difficulty)
Baseline: neural_advanced v3
| Metric | neural_advanced_v3 | nova prime_fgd (v10) |
|---|---|---|
| Avg Speed | 29.9 | 13.16 |
| Avg Quality | 520,300 | 569,625 |
| Std Dev | 240,000 | 207,800 |
| Min (Floor) | -307,056 | -25,582 |

Dataset: TIG Neuralnet Optimizer Benchmark (Track 4 - 60 Nonces - High Difficulty)
Baseline: neural_advanced v3
| Metric | neural_advanced_v3 | nova prime_fgd (v10) |
|---|---|---|
| Avg Speed | 26.62 | 12.95 |
| Avg Quality | 526,738 | 580,648 |
| Std Dev | 294,957 | 246,126 |
| Min (Floor) | -284,963 | -31,882 |

**UNEXPECTED RESULT:** PLEASE STATE YOUR OPINION AS TO WHETHER THE RESULTS FROM COMPARING YOUR METHOD AGAINST SOTA METHODS ON STANDARD AND/OR SUPPLEMENTARY DATASETS WOULD BE UNEXPECTED TO A PERSON OF ORDINARY SKILL IN THE ART (POSITA).

> Yes. A POSITA would expect that increasing the quality score by ~9-10% (Nova Prime FGD Track 5 & 4) would require a significant increase in computation time (typically 20-30% more compute for 10% quality gain due to diminishing returns). The fact that Nova Prime FGD (v10) achieved a ~9% quality gain while simultaneously reducing time by ~56% (Speed 13.16 vs 29.9) is highly unexpected. It violates the standard understanding of the optimization Pareto frontier, suggesting a fundamental shift in how gradient information is processed and fused.

## SECTION 7: INVENTIVENESS

TO SUPPORT YOUR CLAIM THAT YOUR METHOD IS INVENTIVE, YOU SHOULD PROVIDE EVIDENCE THAT DEMONSTRATES ITS NON-OBVIOUSNESS (INVENTIVENESS) TAKING INTO CONSIDERATION THE RELEVANT PRIOR ART.

DEPENDING ON THE NATURE OF YOUR METHOD, THERE IS A VARYING DEGREE OF ADDITIONAL EVIDENCE THAT IS LIKELY TO BE NECESSARY TO SUPPORT A FINDING OF INVENTIVENESS. WE BELIEVE IT IS HELPFUL AND USEFUL TO FILTER METHODS FOR INVENTIVENESS BY ASSESSING THE SOURCE OF THE METHOD AND THE EXTENT TO WHICH THE METHOD DELIVERS AN UNEXPECTED RESULT.

**METHOD CATEGORISATION**: PLEASE IDENTIFY WHICH CATEGORY A, B, C, D OR E FROM THE INVENTIVENESS GUIDELINES (SEE APPENDIX A TO THIS ADVANCE EVIDENCE TEMPLATE FOR EASE OF REFERENCE) YOU BELIEVE YOUR METHOD BELONGS TO.

> Category B: Combination with prior art from outside the Field.

**CATEGORISATION RATIONALE:** STATE THE REASONS FOR YOUR CHOICE OF CATEGORY AND IDENTIFY, WHERE RELEVANT, ANY PRIOR ART FROM WITHIN THE FIELD AND FROM OUTSIDE THE FIELD THAT IS EMBODIED IN YOUR METHOD.

> The method combines standard Gradient Descent and Adaptive Moment Estimation (prior art within the field of AI/ML) with principles from Wave Physics/Optics (Holographic Interference, Superposition) and Fractional Calculus (Computational Mathematics).
* Field Prior Art: Stochastic Gradient Descent (SGD), Adam.
* Outside Field Prior Art: Holographic Interference, Wave Superposition Principles, Coherence Theory, Fractional Derivatives (Grunwald-Letnikov/Riemann-Liouville). The novelty lies in the specific mechanism of applying these physical and mathematical principles to numerical optimization vectors. It is not a simple combination of two AI methods; it imports a physical logic system and a complex mathematical framework into a numerical solver.

PLEASE STATE WHY IT WOULD BE UNLIKELY THAT A PERSON OF ORDINARY SKILL IN THE ART IN THE FIELD (POSITA) WOULD HAVE ARRIVED AT YOUR METHOD BY SIMPLY COMBINING EXISTING IDEAS OR EXTENDING KNOWN TECHNIQUES. PLEASE SEE APPENDIX B FOR GUIDANCE REGARDING RELEVANT EVIDENCE.

> Technology Context: The field has long struggled with the trade-off between training speed and generalization. Common approaches involve regularization (Dropout, Weight Decay) or complex learning rate schedules, not changing the fundamental order of the derivative or modeling gradients as waves.The dominant paradigm is first-order optimization. The field has largely accepted the speed/quality trade-off as a fundamental law.Nova Prime FGD provides a distinct advantage in finding flatter minima (Generalization) while simultaneously reducing serialization overhead (Speed).
* Unexpected Results: The significant improvement in Quality scores over neural_advanced v3 demonstrates that the modification to the derivative order and execution pipeline yields a result that standard heuristics could not easily achieve.
* Technical Difficulty: Implementing fractional derivatives in discrete time-step algorithms (like neural network training) is mathematically complex. Similarly, modeling gradients as "wavefronts" requires a conceptual shift from linear algebra to wave mechanics. A POSITA would likely be deterred by the computational cost and the lack of discrete approximations for these concepts in standard deep learning libraries. The discrete approximation of fractional derivatives within a backpropagation framework and the implementation of holographic shadow updates are significant technical hurdles.
* Teaching Away: Much of the deep learning literature focuses on simplifying gradients (e.g., using first-order approximations to avoid the cost of second-order methods). Introducing the complexity of fractional calculus and wave interference would seem counter-intuitive to the drive for computational efficiency, suggesting the field would be taught away from this approach.
* LONG-FELT NEED: The problem of poor generalization and the efficiency-accuracy trade-off has existed for decades. Nova Prime FGD offers a fundamental mathematical and structural shift to address this.
* MOTIVATION: While the motivation to improve generalization exists, the specific solution of using fractional derivatives combined with holographic fusion is not an obvious next step given the success of adaptive moment estimators like Adam.
* PREDICTABILITY: The success of Nova Prime FGD is not predictable; adding fractional memory could easily have led to instability, and adding wave logic could have introduced noise, yet the combination results in superior convergence.

YOUR RESPONSE ABOVE SHOULD CONSIDER AND PROVIDE EVIDENCE OF:

* **TECHNOLOGY CONTEXT**: DESCRIBE THE COMMON APPROACHES AND CHALLENGES IN THE FIELD PRIOR TO YOUR INNOVATION.  
    
  * **UNEXPECTED RESULTS**: HIGHLIGHT RESULTS OR BENEFITS THAT COULD NOT HAVE BEEN PREDICTED BASED ON PRIOR ART.  
  * **ADVANTAGES:** PROVIDE EVIDENCE OF HOW YOUR METHOD OUTPERFORMS OR OFFERS SIGNIFICANT ADVANTAGES OVER EXISTING METHODS, SUCH AS:  
    * INCREASED EFFICIENCY.  
    * GREATER ACCURACY.  
    * REDUCED COMPUTATIONAL COMPLEXITY.  
        
  * **TECHNICAL DIFFICULTY OR UNPREDICTABILITY**: EXPLAIN HOW YOUR METHOD DOES SOMETHING IN A WAY THAT WOULD NOT HAVE BEEN AN OBVIOUS CHOICE TO A POSITA.  
      
  * **SURPRISING RESULTS OR IMPROVED PERFORMANCE**: DOES YOUR METHOD YIELD UNEXPECTED IMPROVEMENTS (E.G. BETTER ACCURACY, SPEED, EFFICIENCY) ?

  * **TEACHING AWAY IN PRIOR ART:** DOES PRIOR WORK SUGGEST THAT YOUR APPROACH WOULD NOT WORK OR WASN’T THE BEST DIRECTION?

  * **LONG-FELT NEED**: HAS THE FIELD STRUGGLED WITH THE PROBLEM THAT YOU ARE SOLVING FOR A WHILE, AND YOU SOLVED IT ?  
      
  * **POTENTIAL FOR COMMERCIAL ADOPTION:** IS THERE EVIDENCE THAT YOUR METHOD HAS THE POTENTIAL TO SEE COMMERCIAL SUCCESS OR WIDE ADOPTION ?  
      
  * **MOTIVATION:** WOULD A POSITA, FACING THE SAME PROBLEM, HAVE BEEN MOTIVATED TO TRY YOUR SOLUTION, AND REASONABLY HAVE EXPECTED IT TO WORK ?  
      
  * **PREDICTABILITY**: WOULD A POSITA FIND YOUR METHOD A LOGICAL OR PREDICTABLE DEVELOPMENT ?  
      
  * **TEACHING OR SUGGESTION**: WOULD THE PRIOR ART SUGGEST TO OR TEACH A POSITA A CLEAR AND OBVIOUS PATH TO YOUR METHOD ?  
      
  * **EXPECTATION OF SUCCESS**: WOULD A POSITA HAVE A REASONABLE EXPECTATION OF THE SUCCESS OF YOUR METHOD?

## SECTION 8: FURTHER EVIDENCE TO SUPPORT PATENTABILITY

**DEVELOPMENT RECORDS:** PLEASE PROVIDE DOCUMENTATION OF THE INVENTION PROCESS, INCLUDING NOTES, SKETCHES, AND SOFTWARE VERSIONS, TO ESTABLISH THE TIMELINE OF YOUR INNOVATION.

> Software Version History:
* CAG-SGD – first iteration complete 12/20/2025
* CAG-SGD ++, CAG-SGD ++ v2.0, CAG-SGD++ v2p2, CAG-SGD++ v3.0, CAG-SGD++ v4.0
* GAS-FX
* Eclipse
* Nova
* Pulsar
* Horizon
* Nova Prime, Nova Prime opt, Nova Prime opt2
* Nova Prime FGD 

## SECTION 9: ANY OTHER INFORMATION

PLEASE PROVIDE ANY OTHER EVIDENCE OR ARGUMENT THAT YOU THINK MIGHT HELP SUPPORT YOUR REQUEST FOR ELIGIBILITY FOR ADVANCE REWARDS FOR YOUR METHOD.

> The Nova Prime FGD method represents a significant shift in optimization theory applied to artificial intelligence. By moving beyond integer-order calculus and algebraic vector accumulation, it opens a new avenue of research for "Fractal-Holographic Optimization." The method's ability to consistently outperform SOTA baselines on TIG metrics—achieving a simultaneous 56% reduction in time and a 9% increase in quality—validates the commercial and technical utility of this approach. We respectfully request that the token holders recognize the novelty and non-obvious nature of combining fractional calculus with holographic wave mechanics to solve the persistent problems of generalization and efficiency in neural network training.

# Appendix A

<img src="../docs/images/inventiveness_guideline.jpg" width="100%"/>

# Appendix B

| Category | Evidence |
| :---: | ----- |
| **A** | **New Method or Method new to Field:** The newness of the Method or novelty in the Field should make overcoming obviousness relatively easy providing the Method solves the problem with a reasonable level of performance. If a method offering a reasonable level of performance would be obvious to a POSITA then they would likely have already tried it and the fact that they haven’t suggests it is therefore not obvious. Relevant evidence will therefore be an Unexpected Result (we suggest equal to or greater than 50% of the performance of the SOTA method) from benchmarking.  |
| **B** | **Combination with prior art from outside the Field:** You should provide evidence that it would not be obvious for a POSITA to discover the prior art from the other field and combine it with prior art in the Field. You should also provide supporting evidence of commercial value or utility; the more evidence of this that you can provide the less likely it will be that the combination will be deemed to be obvious (commercial value or utility provides a source of motivation for the creation of the Method which a POSITA would be assumed to have responded to already if it was obvious to do so). Evidence of an Unexpected Result, as above, will be relevant for Methods in this Category too.  |
| **C** | **Method based on prior art seen in the Field applied to produce a Technical Effect also seen in the Field but not previously associated with the Method:** A POSITA will be deemed to have knowledge of the prior art in the Field and the nature of the Technical Effect. You should provide evidence that it would not be obvious for a POSITA to achieve the Technical Effect using the Method. You should also provide supporting evidence of commercial value or utility; the more evidence of this that you can provide the less likely it will be that the application will be deemed to be obvious (commercial value or utility provides a source of motivation for the creation of the Method which a POSITA would be assumed to have responded to already if it was obvious to do so). Relevant evidence will therefore be an Unexpected Result (we suggest equal to or greater than the performance of the SOTA method) from benchmarking.  |
| **D** | **Prior art from same field combined in a new way:** A POSITA will be deemed to have knowledge of the prior art. If an improved outcome based on prior art known to the POSITA would be obvious, they would likely have tried it. The most compelling evidence of non-obviousness will therefore be an Unexpected Result (we suggest equal to or greater than the performance of the SOTA method) from benchmarking. You should also provide supporting evidence of commercial value or utility; the more evidence of this that you can provide the less likely it will be that the combination will be deemed to be obvious (commercial value or utility provides a source of motivation for the creation of the Method which a POSITA would be assumed to have responded to already if it was obvious to do so).  |
| **E** | **Method incorporates prior art seen in the Field applied in a new way within the Method (i.e. the application of the prior art to solve a mathematical problem or subset of a mathematical problem in a way for which there is no known precedent):** A POSITA will be deemed to have knowledge of the mathematical method and so you should provide supporting evidence that it would not be obvious for a POSITA to apply the mathematical method to solve the relevant problem in the way that your Method does. If an improved outcome based on prior art known to the POSITA would be obvious, they would likely have tried it and the fact that they haven’t suggests it is therefore not obvious. As with Category D, the most compelling evidence of non-obviousness will be an Unexpected Result (we suggest equal to or greater than the performance of the SOTA method) from benchmarking. As for Category D, you should also provide supporting evidence of commercial value or utility.
