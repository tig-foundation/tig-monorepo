## UNIQUE ALGORITHM IDENTIFIER (UAI)

> c002_a058

# BREAKTHROUGH EVIDENCE 

EVIDENCE IN SUPPORT OF A REQUEST FOR ELIGIBILITY FOR BREAKTHROUGH REWARDS

## OVERVIEW

- TIG TOKEN HOLDERS WILL VOTE ON WHETHER YOUR ALGORITHMIC METHOD IS ELIGIBLE FOR BREAKTHROUGH REWARDS

- TIG TOKEN HOLDERS ARE FREE TO VOTE AS THEY LIKE BUT HAVE BEEN ADVISED THAT IF THEY WANT TO MAXIMISE THE VALUE OF THE TOKENS THAT THEY HOLD, THEN THEY SHOULD BE SATISFYING THEMSELVES THAT ALGORITHIC METHODS THAT THEY VOTE AS ELIGIBLE WILL BE BOTH NOVEL AND INVENTIVE

- THE REASON WHY NOVELTY AND INVENTIVENESS ARE IMPORTANT ATTRIBUTES IS BECAUSE THEY ARE PREREQUISITES OF PATENTATBILITY.

- **THE PURPOSE OF THIS DOCUMENT IS TO:**
  - CAPTURE A DESCRIPTION OF THE ALGORITHMIC METHOD THAT YOU WANT TO BE CONSIDERED FOR ELIGIBILTY.

  - TO IDENTIFY THE CREATOR OF THE ALGORITHMIC METHOD.

  - TO PROMPT YOU TO PROVIDE THE BEST EVIDNCE TO SUPPORT THE CASE THAT THE ALGORITHMIC METHOD IS NOVEL AND INVENTIVE.

  - TO PROMPT YOU TO PROVIDE SUGGESTIONS FOR REAL WORLD APPLICATIONS OF YOUR ALGORITHMIC METHOD WHERE YIOU CAN.

WHEN PROVIDING EVIDENCE, YOU MAY CITE LINKS TO EXTERNAL DATA SOURCES.

## SECTION 1

IT IS IMPORTANT THAT THIS SECTION IS COMPLETED IN SUFFICIENT DETAIL TO FULLY DESCRIBE THE METHOD BECAUSE THIS WILL DEFINE THE METHOD THAT IS THE SUBJECT OF THE ASSIGNMENT THAT YOU EXECUTE.

### DESCRIPTION OF ALGORITHMIC METHOD

PLEASE IDENTIFY WHICH TIG CHALLENGE THE ALGORITHMIC METHOD ADDRESSES.

> vehicle_routing

PLEASE DESCRIBE THE ALGORITHMIC METHOD AND THE PROBLEM THAT IT SOLVES.

> 
> **Algorithmic Method Summary**  
> This submission solves a **Vehicle Routing Problem (VRP)** variant using a hybrid approach that combines:  
>
> 1. **Adaptive, Parameterized Savings Construction**  
>    - Instead of using a fixed savings formula (as in Clarke & Wright), each node has a contribution parameter. A merge’s “score” is influenced by the sum/multiplication of the parameters for the two nodes, which can be tuned over iterations.  
>    - This lets the solver concentrate on merging certain routes more aggressively, adapting to the topology or demands.  
>
> 2. **Two-Phase Acceptance Criterion**  
>    - While striving to improve the solution cost, the algorithm sometimes accepts worse parameter configurations via an exponential acceptance function (“\(\exp(-\Delta / \text{scaling})\)”), akin to simulated annealing.  
>    - This “stochastic acceptance” allows the search to escape local minima where deterministic greedy merges might stall.  
>
> 3. **Local Search Enhancements**  
>    - **Intra-route Optimization (2-opt):** Loops through each route to remove crossing edges and reduce total distance.  
>    - **Inter-route Swaps:** Systematically tries to swap nodes across different routes—subject to capacity constraints—to capture improvements that 2-opt alone would miss.  
>    - **Relocate:** Moves single nodes between routes, evaluating each possible insertion position while respecting capacity constraints.  
>    - A specialized "best-improvement first" mechanism checks possible node swaps between routes and applies them in descending order of potential gain, maximizing benefit while avoiding conflicts. This approach allows parallelization and hence significant performance improvement in contrast with sequential solution improving.
>
> 4. **Dynamic Pruning and Early-Exit Criteria**  
>    - Uses thresholds (e.g., a “max distance / N” cutoff for merges, or an early exit if a solution’s cost is too high) to avoid fruitless merges or iterations.  
>    - Tracks stagnation (no improvement over multiple iterations) to terminate the search early.  
>
> **Problem Addressed**  
> This approach addresses **Vehicle Routing** with capacity constraints, focusing on finding cost-effective tours that collectively visit all customer nodes without exceeding vehicle capacity. By injecting an adaptive parameter into the classical savings framework and unifying it with iterative local search and a quasi-simulated annealing acceptance, it can overcome many pitfalls of purely greedy or purely metaheuristic solutions.


## SECTION 2

THE COPYRIGHT IN THE IMPLEMENTATION WILL BE THE SUBJECT OF THE ASSIGNMENT THAT YOU EXECUTE.

### IMPLEMENTATION OF ALGORITHMIC METHOD

TO THE EXTENT THAT YOU HAVE IMPLEMENTED THE ALGORITHMIC METHOD IN CODE YOU SHOULD IDENTIFY THE CODE AND SUBMIT IT AS AN EMAIL ATTACHMENT TOGETHER WITH THIS DOCUMENT.


## SECTION 3

### NOVELTY AND INVENTIVENESS

To support your claim that an algorithmic method is novel and inventive, you should provide evidence that demonstrates both its uniqueness (novelty) and its non-obviousness (inventiveness) in relation to the existing state of the art.

### Establish the State of the Art

- **Prior Art Search:** Conduct a comprehensive review of existing methods, algorithms, patents, academic papers, and industry practices to identify prior art in the domain.
  - Highlight documents and technologies most closely related to your method.

    > Clarke and Wright Savings Algorithm (classic approach).
  Simulated Annealing and Tabu Search variants for VRP.
  Genetic Algorithms and other metaheuristics for VRP.

  - Show where these existing methods fall short or lack the features your algorithmic method provides.
  
    > - Rigid Parameterization: Classical savings-based heuristics do not typically allow per-node weighting that changes over iterations, limiting adaptivity. 
    > - Uncoordinated Metaheuristics: Many hybrid approaches fail to integrate a smoothly parameterized savings step with a dynamic acceptance criterion. Metaheuristics are often “bolted on” after an initial route construction, not continuously tuned.
    > - Limited Inter-Route Coordination: Some known solvers rely heavily on local searches within a single route or basic route merging, but do not systematically explore route–route swaps with a “best-improvement-first” aggregator that avoids conflicts.

- **Technical Context:** Describe the common approaches and challenges in the field prior to your innovation.

    > The VRP field has long recognized that combining constructive heuristics with iterative local search is powerful. However, constructive heuristics (like savings) typically remain static. Metaheuristics (like simulated annealing) often operate purely at the solution level, not in the parameter space of the constructive method.
    By making the savings calculation itself dynamic—guided by evolving per-node parameters—this approach smoothly unifies the initial construction and subsequent improvements, ensuring that each iteration’s constructive step remains relevant to the partial solutions discovered so far.

### Evidence of Novelty

- **Unique Features:** List the features, mechanisms, or aspects of your algorithmic method that are absent in prior art.

Here's the corrected text:

> 1. Parametric Savings Factor  
>    - Each node i is assigned a parameter (Bl <= param_i <= Bh), where Bl and Bh are lower and upper bounds of parameters.  
>    - Merges are scored as (param_i + param_j)*(distance_out_and_back - distance_merge).  
>    - Dynamically altering these parameters over iterations is uncommon in standard VRP heuristics.  
>
> 2. Stochastic Acceptance for Parameter Changes  
>    - The code calculates a cost delta (`Δ`) when switching from current parameters to new ones and decides acceptance based on `exp(-Δ / scale)`.  
>    - This type of acceptance is rarely seen integrated directly into a savings-based approach.  
>
> 3. Selective Pairwise Merging  
>    - The algorithm uses a distance threshold to skip merges that are unlikely to be beneficial and reduce search space.  
>    - This targeted pruning is more efficient than blindly computing all pairwise merges.  
>
> 4. Iterated Postprocessing  
>    - After constructing or updating the solution, 2-opt, route-route swaps and relocates are run repeatedly until no improvement is found.  
>    - Inter-route swaps and relocate operator are aggregated in a single pass and applied in rank order, avoiding conflicts—this differs from many solutions that check swaps one by one and accept or reject them on the spot. 
    
- **New Problem Solved:** Describe how your algorithmic method provides a novel solution to an existing problem.

  > The algorithm addresses the problem of **adaptively tuning the classical savings construction** to suit the problem instance’s capacity and spatial/demand distributions. It extends standard local searches to keep pace with the new solutions generated by parameter changes, ensuring that the final solution benefits from both flexible construction and thorough refinement.  
    
- **Comparative Analysis:** Use a side-by-side comparison table to highlight the differences between your method and similar existing methods, clearly showing what’s new.

>   
> > | **Feature**                                           | **Classical Savings**       | **Generic Metaheuristics**        | **New Method**                                                                    |  
> > |-------------------------------------------------------|-----------------------------|-----------------------------------|-----------------------------------------------------------------------------------|  
> > | Node-Specific Parameters                              | No (static formula)         | Indirect or none                  | **Yes; each node has a separate param value [Bl,Bh]**                          |  
> > | Continuous Parameter Adaptation                       | No                           | Rarely (not in savings)           | **Yes; parameters updated with acceptance criterion**                             |  
> > | Distance Thresholding in Merge                        | Sometimes basic             | Not typically used                | **Yes; merges only if `dist <= threshold`**                                     |  
> > | 2-OPT + Inter-Route Swaps + Relocates                            | Possibly 2-OPT only         | Varies                            | **Yes; combined route-level local search with multi-route best-improvement**     |  
> > | Exponential Acceptance (Simulated-Annealing-Like)     | No                           | Yes, but rarely in savings        | **Yes; integrated specifically into param adjustments**                          |                                              |  
>   
    
### Evidence Inventiveness of Inventiveness

- **Non-Obviousness:** Argue why a skilled person in the field would not have arrived at your method by simply combining existing ideas or extending known techniques.
  - Demonstrate that the development involved an inventive step beyond straightforward application of existing principles.
  - Unexpected Results: Highlight results or benefits that could not have been predicted based on prior art.
    
  > Recent papers related to saving Clarke-Wright algorithm show that researches focus more on heuristic that use global parameters or different kind of pertrubation techniques. This papers looks at the problem at different angle and focus on initializing good initial solution by adjusting weight of each node and refining solution with local search techniques.
    
- **Advantages:** Provide evidence of how your algorithm outperforms or offers significant advantages over existing methods, such as:
  - Increased efficiency.
  - Greater accuracy.
  - Reduced computational complexity.
  - Novel applications.

  > In comparison with other Clarke Wright algorithms, described above method gives us close to optimal solutions in less than half of second for old benchmarks and gap in 0-2% between novel method and BKS for Uchoa benchmarks.

### Supporting Data

- **Experimental Results:** Include performance benchmarks, simulations, or empirical data that substantiate your claims of novelty and inventiveness.

  > Results for benchmark A:
784 
661 
742 
778 
799 
678 
949 
730 
822 
831 
937 
949 
1146
914 
1086
1017
1167
1073
1354
1035
1298
1616
1315
1401
1174
1164
1763
Optimal values can be found here: http://vrp.galgos.inf.puc-rio.br/index.php/en/ 
Note that result heavy depend on number of iteration, so to make fair comparison we should aim same time limit as other algorithms do.

- **Proof of Concept:** If possible, show a working prototype or implementation that validates the method.

  > -

### Citations and Expert Opinions

- **Literature Gaps:** Affirm, to the best of your knowledge, the absence of similar solutions in published literature to reinforce your novelty claim.

> No publications have been identified that combine:  
> 1. **Node-based, dynamic parameters** in a savings proach;  
> 2. **Adaptive acceptance** akin to simulated nealing directly on the parameter space of the vings heuristic;  
> 3. **Iterated multi-route local searches** with a est-improvement aggregator” for cross-route swaps.  
  

- **Endorsements:** Include reviews or opinions from industry experts, researchers, or peer-reviewed publications that evidences the novelty and impact of your algorithm.

  > -

## SECTION 4

### EVIDENCE TO SUPPORT PATENTABILITY

- **Development Records:** Please provide documentation of the invention process, including notes, sketches, and software versions, to establish a timeline of your innovation.

  > Draft version was submitted before as advanced_routing.

## SECTION 5

### SUGGESTED APPLICATIONS

- Please provide suggestions for any real world applications of your abstract algorithmic method that occur to you.

  > Tools or staff with capacity-limited vehicles, requiring flexible scheduling.  

## SECTION 6

### ANY OTHER INFORMATION

- Please provide any other evidence or argument that you think might help support you request for eligibility for Breakthrough Rewards.

  > -