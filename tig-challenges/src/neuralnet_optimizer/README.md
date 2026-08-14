## Overview

The recent surge in Artificial Intelligence (AI) has been largely driven by deep learning—an approach made possible by vast datasets, highly parallel computing power, and neural network frameworks that support automatic differentiation. Neural networks serve as the fundamental building blocks for these complex AI systems.

The training process of a neural network is often incredibly resource-intensive, requiring massive amounts of data and computational power, which translates to significant financial cost. At the heart of this training process lie optimization algorithms based on gradient descent. These algorithms systematically adjust the network's parameters to minimize a "loss function," effectively teaching the model to perform its designated task accurately. Much of the field's progress was built upon Stochastic Gradient Descent (SGD), which iteratively adjusts network parameters in the direction that most steeply reduces this training loss.

In modern practice, adaptive variants of SGD have become ubiquitous. The most prominent of these is Adam, whose foundational paper is one of the most cited in computer science history. The widespread adoption of optimizers like Adam underscores their central role in enabling deep learning's breakthroughs. Even small gains in optimisation efficiency translate into shorter training times, lower energy usage, and significant cost savings, underscoring the value of continued research into better optimizers.


## Our Challenge

TIG’s neural network optimizer challenge asks innovators to implement an optimizer that plugs into a fixed CUDA-based training framework and trains a multi-layer perceptron (MLP) on a synthetic regression task. The goal is to beat a target (ground truth) test loss threshold derived from the noise level in the data `σ` and an `accuracy_factor`.

- **Task**: Minimize MSE on a held-out validation set during training; final acceptance is based on test loss.
- **Model**: MLP with batch normalization. Hidden layer width is fixed at 256; the number of hidden layers is part of the difficulty.
- **Training budget**: Batch size 128, up to 1000 epochs, early stopping with patience 50.

**Difficulty Parameters**
- Parameter 1: `num_hidden_layers` = Number of hidden layers in the MLP.
- Parameter 2: `accuracy_factor` = Tightens the acceptance threshold; higher values make the target loss stricter.

**Data Generation**:
  Synthetic regression via Random Fourier Features: RFF count K = 128, amplitude scaling `√(2/K)`, lengthscale l = 0.3, and Additive Gaussian noise σ = 0.2. Input dims = 8, Output dims = 1. That is, for an input point $x\in [-1,1]^{8}$ a target point $y \in \mathbb{R}$ is constructed as 

$$y = f(x) + \xi, \quad f(x) = \mathbf{a} \cdot \boldsymbol{\phi}(x), \quad \mathbf{a} \sim \mathcal{N}(\mathbf{0}_K, \mathbf{I}_K),$$

where

$$\boldsymbol{\phi}(x) = \sqrt{\frac{2}{K}} \left[ \cos{(\boldsymbol{\omega}_1 \cdot x + b_1)}, \ldots, \cos{(\boldsymbol{\omega}_K \cdot x + b_K)} \right],$$

with $\boldsymbol{\omega} \sim \mathcal{N}(0, l^{-2} \, \mathbf{I}_8)$ and $b \sim \text{Uniform}(0, 2\pi)$, $\xi \sim \mathcal{N}(0, \sigma^2)$, where $l$ is the lengthscale parameter.

The data has the following split: Train = 1000, Validation = 200, Test = 250.


**Training Loop and Optimizer API**
Innovator optimizers integrate into the training loop via three functions:
- `optimizer_init_state(seed, param_sizes, ...) -> state`. This is a one-time setup function that initialises the optimizer state.
- `optimizer_query_at_params(state, model_params, epoch, train_loss, val_loss, ...) -> Option<modified_params>`. This is an optional “parameter proposal” function: if you return modified parameters, the forward/backward uses them for that batch; the original parameters are then restored before applying updates. This enables lookahead optimizer schemes.
- `optimizer_step(state, model_params, gradients, epoch, train_loss, val_loss, ...) -> updates`. This is the main function in the submission; it receives the current model parameters and gradients and returns per-parameter update tensors. 

You may only change optimizer logic and its internal hyperparameters/state. Model architecture (beyond `num_hidden_layers` from difficulty), data, batch size, and training loop controls are fixed.

Each epoch (iteration of the training loop) consists of: 
* Shuffle the training data and iterate over mini-batches in random order.
* For each mini-batch:
   - Optional parameter proposal: the harness calls optimizer_query_at_params(...). If you return modified parameters, the forward/backward uses them for this batch; the original parameters are restored immediately after.
    - Run a forward pass to compute the batch loss, then a backward pass to compute gradients with respect to the current model parameters.
    - Update computation: the harness calls optimizer_step(...) with the current model parameters and gradients (and optional loss signals). Your function returns per-parameter update tensors.
    - Apply the returned updates to the model parameters.
- After all batches, evaluate on the validation set, track the best validation loss for early stopping, and save the best model so far.

**Scoring and Acceptance**
Your optimizer integrates into the training loop; the harness evaluates the best model state produced during the run (no separate "returned" solution). The evaluated metric is **quality** (a fixed-point integer with 6 decimal places); see the challenge code for how it is derived from test loss and the acceptance threshold.

After training, we compute the average MSE on the test set (`avg_model_loss_on_test`) and compare it to a target computed from the data’s noise:

- Let `alpha = 4.0 - accuracy_factor / 1000.0`.
- Let `ε*² = (alpha / N_test) * Σ(y_noisy - f_true)²` over the test set.

You pass if:
- `avg_model_loss_on_test ≤ ε*²`

Higher `accuracy_factor` lowers `alpha`, making `ε*²` smaller and the challenge harder.



## Applications

Neural networks now underpin everything from chatbots to self-driving cars, and their training efficiency dictates cost, speed, and energy use. Since nearly all of that training hinges on gradient-descent methods, even small optimizer improvements ripple across AI—and into other fields.

Below are some of the highest-impact domains where faster, more reliable training already yields real-world gains:

- **Large language and multimodal models** – Massive chatbots and image generators with hundreds of billions of parameters can shave weeks and millions of dollars off training runs when optimizers become just a few percent more efficient [1].
    
- **Protein structure prediction & drug discovery** – Leaner training pipelines let researchers fold larger protein databases and explore more drug candidates under tight compute budgets [2].
    
- **Autonomous driving & robotics** – Rapidly retraining perception and planning nets on fleets’ weekly data drops delivers safer software to vehicles and robots sooner [3].
    
- **Real-time recommendation engines** – Sharper optimizers cut data-centre power, hardware spend, and user-facing latency for the personalised feeds that dominate the web [4].
    
- **Global weather and climate forecasting** – Neural surrogates now rival traditional supercomputer models; better training enables higher resolution and faster refresh cycles [5].
    

---

### References

1. OpenAI. “GPT-4 Technical Report.” arXiv (2023).
    
2. Jumper, J. _et al._ “Highly accurate protein structure prediction with AlphaFold.” _Nature_ 596 (2021): 583-589.
    
3. Tesla AI Day 2022. “Training FSD on 10 000 GPUs.”
    
4. Naumov, M. _et al._ “Deep Learning Recommendation Model for Personalization and Recommendation Systems.” arXiv (2019).
    
5. Lam, R-M. _et al._ “Learning skillful medium-range global weather forecasting with GraphCast.” _Science_ 382 (2023): 109-115.