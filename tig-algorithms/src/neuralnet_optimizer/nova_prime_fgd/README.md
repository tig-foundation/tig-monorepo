## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** nova_prime_fgd
* **Copyright:** 2026 Brent Beane 
* **Identity of Submitter:** Brent Beane
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses 

Overview
v10 Nova Prime FGD is the official release of the next-generation optimization architecture. This version introduces the Fractal Gradient Descent (FGD) method, leveraging fractional calculus to navigate complex loss landscapes and target flatter minima for improved generalization.
The release includes the standard v10 Nova Prime implementation and the specialized v10 Nova Prime CGA+HUF variant.
Key Features
•	Fractal Gradient Descent (FGD): Utilizes non-integer order derivatives to incorporate gradient memory, distinguishing between structural trends and noise.
•	Phase-Aware Adaptation: Dynamically adjusts the fractional order $\alpha$ based on the training phase (exploration vs. fine-tuning).
•	Flattened Minima Targeting: Explicitly optimizes trajectories to converge on wider, flatter basins in the loss landscape, enhancing model robustness.
•	Covariant Update Execution: Ensures parameter updates remain consistent regardless of parameter scaling or coordinate systems.
