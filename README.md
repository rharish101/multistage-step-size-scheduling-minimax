# Multistage Step Size Scheduling for Minimax Problems

This is a repository for my Master's thesis under the [Optimization & Decision Intelligence group](https://odi.inf.ethz.ch/) at ETH Zürich.
The full thesis can be found [here](https://www.research-collection.ethz.ch/handle/20.500.11850/572991).

#### Supervisors
- [Ilyas Fatkhullin](https://scholar.google.com/citations?user=G2OzFpIAAAAJ)
- [Dr. Anas Barakat](https://scholar.google.com/citations?user=5YyyWPkAAAAJ)
- [Prof. Dr. Niao He](https://odi.inf.ethz.ch/niaohe)

## Abstract

In response to the increasing popularity of adversarial approaches in machine learning, much research has been done to tackle the challenges of minimax optimization.
Approaches that have proven themselves in minimization are now being considered for potential use in minimax.
Multistage step size scheduling is one such class of approaches, that have shown not only near-optimal convergence rates in minimization, but also impressive performance across multiple experiments.
In this thesis, we formulate Step Decay and Increasing-Phase Step Decay, two kinds of multistage step size scheduling algorithms, for stochastic first-order optimization in minimax problems.
We then study these multistage schedulers for three classes of minimax problems: two-sided Polyak-Łojasiewicz (PL) functions, non-convex one-sided PL functions, and non-convex non-concave functions, representing increasing difficulties of minimax optimization.
Our theoretical analysis of their convergence rates for two-sided PL and non-convex PL functions shows that multistage schedulers have better convergence rates w.r.t. the variance of the stochastic gradients, thus improving robustness of the optimization in the presence of stochasticity.
We also show how multistage schedulers also improve performance when run on practical machine learning scenarios, such as training Generative Adversarial Networks for image generation.

## Code

The [code](code/) directory contains the code used for running the experiments.
Please refer to the README located at [`code/README.md`](code/README.md) for details on how to run the code.

## Citation

```
@MASTERSTHESIS{20.500.11850/572991,
	copyright = {In Copyright - Non-Commercial Use Permitted},
	year = {2022},
	type = {Master Thesis},
	author = {Rajagopal, Harish},
	size = {67 p.},
	language = {en},
	address = {Zurich},
	publisher = {ETH Zurich},
	DOI = {10.3929/ethz-b-000572991},
	title = {Multistage Step Size Scheduling for Minimax Problems},
	school = {ETH Zurich}
}
```
