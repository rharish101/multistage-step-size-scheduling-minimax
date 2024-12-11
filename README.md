# Multistage Step Size Scheduling for Minimax Problems

This is a repository for the code used for running the experiments for my Master's thesis under the [Optimization & Decision Intelligence group](https://odi.inf.ethz.ch/) at ETH Zürich.
The full thesis can be found [here](https://doi.org/10.3929/ethz-b-000572991).

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

## Setup
[Poetry](https://python-poetry.org/) is used for conveniently installing and managing dependencies.
[pre-commit](https://pre-commit.com/) is used for managing hooks that run before each commit, to ensure code quality and run some basic tests.

1. *[Optional]* Create and activate a virtual environment with Python >= 3.8.5.

2. Install Poetry globally (recommended), or in a virtual environment.
    Please refer to [Poetry's installation guide](https://python-poetry.org/docs/#installation) for recommended installation options.

3. Install all dependencies, including extra dependencies for development, with Poetry:
    ```sh
    poetry install
    ```

    To avoid installing development dependencies, run:
    ```sh
    poetry install --no-dev
    ```

    If you didn't create and activate a virtual environment in step 1, Poetry creates one for you and installs all dependencies there.
    To use this virtual environment, run:
    ```sh
    poetry shell
    ```

4. Install pre-commit hooks:
    ```sh
    pre-commit install
    ```

**NOTE:** You need to be inside the virtual environment where you installed the above dependencies every time you commit.
However, this is not required if you have installed pre-commit globally.

## Tasks

The optimizers are tested on multiple tasks.
Each task involves training a certain model in a certain manner (supervised, unsupervised, etc.) on a certain dataset.
Every task is given a task ID, which is used when running scripts.

The list of tasks implemented, along with their IDs, are:

Task ID | Description | Thesis Chapter
-- | -- | --
`rls/low/full` | Train an RLS model for the low condition number dataset using deterministic AGDA | Chapter 3
`rls/low/stoc` | Train an RLS model for the low condition number dataset using SAGDA | Chapter 3
`rls/high/full` | Train an RLS model for the high condition number dataset using deterministic AGDA | Chapter 3
`rls/high/stoc` | Train an RLS model for the high condition number dataset using SAGDA | Chapter 3
`covar/linear` | Train a WGAN with a linear generator for learning a zero-mean multivariate Gaussian. | Chapter 4
`covar/nn` | Train a WGAN with a neural network generator for learning a zero-mean multivariate Gaussian. | Chapter 4
`cifar` | Train an SN-GAN on the CIFAR10 dataset. | Chapter 5

For details, please refer to the relevant chapters in the thesis.

## Scripts

All scripts use argparse to parse commandline arguments.
Each Python script takes the task ID as a positional argument.
To view the list of all positional and optional arguments for a script `script.py`, run:
```sh
./script.py --help
```

## Training

For training a model for a task, run the training script `train.py` as follows:
```sh
./train.py task
```

## Hyper-Parameters

### Configuration
Hyper-parameters can be specified through YAML configs.
For example, to specify a batch size of 32 and total steps of 2000, use the following config:
```yaml
batch_size: 32
total_steps: 2000
```

You can store configs in a directory named `configs` located in the root of this repository.
It has an entry in the [`.gitignore`](./.gitignore) file so that custom configs aren't picked up by git.
This directory already contains the configs corresponding to the experiments whose results are used in the thesis.

The available hyper-parameters, their documentation and default values are specified in the `Config` class in the file [`src/config.py`](./src/config.py).

**NOTE:** You do not need to mention every single hyper-parameter in a config.
In such a case, the missing ones will use their default values.

### Tuning
Support for tuning hyper-parameters for the optimizers is available in the tuning script `tune.py`.
Thus, to tune hyper-parameters for models on a certain task, run the tuning script as follows:
```sh
./tune.py task
```

## Logs
Logging is done using [TensorBoard](https://github.com/tensorflow/tensorboard/).
Logs are stored with certain directory structures.
For training, this is:
```
this directory
|_ root log directory
   |_ task name
      |_ experiment name
         |_ timestamped run directory
```

For tuning, this is:
```
this directory
|_ root log directory
   |_ task name
      |_ experiment name
         |_ timestamped tuning run directory
            |_ training run 0 directory
            |_ training run 1 directory
            ...
```

Note that the task name may contain slashes (/).
This would mean that further sub-directories would be made according to the task name.
For example, if the task is `rls/linear/stoc`, then there would be an `rls/` directory, containing a sub-directory `linear/`, which contains `stoc/`.

The timestamp uses the ISO 8601 convention along with the local timezone.
The root log directory can be specified with the `--log-dir` argument.
By default, this is `logs`.

The sub-directory for each training run will contain:
- The latest checkpoint of the trained model, within the `checkpoints` sub-directory
- Training logs, as a file with the prefix `events.out.tfevents.`
- The hyper-parameter config (including defaults), as a YAML file named `hparams.yaml`

The sub-directory for a tuning run will contain:
- Sub-directories for each training run
- The best hyper-parameter config (including defaults), as a YAML file named `best-hparams.yaml`

To view all logs in a directory or in any of its sub-directories, run TensorBoard as follows:
```sh
tensorboard --logdir /path/to/log/dir
```

### Plotting

For plotting the logs in one or more directories, use the plotting script `plot.py` as follows:
```sh
./plot.py task mode /path/to/log/dir/1 /path/to/log/dir/2 ...
```

Here, `mode` specifies the mode of grouping for plotting.
The following modes are available:
- `sched`: This groups all logs by the scheduler used to train them.
    This is the mode used to generate the plots in the thesis.
- `decay`: This groups all logs by the decay factor in the scheduler used to train them.
    This can be used to visualize the effects of different decay values.

The plots will show the mean and standard deviation of task-specific metrics.
These metrics are as follows:

Task ID | Metrics | TensorBoard Tag
-- | -- | --
`rls/low/full` | Distance, Potential | `metrics/distance`, `metrics/potential`
`rls/low/stoc` | Distance, Potential | `metrics/distance`, `metrics/potential`
`rls/high/full` | Distance, Potential | `metrics/distance`, `metrics/potential`
`rls/high/stoc` | Distance, Potential | `metrics/distance`, `metrics/potential`
`covar/linear` | Distance, Gradients w.r.t. x | `metrics/distance`, `gradients/x`
`covar/nn` | Distance, Gradients w.r.t. x | `metrics/distance`, `gradients/x`
`cifar` | FID, Inception Score | `metrics/fid`, `metrics/inception_score`

## Miscellaneous Features

### Multi-GPU Training
For choosing how many GPUs to train on, use the `-g` or the `--num-gpus` flag when running a script as follows:
```sh
./script.py --num-gpus 3
```

This selects three available GPUs for training.
By default, only one GPU is chosen.

### Mixed Precision Training
This implementation supports mixed-precision training, which is disabled by default.
To set the floating-point precision, use the `-p` or the `--precision` flag when running a script as follows:
```sh
./script.py --precision 16
```

Note that mixed-precision training will only provide significant speed-ups if your GPUs have special support for mixed-precision compute.

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
