# Codebase

This directory contains the code used for running the experiments for this thesis.

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
