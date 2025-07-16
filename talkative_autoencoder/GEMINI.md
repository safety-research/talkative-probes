# Gemini Code Guide: Consistency Lens

This document provides a guide to understanding and running the Consistency Lens project.

## Project Overview

This project is designed to train a "Consistency Lens," an interpretability model composed of an encoder and a decoder. The goal is to probe a large language model (LLM), referred to as `orig_model`, to understand its internal representations. The lens is trained on the activations of the `orig_model` to generate explanatory text.

The training process is highly configurable and optimized for multi-GPU, distributed environments, leveraging `torch.distributed` and `hydra` for configuration management. Experiment tracking is handled through Weights & Biases (W&B).

## Core Components

-   `lens/`: A Python package containing the core logic for the consistency lens, including data handling, model definitions (encoder, decoder), and training loops.
-   `conf/`: Contains all `hydra` configuration files (`.yaml`). This is the central place to define experiment parameters, including model settings, learning rates, and data paths. The `gemma...` files are specific configurations for training with Gemma models.
-   `scripts/01_train_distributed.py`: The main, executable script for launching a training run. It reads a hydra configuration and sets up the distributed training environment.
-   `scripts/submit_with_config.sh`: A shell script used to submit a training job, likely to a SLURM cluster. It takes a configuration file as an argument and executes the main training script.

## Data Pipeline: Pre-dumping vs. On-the-fly

A key aspect of this project is how it handles the LLM activations used for training. There are two primary modes of operation, configured in your `.yaml` file:

1.  **Dumping Activations (Pre-computation):**
    *   In this mode, you first run a process to compute and save (or "dump") the activations from the `orig_model` to disk.
    *   The training script (`01_train_distributed.py`) then reads these pre-computed activations from the directory specified in `activation_dir`.
    *   This is efficient if you plan to run many training experiments on the same set of activations, as you only need to compute them once.

2.  **On-the-fly Generation:**
    *   This mode is enabled by setting `dataset.on_the_fly.enabled: true` in the configuration.
    *   The training script generates the necessary activations from the `orig_model` in memory during the training loop itself.
    *   This avoids the need for a separate dumping step and saves disk space, making it convenient for one-off experiments or when disk I/O is a bottleneck.

## How to Run Training

1.  **Select or Create a Configuration:**
    *   Choose a configuration file from the `conf/` directory that matches your desired experiment (e.g., `conf/gemma2_2b_unfrozen_nopostfix.yaml`).
    *   Modify the `.yaml` file to set parameters like model paths, learning rates, dataset paths (`activation_dir` if not using on-the-fly), and W&B project details.

2.  **Launch the Training Job:**
    *   Use the `submit_with_config.sh` script to start the training run. Pass the path to your chosen configuration file as an argument.

    ```bash
    # Example of launching a training job
    bash scripts/submit_with_config.sh conf/gemma2_2b_unfrozen_nopostfix.yaml
    ```

## Key Features

-   **Distributed Training:** Natively supports multi-GPU and multi-node training to scale experiments.
-   **Hydra Configuration:** Allows for clean, modular, and overridable experiment configuration from the command line.
-   **Advanced Checkpointing:** The system saves checkpoints and can automatically resume training, with special handling for SLURM preemptions.
-   **W&B Integration:** All metrics, logs, and system information are logged to Weights & Biases for comprehensive experiment tracking and analysis.