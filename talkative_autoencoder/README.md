# Consistency Lenses: Scalable LLM Interpretation via textual bottlenecks

This document outlines the architectural plan for training talkative probes as autoencoders to interpretthe internal states of LLMs by forcing them through a textual, human-readable bottleneck.

```bash
make
```
The main entrypoint is scripts/submit_with_config.sh, which torchruns 01_distributed_train.py. We configure hyperparameters via `conf/` using hydra.

Default is data parallel.