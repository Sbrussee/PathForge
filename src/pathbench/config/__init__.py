{
  "mode": "benchmark",
  "data": {"dataset_type": "bag", "root": "/data/tiles", "split_strategy": "kfold", "n_splits": 5},
  "train": {"batch_size": 1, "max_epochs": 20, "precision": 32, "accelerator": "gpu", "devices": 1, "seed": 17},
  "model": {"name": "mil_base", "params": {"embed_dim": 256}},
  "optimization": {"n_trials": 30, "sampler": "TPESampler", "pruner": "MedianPruner"}
}