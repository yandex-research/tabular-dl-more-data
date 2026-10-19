# Dataset preparation

Download the raw data from its original provider. For Cooking Time, pass the
directory containing `cooking_time.parquet`:

```sh
uv run preprocess cooking-time /path/to/directory --train-sampling random
```

Outputs go to `data/<dataset>/`. See `uv run preprocess --help` and the
dataset's Python script for input files and options.
