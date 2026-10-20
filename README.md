# On Data Scaling in Tabular ML

Code and results for [*On Data Scaling in Tabular ML: More Data, Better Generalization?*](https://arxiv.org/abs/2610.26794).
We study training-data scaling on eight real-world datasets. More data generally
improves performance, but gains under distribution shift depend on how models
use validation data.

## Run

Install [uv](https://docs.astral.sh/uv/) and
[prepare the data](src/preprocessing/README.md). Run from the repository root.

```sh
uv sync --locked
export CUDA_VISIBLE_DEVICES=0
uv run tune experiments/main/mlp/muon/weather/10000/tuning --resume
uv run evaluate from-tuning experiments/main/mlp/muon/weather/10000/tuning \
   --n-seeds 10 --save-experiments --resume
```

Each method/dataset/training-size combination is run separately:

- **MLP, MLP-PLE, TabM and tree models:** 100 TPE trials in `tuning/`, followed
  by 10-seed evaluation of the selected configuration in `evaluation/`.
- **TabPack:** pack-based ensemble selection.
- **Foundation models:** direct evaluation.

Results are written alongside configs. `--resume` skips completed runs.
The main benchmark uses `time` validation/test (`core` for Istella).

Additional dependencies:

- **TabPFN/TabICL:** `uv sync --locked --group foundation`
- **Kumo:** `uv sync --locked --group kumo`

Include the same group in `uv run` commands. Pretrained weights are not included.

## Results

- [Data scaling and validation-data experiments](experiments/main)
- [ID-validation control for MLP and LightGBM](experiments/iid)
- [MLP capacity sweep with ID/OOD model selection](experiments/sweep)

## Citation

If you use this code or results, please cite:

```bibtex
@misc{rubachev2026datascaling,
  title = {On Data Scaling in Tabular {ML}: More Data, Better Generalization?},
  author = {Ivan Rubachev and Nikita Lazarev and Artem Babenko},
  year = {2026},
  eprint = {2610.26794},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.26794}
}
```
