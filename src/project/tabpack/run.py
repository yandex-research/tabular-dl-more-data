"""Run a TabPack experiment, then re-evaluate its selected ensembles at more seeds.

Adapted from `yandex-research/tabpack`'s `run_tabpack_experiment.py`. TabPack's
`main` (`project.tabpack.model.main`) trains a pack of models and, along the
way, tracks online ensembles over the pack. This module runs that main
experiment and then, for the requested ensembles, re-runs *only* the selected
pack members (via `lib.tools.evaluate.main`) across more seeds to get a more
reliable estimate of the ensemble's quality.
"""

import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Literal

import lib.env
import lib.experiment
import lib.tools.evaluate

from . import model as tabpack_model

type OfflineEnsembleType = Literal['greedy']
type OnlineEnsembleType = Literal['greedy']

_LOGGER = logging.getLogger(__name__)


def _check_source_exp(source_exp: str | Path) -> Path:
    assert lib.experiment.is_experiment(source_exp)
    assert lib.experiment.is_completed(source_exp)
    return Path(source_exp)


def _evaluate_ensemble(
    source_exp: str | Path,
    *,
    name: str,
    n_seeds: int,
    is_offline: bool,
    force: bool = False,
) -> None:
    """Retrieve ids from the selected ensemble (in the main experiment) and
    rerun the method with only the corresponding configs. Works for both
    online and offline ensembles.
    """
    assert n_seeds > 0

    source_exp = _check_source_exp(source_exp)
    source_config = lib.experiment.load_config(source_exp)
    source_report = lib.experiment.load_report(source_exp)

    ensemble_report = source_report['online_ensembles'][name]['report']
    ensemble_unique_ids = sorted(set(ensemble_report['ids']))

    pack_experiments = json.loads(source_exp.joinpath('experiments.json').read_text())
    # NOTE
    # `experiments.json` is guaranteed to be sorted by model id and to contain
    # all configs (even for models without a report), unlike
    # `report['experiments']`, which only contains stopped models in the
    # order in which they stopped.

    base_config = deepcopy(source_config)
    if is_offline:
        assert not base_config['online_ensembles'][name][
            'include_current_ensemble_in_pool'
        ]
        assert base_config['online_ensembles'][name]['update_type'] != 'latest'

    # Adjust the config for the ensemble evaluation.
    base_config['n_models'] = len(ensemble_unique_ids)
    base_config['configs'] = [
        pack_experiments[x]['config'] for x in ensemble_unique_ids
    ]
    if base_config['optimizer']['type'] in ('AdamWPack', 'MuonAdamWPack'):
        base_config['optimizer']['shared_step'] = True

    # Remove irrelevant config fields.
    for key in list(source_config['online_ensembles'].keys()):
        if key != name:
            del base_config['online_ensembles'][key]
    for key in ('seed', 'sampler'):
        base_config.pop(key, None)

    exp = (
        source_exp.parent
        / ('eval-offline-ensembles' if is_offline else 'eval-online-ensembles')
        / name
        / 'evaluation'
    )
    config = {
        'function': f'{tabpack_model.__name__}.main',
        'n_seeds': n_seeds,
        'base_config': base_config,
    }
    if lib.experiment.get_config_path(exp).exists():
        assert lib.experiment.load_config(exp) == config
    else:
        lib.experiment.create(exp, config=config, parents=True, force=True)
    lib.tools.evaluate.main(exp, force=force, resume=not force)


def main(
    exp_prefix: str | Path,
    *,
    eval_online_ensembles: list[OnlineEnsembleType] = [],
    eval_online_ensembles_n_seeds: None | int = None,
    eval_offline_ensembles: list[OfflineEnsembleType] = [],
    eval_offline_ensembles_n_seeds: None | int = None,
    force: bool = False,
) -> None:
    exp_prefix = Path(exp_prefix)
    main_exp = exp_prefix / 'main'
    assert lib.experiment.is_experiment(main_exp)

    # If the main experiment is not completed, then all secondary experiments
    # are invalid and must be removed.
    if not lib.experiment.is_completed(main_exp) or force:
        for path in exp_prefix.iterdir():
            if path.is_dir() and path.name != main_exp.name:
                _LOGGER.warning(f'Removing {path}')
                shutil.rmtree(path)
            del path

    # Run the main experiment.
    tabpack_model.main(main_exp, force=force)

    # Run secondary experiments.
    for online_ensemble_name in eval_online_ensembles:
        assert eval_online_ensembles_n_seeds is not None
        _evaluate_ensemble(
            main_exp,
            name=online_ensemble_name,
            n_seeds=eval_online_ensembles_n_seeds,
            is_offline=False,
            force=force,
        )

    for offline_ensemble_type in eval_offline_ensembles:
        assert eval_offline_ensembles_n_seeds is not None
        _evaluate_ensemble(
            main_exp,
            name=offline_ensemble_type,
            n_seeds=eval_offline_ensembles_n_seeds,
            is_offline=True,
            force=force,
        )


if __name__ == '__main__':
    lib.env.initialize()

    import cyclopts

    cyclopts.run(main, result_action='return_value')
