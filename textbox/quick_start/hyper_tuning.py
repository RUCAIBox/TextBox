import logging
import math
from copy import copy
from logging import getLogger
from time import time
from typing import Optional, Dict, Any, Callable, Union, Iterable, Iterator

import hyperopt
import numpy as np
from hyperopt import fmin, hp, pyll
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import Apply

from .experiment import Experiment
from ..utils.dashboard import EpochTracker

SpaceType = Union[Apply, Iterable, dict]


class HyperTuning:
    """
    Hyperparameters optimizing.

    You can inherit and re-implement `fn()` to modify the process.

    Args:
        model: The name of model.
        dataset: The name of dataset.
        base_config_file_list: A list of filenames of base configuration files
        base_config_dict: A list of dictionary of base configuration
        space: The search space.
        algo: The algorithm to be adapted.
    """

    def __init__(
        self,
        model: Optional[str],
        dataset: Optional[str],
        base_config_file_list: list,
        base_config_dict: dict,
        space: Union[SpaceType, str],
        algo: Union[Callable, str],
    ):

        if isinstance(space, dict):
            self.space = space
        elif isinstance(space, str):
            self.space = self._build_space_from_file(space)
        else:
            raise ValueError(f'Unrecognized search space configuration {space}.')

        if isinstance(algo, str):
            algo = algo.split('.')[-1]
            self.algo = getattr(self, algo) if hasattr(self, algo) else getattr(hyperopt.tpe, algo)
        elif callable(algo):
            self.algo = algo
        else:
            raise ValueError(f'Unrecognized algorithm configuration {space}.')

        self.max_evals = _space_size(self.space)

        self.base_config_kwargs = dict(
            model=model,
            dataset=dataset,
            config_file_list=base_config_file_list,
        )
        self.base_config_dict = base_config_dict
        self.base_config_dict['_hyper_tuning'] = list(self.space.keys())
        self.base_config_dict['disable_tqdm'] = True
        self.base_config_dict.setdefault('max_save', 1)
        self.experiment = Experiment(model, dataset, base_config_file_list, base_config_dict)
        self.config = self.experiment.get_config()
        self.metrics_for_best_model = self.config['metrics_for_best_model']
        getLogger('textbox').setLevel(logging.WARNING)
        self.logger = getLogger('hyper_tuning')
        self.logger.disabled = False
        self._trial_count = 0
        self.best_trial = -1
        self.best_score = -math.inf
        self.best_params = None

    def fn(self, params: dict) -> dict:
        """
        Args:
            params: Hyper-parameters to be tuned.
        """
        st_time = time()
        extended_config = copy(params)
        self.logger.info(f"Optimizing parameters: {params}")

        _, test_result = self.experiment.run(extended_config)
        if not isinstance(test_result, dict):
            return {'status': hyperopt.STATUS_FAIL, 'loss': None}
        ed_time = time()

        current_best = False
        if test_result['score'] > self.best_score:
            self.best_score = test_result['score']
            self.best_trial = self._trial_count
            self.best_params = copy(params)
            current_best = True

        et = EpochTracker(self.metrics_for_best_model, metrics_results=test_result)
        et.epoch_info(
            desc='Trial',
            serial=self._trial_count,
            time_duration=ed_time - st_time,
            current_best=current_best,
            source=self.logger.info
        )

        test_result['loss'] = test_result['score']
        test_result['status'] = hyperopt.STATUS_OK
        del test_result['score']
        self._trial_count += 1
        return test_result

    def run(self):
        self.logger.info('======Hyper Tuning Start======')
        fmin(fn=self.fn, space=self.space, algo=self.algo, max_evals=self.max_evals)
        self.logger.info(
            f'======Hyper Tuning Finished. Best at {self.best_trial}'
            f' trial (score = {self.best_score:4f}).======'
        )
        self.logger.info(f'Best params: {self.best_params}')

    @staticmethod
    def _build_space_from_file(file: Optional[str]) -> Dict[str, Any]:
        space = {}
        assert file is not None, "Configuration of search space must be specified with `params_file`"
        with open(file, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if line.startswith('#'):
                    continue
                para_list = line.split()
                if len(para_list) < 3:
                    continue
                para_name, para_type, para_value = para_list[0], para_list[1], "".join(para_list[2:])

                para_value = eval(para_value)
                if para_type == 'choice':
                    space[para_name] = hp.choice(para_name, para_value)
                else:
                    para_type = getattr(hp, para_type)
                    space[para_name] = para_type(para_name, *para_value)

        return space

    @staticmethod
    def exhaustive(new_ids, domain, trials, seed, nb_max_successive_failures=1000):
        r""" This is for exhaustive search in HyperTuning.

        """
        # Build a hash set for previous trials
        hashset = set([
            hash(
                frozenset([(key, value[0]) if len(value) > 0 else (key, None)
                           for key, value in trial['misc']['vals'].items()])
            ) for trial in trials.trials
        ])

        rng = np.random.default_rng(seed)
        r_val = []
        for _, new_id in enumerate(new_ids):
            new_sample = False
            nb_successive_failures = 0
            new_result = None
            new_misc = None
            while not new_sample:
                # -- sample new specs, indices, values
                indices, values = pyll.rec_eval(
                    domain.s_idxs_vals, memo={
                        domain.s_new_ids: [new_id],
                        domain.s_rng: rng,
                    }
                )
                new_result = domain.new_result()
                new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
                miscs_update_idxs_vals([new_misc], indices, values)

                # Compare with previous hashes
                h = hash(
                    frozenset([(key, value[0]) if len(value) > 0 else (key, None) for key, value in values.items()])
                )
                if h not in hashset:
                    new_sample = True
                else:
                    # Duplicated sample, ignore
                    nb_successive_failures += 1

                if nb_successive_failures > nb_max_successive_failures:
                    # No more samples to produce
                    return []

            r_val.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
        return r_val


def run_hyper(
    algo: str,
    model: str,
    dataset: str,
    base_config_file_list: list,
    base_config_dict: dict,
    space: str,
):
    hyper_tuning = HyperTuning(
        model=model,
        dataset=dataset,
        base_config_file_list=base_config_file_list,
        base_config_dict=base_config_dict,
        space=space,
        algo=algo,
    )
    hyper_tuning.run()


def _find_all_nodes(root: SpaceType, node_type: str = 'switch') -> Iterator[Apply]:
    if isinstance(root, (list, tuple)):
        for node in root:
            yield from _find_all_nodes(node, node_type)
    elif isinstance(root, dict):
        for node in root.values():
            yield from _find_all_nodes(node, node_type)
    elif isinstance(root, Apply):
        if root.name == node_type:
            yield root
        for node in root.pos_args:
            if node.name == node_type:
                yield node
        for _, node in root.named_args:
            if node.name == node_type:
                yield node


def _space_size(space: SpaceType) -> int:
    num = 1
    for node in _find_all_nodes(space, 'switch'):
        assert node.pos_args[0].name == 'hyperopt_param'
        num *= len(node.pos_args) - 1
    return num
