import logging
import math
from collections import defaultdict
from logging import getLogger
from time import time

import numpy as np
from tqdm import tqdm

from .experiment import Experiment
from ..utils.dashboard import EpochTracker


def run_multi_seed(
    multi_seed: int,
    model: str,
    dataset: str,
    base_config_file_list: list,
    base_config_dict: dict,
):
    experiment = Experiment(model, dataset, base_config_file_list, base_config_dict)
    config = experiment.get_config()
    rng = np.random.default_rng(config['seed'])
    base_config_dict.update({'do_test': False, 'disable_tqdm': True, 'max_save': 0})

    logger = getLogger('multi_seed')
    getLogger('textbox').setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    avg_results = defaultdict(float)
    best_trial = -1
    best_score = -math.inf

    logger.info('======Multiple Random Seeds Test Start======')
    trial_tqdm = tqdm(range(multi_seed), unit='trial', desc="multi_seed", dynamic_ncols=True)
    for trial_idx in trial_tqdm:
        st_time = time()
        trial_seed = rng.integers(int(1e9))
        trial_tqdm.set_postfix(seed=str(trial_seed))
        extend_config = {'seed': trial_seed}
        extend_config.update(base_config_dict)
        valid_result, _ = experiment.run(extend_config)

        if 'generated_corpus' in valid_result:
            del valid_result['generated_corpus']
        if valid_result['score'] > best_score:
            best_trial = trial_idx
            best_score = valid_result['score']
        ed_time = time()

        et = EpochTracker(metrics_results=valid_result)
        et._update_metrics(seed=trial_seed)
        et.epoch_info(desc='Trial', serial=trial_idx, time_duration=ed_time - st_time, source=logger.info)
        for key, value in valid_result.items():
            avg_results[key] *= trial_idx / (trial_idx + 1)
            avg_results[key] += value / (trial_idx + 1)

    logger.info(
        f'======Multiple Random Seeds Test Finished. Best at trial {best_trial} '
        f'(score = {best_score:4f}).======'
    )
    logger.info(f'Average results:')
    for key, value in avg_results.items():
        logger.info(f' {key}: {value}')
