from copy import copy
from logging import getLogger


from .experiment import Experiment
from ..config.configurator import Config

from hyperopt import Trials, fmin, hp

from typing import Optional, Dict, Any
from hyperopt.pyll import Apply

from ..utils.dashboard import EpochTracker


class AbstractHyperTuning:
    """
    Args:
        model:
        dataset:
        base_config_file_list:
        base_config_dict:
        space_file:
        path_to_output:
        space_dict:
    """

    def __init__(
            self,
            model: Optional[str],
            dataset: Optional[str],
            base_config_file_list: list,
            base_config_dict: dict,
            space_file: Optional[str],
            path_to_output: str,
            space_dict: Optional[dict] = None
    ):

        self.path_to_output = path_to_output
        self.base_config_file_list = base_config_file_list
        self.base_config_dict = base_config_dict
        self.base_config = Config(model, dataset, base_config_file_list, base_config_dict)
        self.metrics_for_best_model = self.base_config['metrics_for_best_model']

        self.space = space_dict or self._build_space_from_file(space_file)
        self.algo = None
        self.max_evals = None

    def fn(self, params: dict):
        raise NotImplementedError

    def run(self):
        fmin(fn=self.fn, space=self.space, algo=self.algo, max_evals=self.max_evals)

    @staticmethod
    def _build_space_from_file(file: Optional[str]) -> Dict[str, Any]:
        space = {}
        assert file is not None, "Configuration of search space must be specified with `params_file`"
        with open(file, 'r', encoding='utf-8') as fp:
            for line in fp:
                para_list = line.strip().split(' ')
                if len(para_list) < 3:
                    raise ValueError(f'Unrecognized parameter: {line}')
                para_name, para_type, para_value = para_list[0], para_list[1], "".join(para_list[2:])

                if para_type == 'choice':
                    para_value = eval(para_value)
                    space[para_name] = hp.choice(para_name, para_value)
                elif para_type == 'uniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = hp.uniform(para_name, float(low), float(high))
                elif para_type == 'quniform':
                    low, high, q = para_value.strip().split(',')
                    space[para_name] = hp.quniform(para_name, float(low), float(high), float(q))
                elif para_type == 'loguniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = hp.loguniform(para_name, float(low), float(high))
                else:
                    raise ValueError('Illegal param type [{}]'.format(para_type))

        return space


class ExhaustiveTuning(AbstractHyperTuning):

    def __init__(
            self,
            model: Optional[str],
            dataset: Optional[str],
            base_config_file_list: list,
            base_config_dict: dict,
            space_file: Optional[str],
            path_to_output: str,
            space_dict: Optional[dict] = None
    ):
        super().__init__(model, dataset, base_config_file_list, base_config_dict, space_file, path_to_output, space_dict)

    def fn(self, params: dict) -> dict:
        """
        Args:
            params: Hyper-parameters to be tuned.
        """
        copied = copy(params)
        copied.update(self.base_config_dict)
        experiment = Experiment(config_file_list=self.base_config_file_list, config_dict=copied)
        valid_result, test_result = experiment.run()
        et = EpochTracker('hyper_tuning', -1, self.metrics_for_best_model)
        print(valid_result, test_result, et.calc_score())
        return {'loss': valid_result['score']}


def run_hyper(
        model: str,
        dataset: str,
        base_config_file_list: list,
        base_config_dict: dict,
        space_file: str,
        path_to_output: str,
):
    hyper_tuning = ExhaustiveTuning(
        model=model,
        dataset=dataset,
        base_config_file_list=base_config_file_list,
        base_config_dict=base_config_dict,
        space_file=space_file,
        path_to_output=path_to_output,
    )
    hyper_tuning.run()
