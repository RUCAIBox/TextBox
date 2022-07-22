from .experiment import Experiment

from typing import Optional

from ..config.configurator import Config
from hyperopt import Trials, fmin


class HyperTuning:

    def __init__(
            self,
            config: Config,
    ):
        self.fn = None
        self.space = None
        self.algo = None
        self.max_evals = None
        self.experiment = None

    def fn(self, params):


    def run(self):
        fmin(fn=self.fn, space=self.space, algo=self.algo, max_evals=self.max_evals)
        experiment = Experiment()
        return {'loss'}

def run_hyper(
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        config_file_list: Optional[list] = None,
        config_dict: Optional[dict] = None
):
    experiment = Experiment(model, dataset, config_file_list, config_dict)
    hp = HyperTuning()
