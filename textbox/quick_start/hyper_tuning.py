from .experiment import Experiment

from typing import Optional


class HyperTuning:

    def __init__(
            self,
    ):


def run_hyper(
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        config_file_list: Optional[list] = None,
        config_dict: Optional[dict] = None
):
    experiment = Experiment(model, dataset, config_file_list, config_dict)
