from typing import Optional


def run_textbox(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    config_file_list: Optional[list] = None,
    config_dict: Optional[dict] = None
):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
    """
    from textbox.quick_start.experiment import Experiment
    experiment = Experiment(model, dataset, config_file_list, config_dict)
    experiment.run()
