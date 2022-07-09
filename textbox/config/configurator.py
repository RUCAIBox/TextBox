import re
import os
import sys
import yaml
import torch
from logging import getLogger

from typing import List, Dict, Optional

from textbox.utils.utils import get_local_time
from textbox.utils.argument_list import general_arguments, training_arguments, evaluation_arguments


class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in TextBox and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '\-\-learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file (model > dataset > overall)

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str): the model name, default is None, if it is None, config will search the parameter 'model'
            from the external input as the model name.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()
        self._load_overall_config()
        self.external_sources = list()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.variable_config_dict = self._load_variable_config_dict(config_dict)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()

        self._init_device()
        self.internal_sources = list()
        self.model, self.dataset = self._get_model_and_dataset(model, dataset)
        self._load_internal_config_dict(self.model, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()

    def _init_parameters_category(self):
        self.parameters: Dict[str, List[str]] = dict()
        self.parameters['General'] = general_arguments
        self.parameters['Training'] = training_arguments
        self.parameters['Evaluation'] = evaluation_arguments
        self.parameters['Model']: List[str] = ['model', 'model_name']
        self.parameters['Dataset']: List[str] = []

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def _convert_config_dict(self, config_dict: dict) -> dict:
        r"""This function convert the str parameters to their original type.

        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if not isinstance(value, (str, int, float, list, tuple, dict, bool)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_overall_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, '../properties/overall.yaml')

        if os.path.isfile(overall_init_file):
            with open(overall_init_file, 'r', encoding='utf-8') as f:
                self.overall_config_dict = yaml.load(f.read(), Loader=self.yaml_loader)

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
                self.external_sources.append(os.path.abspath(file))
        return file_config_dict

    def _load_variable_config_dict(self, config_dict) -> dict:
        # HyperTuning may set the parameters such as mlp_hidden_size in NeuMF in the format of ['[]', '[]']
        # then config_dict will receive a str '[]', but indeed it's a list []
        # temporarily use _convert_config_dict to solve this problem
        if config_dict:
            self.external_sources.append('variables')
            return self._convert_config_dict(config_dict)
        else:
            return dict()

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning('command line args [{}] will not be used in TextBox'.format(' '.join(unrecognized_args)))
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)

        if len(cmd_config_dict) > 0:
            self.external_sources.append('cmd')
        return cmd_config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model: Optional[str], dataset: Optional[str]):
        if model is None:
            if 'model' not in self.external_config_dict:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
            final_model = self.external_config_dict['model']
        else:
            final_model = model

        if dataset is None:
            if 'dataset' not in self.external_config_dict:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
            final_dataset = self.external_config_dict['dataset']
        else:
            final_dataset = dataset

        return final_model, final_dataset

    def _update_internal_config_dict(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            if config_dict is not None:
                self.internal_config_dict.update(config_dict)
        return config_dict

    def _load_internal_config_dict(self, model, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_init_file = os.path.join(current_path, '../properties/model/' + model + '.yaml')
        dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset + '.yaml')
        if not os.path.exists(dataset_init_file):
            raise ValueError("dataset {} can't be found".format(dataset_init_file))

        self.internal_config_dict = self.overall_config_dict

        self.all_parameters = set()
        for params in self.parameters.values():
            self.all_parameters.update(params)

        if os.path.isfile(model_init_file):
            model_config_dict = self._update_internal_config_dict(model_init_file)
            print(model_config_dict)
            self.parameters['Model'] += list(set(model_config_dict.keys()) - self.all_parameters)
            self.all_parameters.update(self.parameters['Model'])
            self.internal_sources.append(os.path.abspath(model_init_file))

        if os.path.isfile(dataset_init_file):
            dataset_config_dict = self._update_internal_config_dict(dataset_init_file)
            self.parameters['Dataset'] += list(set(dataset_config_dict.keys()) - self.all_parameters)
            self.all_parameters.update(self.parameters['Dataset'])
            self.internal_sources.append(os.path.abspath(dataset_init_file))

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def _simplify_parameter(self, key: str):
        if isinstance(self.final_config_dict[key], str):
            self.final_config_dict[key] = self.file_config_dict[key].lower()

    def _set_default_parameters(self):
        self.final_config_dict['dataset'] = self.dataset
        self.final_config_dict['model'] = self.model
        self.final_config_dict['model_name'] = self.model.lower()
        self.final_config_dict['data_path'] = os.path.join(self.final_config_dict['data_path'], self.dataset)
        self.final_config_dict['filename'] = '{}-{}-{}'.format(
            self.final_config_dict['model'], self.final_config_dict['dataset'], get_local_time()
        )
        self._simplify_parameter('optimizer')
        self._simplify_parameter('scheduler')
        self._simplify_parameter('src_lang')
        self._simplify_parameter('tgt_lang')
        self._simplify_parameter('task_type')

    def _init_device(self):
        if 'use_gpu' not in self.external_config_dict:
            use_gpu = self.overall_config_dict['use_gpu']
        else:
            use_gpu = self.external_config_dict['use_gpu']

        if use_gpu:
            if 'gpu_id' not in self.external_config_dict:
                gpu_id = self.overall_config_dict['gpu_id']
            else:
                gpu_id = self.external_config_dict['gpu_id']

            if type(gpu_id) == tuple:
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_id)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if 'DDP' not in self.external_config_dict:
            use_DDP = self.overall_config_dict['DDP']
        else:
            use_DDP = self.external_config_dict['DDP']

        if use_DDP:
            torch.distributed.init_process_group(backend="nccl")

        self.external_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = f'{len(self.final_config_dict)} parameters found.\n'
        args_info += 'external_config_source: ' + ', '.join(self.external_sources) + '\n'
        args_info += 'internal_config_source: ' + ', '.join(self.internal_sources) + '\n'
        args_info += '=' * 80 + '\n'
        for category in self.parameters:
            args_info += category + ' Hyper Parameters: \n'
            args_info += '\n'.join([
                f'    {arg} = {self.final_config_dict[arg]}'
                for arg in self.parameters[category] if arg in self.final_config_dict
            ])
            args_info += '\n\n'

        unrecognized = set(self.final_config_dict.keys()) - self.all_parameters
        if len(unrecognized) > 0:
            args_info += 'Unrecognized Parameters: \n    '
            args_info += '\n    '.join(unrecognized)
            args_info += '\n'

        args_info += '=' * 80 + '\n'
        return args_info

    def __repr__(self):
        return self.__str__()
