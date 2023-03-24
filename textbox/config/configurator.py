import re
import os
import sys
import yaml
import torch
from logging import getLogger
from colorama import init, Fore

init(autoreset=True)

from typing import List, Dict, Optional, Iterable, Any

from textbox.utils.utils import get_local_time
from textbox.utils.argument_list import general_parameters, training_parameters, evaluation_parameters, model_parameters, \
    dataset_parameters


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
        self.file_config_dict = self._load_config_files(config_file_list)
        self.variable_config_dict = self._load_variable_config_dict(config_dict)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()

        self._init_device()
        self.model, self.dataset = self._get_model_and_dataset(model, dataset)
        self._load_internal_config_dict(self.model, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()
        self._set_associated_parameters()

    def _init_parameters_category(self):
        self.parameters: Dict[str, Iterable[str]] = dict()
        self.parameters['General'] = general_parameters
        self.parameters['Training'] = training_parameters
        self.parameters['Evaluation'] = evaluation_parameters
        self.parameters['Model']: List[str] = model_parameters
        self.parameters['Dataset']: List[str] = dataset_parameters

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
        return file_config_dict

    def _load_variable_config_dict(self, config_dict) -> dict:
        # HyperTuning may set the parameters such as mlp_hidden_size in NeuMF in the format of ['[]', '[]']
        # then config_dict will receive a str '[]', but indeed it's a list []
        # temporarily use _convert_config_dict to solve this problem
        if config_dict:
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
            logger = getLogger(__name__)
            logger.warning('command line args [{}] will not be used in TextBox'.format(' '.join(unrecognized_args)))
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)

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
        model_init_file = os.path.join(current_path, '../properties/model/' + model.lower() + '.yaml')
        dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset.lower() + '.yaml')
        if not os.path.exists(dataset_init_file):
            raise ValueError("dataset {} can't be found".format(dataset_init_file))

        self.internal_config_dict = self.overall_config_dict

        self.all_parameters = set()
        for params in self.parameters.values():
            self.all_parameters.update(params)

        if os.path.isfile(model_init_file):
            model_config_dict = self._update_internal_config_dict(model_init_file)
            self.parameters['Model'] += list(set(model_config_dict.keys()) - self.all_parameters)
            self.all_parameters.update(self.parameters['Model'])

        if os.path.isfile(dataset_init_file):
            dataset_config_dict = self._update_internal_config_dict(dataset_init_file)
            self.parameters['Dataset'] += list(set(dataset_config_dict.keys()) - self.all_parameters)
            self.all_parameters.update(self.parameters['Dataset'])

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def _simplify_parameter(self, key: str):
        if key in ['src_lang', 'tgt_lang']:
            return
        if key in self.final_config_dict:
            if isinstance(self.final_config_dict[key], str):
                self.final_config_dict[key] = self.final_config_dict[key].lower()
            elif isinstance(self.final_config_dict[key], list):
                self.final_config_dict[key] = [x.lower() for x in self.final_config_dict[key]]

    def _set_default_parameters(self):
        self.final_config_dict['dataset'] = self.dataset
        self.final_config_dict['model'] = self.model
        self.final_config_dict['model_name'] = self.final_config_dict.get('model_name', self.model.lower())
        self.final_config_dict['data_path'] = os.path.join(self.final_config_dict['data_path'], self.dataset)
        self.final_config_dict['cmd'] = ' '.join(sys.argv)
        self.setdefault(
            'filename', f'{self.final_config_dict["model"]}'
            f'-{self.final_config_dict["dataset"]}'
            f'-{get_local_time()}'
        )  # warning: filename is not replicable
        self.setdefault('saved_dir', 'saved/')
        self.setdefault('_hyper_tuning', [])
        self.setdefault('do_train', True)
        self.setdefault('do_valid', True)
        self.setdefault('do_test', True)
        self.setdefault('valid_strategy', 'epoch')
        self.setdefault('valid_steps', 1)
        self.setdefault('disable_tqdm', False)
        self.setdefault('resume_training', True)
        self.setdefault('wandb', 'online')
        self._simplify_parameter('optimizer')
        self._simplify_parameter('scheduler')
        self._simplify_parameter('src_lang')
        self._simplify_parameter('tgt_lang')
        self._simplify_parameter('task_type')
        self._simplify_parameter('metrics_for_best_model')

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

        # self.external_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def setdefault(self, _key: str, _default: Any):
        """
        Insert key with a value of default if key is not in the final_config_dict dictionary.

        Return the value for key if key is in the final_config_dict dictionary, else default.
        """
        if _key not in self.final_config_dict:
            self.final_config_dict[_key] = _default
            return _default
        else:
            return self.final_config_dict[_key]

    def _set_associated_parameters(self):
        if 'model_path' not in self.final_config_dict:
            self.final_config_dict['load_type'] = 'from_scratch'
        elif os.path.exists(os.path.join(self.final_config_dict['model_path'], 'textbox_configuration.pt')):
            self.final_config_dict['load_type'] = 'resume'
        else:
            self.final_config_dict['load_type'] = 'from_pretrained'

        if self.final_config_dict['model_name'].find('t5') != -1:
            self.final_config_dict['optimizer'] = 'adafactor'
            self.final_config_dict['grad_clip'] = None

        if 'pretrain_task' in self.final_config_dict and self.final_config_dict['pretrain_task'] != 'disabled':
            self.final_config_dict['do_test'] = False
            self.final_config_dict['metrics_for_best_model'] = ['loss']

    def update(self, _m, **kwargs):
        self.final_config_dict.update(_m, **kwargs)

    def __setitem__(self, key, value):
        # for safety reasons, a direct modification of config is not suggested
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
        args_info += '=' * 80 + '\n'
        unrecognized = set(self.final_config_dict.keys()) - self.all_parameters
        if len(unrecognized) > 0:
            self.parameters['Unrecognized'] = unrecognized

        for category in self.parameters:
            if category == 'Unrecognized':
                args_info += Fore.YELLOW
            args_info += '\n# ' + category + ' Hyper Parameters: \n\n'
            for arg in self.parameters[category]:
                if arg.startswith('_'):
                    continue
                if arg in self.final_config_dict:
                    if arg in self.final_config_dict['_hyper_tuning']:
                        args_info += '#[HYPER_TUNING] '
                    args_info += f'{arg}: {self.final_config_dict[arg]}\n'
            if category == 'Unrecognized':
                args_info += Fore.RESET
            args_info += '\n'

        args_info += '=' * 80
        return args_info

    def __repr__(self):
        return str(self.final_config_dict)
