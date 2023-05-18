import collections
import os
from accelerate.logging import get_logger
from typing import Optional, Union, List, Dict
import math

import torch
import torch.optim as optim
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from textbox import Config
from textbox.utils.dashboard import get_dashboard, Timestamp, EpochTracker
from .scheduler import (
    AbstractScheduler, InverseSquareRootScheduler, CosineScheduler, LinearScheduler, ConstantScheduler
)
from torch.utils.data import DataLoader
from ..evaluator import BaseEvaluator
from ..model.abstract_model import AbstractModel
from ..utils import serialized_save, init_seed
from ..utils.enum_type import PLM_MODELS, RNN_MODELS


class AbstractTrainer:
    r"""Trainer Class is used to manage the training and evaluation processes of text generation system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config: Config, model: AbstractModel):
        self.config = config
        self.model = model
        self.logger = get_logger(__name__)

    def fit(self, train_data: DataLoader):
        r"""Train the model based on the train data.
        """

        raise NotImplementedError('Method `fit()` should be implemented.')

    def evaluate(self, eval_data: DataLoader):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method `evaluate()` should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in text generation systems.

    This class defines common functions for training and evaluation processes of most text generation system models,
    including `fit()`, `evaluate()`, `resume_checkpoint()` and some other features helpful for model training and
    evaluation.

    Generally speaking, this class can serve most text generation system models, If the training process of the model
    is to simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters' information
    for controlling training and evaluation, such as `learning_rate`, `epochs` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.
    """

    def __init__(self, config: Config, model: AbstractModel, accelerator: Accelerator):
        super(Trainer, self).__init__(config, model)
        self.device: torch.device = config['device']
        self.filename = config['filename']
        self.post_processing = config['post_processing']
        self.accelerator = accelerator

        # Optimization strategy
        self.learning_rate = config['learning_rate']
        self.optimizer_kwargs = {'lr': config['learning_rate']}
        self.optimizer_kwargs.update(config['optimizer_kwargs'])
        self.adafactor_kwargs = config['adafactor_kwargs']
        self.scheduler_kwargs = config['scheduler_kwargs']
        self.grad_clip = config['grad_clip']
        self._trainable_parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = self._build_optimizer(config['optimizer'], config['scheduler'])
        self.accumulation_steps = config['accumulation_steps']

        # Training strategy
        self.quick_test = bool(config['quick_test'])
        self.max_steps = config['max_steps'] or 10000000000  # max training batch step
        self.start_epoch = 1
        r"""Start epoch index. That is, `epoch_idx` iterates through `range(self.start_epoch, self.epochs)`"""
        self.epochs = config['epochs'] if self.max_steps == 10000000000 else 10000000000
        r"""End epoch index + 1, aka max iteration times. That is, `epoch_idx` iterates through 
        `range(self.start_epoch, self.epochs)`"""

        self.valid_steps = self.config['valid_steps']
        self.valid_strategy = self.config['valid_strategy']
        self._valid_count = 0
        self.train_loss_list: List[float] = list()
        self.valid_result_dict: Dict[int, EpochTracker] = dict()
        self.stopping_steps = config['stopping_steps']
        self.stopped = False
        self.stopping_count = 0

        # Evaluation strategy
        self.metrics_for_best_model = set(self.config["metrics_for_best_model"])
        self.evaluator = BaseEvaluator(config, self.config["metrics"])

        # Functionality
        self.saved_dir = os.path.join(config['saved_dir'], self.filename)
        self.saved_model_filename = os.path.join(self.saved_dir, 'checkpoint')
        self.saved_text_filename: str = os.path.join(self.saved_dir, 'generation.txt')

        self.max_save = config['max_save'] if config['max_save'] is not None else 2
        if self.max_save == 0:
            # The saved checkpoint will be deleted at the end of experiment
            self.logger.warning('max_save has been set to 0. None of the checkpoint will be saved.')
            self.max_save = 1
        self.disable_tqdm = config['disable_tqdm'] or not self.accelerator.is_local_main_process
        self._summary_tracker = get_dashboard()

    def _build_optimizer(self, optimizer: str, scheduler: Optional[str])\
            -> Union[optim.Optimizer, AbstractScheduler]:
        """Init the optimizer and scheduler.

        Returns:
            Union[optim.Optimizer, AbstractScheduler]: the optimizer
        """

        optimizer_class = collections.defaultdict(
            lambda: optim.AdamW, {
                'adam': optim.Adam,
                'adamw': optim.AdamW,
                'sgd': optim.SGD,
                'adagrad': optim.Adagrad,
                'rmsprop': optim.RMSprop,
                'adafactor': transformers.Adafactor,
            }
        )
        scheduler_class = {
            'inverse': InverseSquareRootScheduler,
            'cosine': CosineScheduler,
            'linear': LinearScheduler,
            'constant': ConstantScheduler,
        }

        # dealing with adafactor
        if optimizer == 'adafactor':
            # using adafactor_kwargs in overall.yaml
            if self.grad_clip is not None:
                self.grad_clip = None
                self.logger.warning(
                    "Additional optimizer operations like gradient clipping "
                    "should not be used alongside Adafactor."
                )
            self.optimizer_kwargs.update(self.adafactor_kwargs)

        # get optimizer (use default value of pytorch if self.optimizer_kwargs is empty)
        self.logger.debug(f'Using optimizer {optimizer}')
        optimizer = optimizer_class[optimizer](params=self._trainable_parameters, **self.optimizer_kwargs)

        # scheduling
        if scheduler is not None and scheduler in scheduler_class:
            assert isinstance(self.scheduler_kwargs, dict), "Please specify scheduler_kwargs"
            self.logger.debug(f'Using scheduler {scheduler}.')
            self.scheduler_kwargs.setdefault("max_lr", self.learning_rate)
            optimizer = scheduler_class[scheduler](base_optimizer=optimizer, **self.scheduler_kwargs)

        return optimizer

    @property
    def timestamp(self) -> Timestamp:
        """Return the timestamp for the moment."""
        return self._summary_tracker.axes

    @property
    def best_valid_result(self) -> EpochTracker:
        """Retrieve best result dict from `self.valid_result_list`."""
        return self.valid_result_dict[self.best_valid_timestamp.valid_epoch]

    @property
    def best_valid_timestamp(self) -> Timestamp:
        """Retrieve timestamp of best valid result."""
        return self._summary_tracker.best_valid_timestamp

    def _train_epoch(
        self,
        train_data: DataLoader,
        epoch_idx: int,
        valid_data: Optional[DataLoader] = None,
    ) -> dict:
        r"""Train the model in an epoch

        Args:
            train_data:
            epoch_idx: the current epoch index.
            valid_data: Optional (default = None) the dataloader of validation set

        Returns:
            dict: Training losses.
        """
        self.model.train()
        if not self.disable_tqdm:
            train_data_len = math.ceil(len(train_data) / self.accumulation_steps)
            train_tqdm = tqdm(
                range(train_data_len),
                desc=f"train {epoch_idx:4}",
                dynamic_ncols=True,
                postfix={'loss': None},
                unit='step'
            )

        with self._summary_tracker.new_epoch('train'):
            for step, data in enumerate(train_data):
                with self.accelerator.accumulate(self.model):
                    if step % self.accumulation_steps == 0:
                        self._summary_tracker.new_step()
                        if self.timestamp.train_step == self.max_steps + 1:
                            self.stopped = True
                            break

                    self.optimizer.zero_grad()
                    loss = self.model(data, epoch_idx=epoch_idx)
                    # avg_loss = self.accelerator.gather(loss).mean().item()
                    avg_loss = loss.item()
                    self._summary_tracker.append_loss(avg_loss)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients and self.grad_clip is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                    if self.accelerator.sync_gradients:
                        if not self.disable_tqdm:
                            train_tqdm.update(1)
                            train_tqdm.set_postfix(loss=self._summary_tracker.epoch_loss)
                        if valid_data:
                            self.stopped |= self._valid(valid_data, 'step')
                            self.accelerator.gradient_state._set_end_of_dataloader(False)
                        if self.stopped:
                            break

            if not self.disable_tqdm:
                train_tqdm.close()
            return self._summary_tracker.epoch_dict()

    @torch.no_grad()
    def _valid(
        self,
        valid_data: DataLoader,
        valid_mode: str,
    ) -> bool:
        """Validate every `self.eval_interval` step or epoch if evaluation strategy matches attribute
        `self.eval_strategy`. Specifically, if `self.eval_interval` is set to `0`, validation will be skipped.

        Early stopping will also be checked if `self.stopping_steps` is positive integer.

        Args:
            valid_data: The dataloader of validation set.
            valid_mode: The evaluation strategy of current call ("epoch" or "step").

        Returns:
            bool: Early stopping. Return true if `self.stopping_steps` is positive integer and `self._early_stopping()`
            is True.
        """
        if (self.valid_steps <= 0) or (valid_mode != self.valid_strategy):
            return False

        self._valid_count += 1
        if self._valid_count % self.valid_steps != 0:
            return False
        self.temp_mode = self._summary_tracker._current_mode
        self.temp_epoch = self._summary_tracker._current_epoch
        state = torch.default_generator.get_state()
        with self._summary_tracker.new_epoch('valid'):
            if 'loss' in self.metrics_for_best_model:
                self.model.eval()
                if not self.disable_tqdm:
                    valid_tqdm = tqdm(
                        valid_data,
                        desc=f"valid {self.timestamp.valid_epoch:4}",
                        dynamic_ncols=True,
                        postfix={'loss': None},
                        unit='step'
                    )
                else:
                    valid_tqdm = valid_data

                for data in valid_tqdm:
                    self._summary_tracker.new_step()
                    loss = self.model(data)
                    avg_loss = self.accelerator.gather(loss).mean().item()
                    self._summary_tracker.append_loss(avg_loss)
                    if not self.disable_tqdm:
                        valid_tqdm.set_postfix(loss=self._summary_tracker.epoch_loss)
                valid_results = self._summary_tracker.epoch_dict()
            else:
                valid_results = self.evaluate(valid_data, is_valid=True)
            self._summary_tracker.set_metrics_results(valid_results)
            self.valid_result_dict[self.timestamp.valid_epoch] = self._summary_tracker._current_epoch
        self._summary_tracker._current_mode = self.temp_mode
        self._summary_tracker._current_epoch = self.temp_epoch
        self.model.train()

        stopped = bool(self.stopping_steps) and self._early_stopping(self._summary_tracker.is_best_valid)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            # os.system('sleep 10s')
            self.save_checkpoint()
        torch.default_generator.set_state(state)
        return stopped

    def _early_stopping(self, current_best: bool) -> bool:
        r""" Check early stopping with `stopping_steps`, a maximum amount of non-best validation.

        Args:
            current_best: Whether current epoch is the one with the best score.

        Return:
            bool: If true, the training process will be stopped, else the `self.stopping_count` will accumulate.
        """

        stop_flag = False

        if current_best:
            self.stopping_count = 0
        else:
            self.stopping_count += 1
            stop_flag = self.stopping_count > self.stopping_steps

        return stop_flag

    def _get_checkpoint(self) -> Optional[dict]:
        if len(self.valid_result_dict) == 0:
            self.logger.warning('Get checkpoint failed. No validation has been performed.')
            return None

        # get optimizer, config and validation summary
        checkpoint = {
            # parameters that needed to be loaded
            'stopping_count': self.stopping_count,
            'best_valid_score': self._summary_tracker.best_valid_score,
            'epoch': self.timestamp.train_epoch,
            'timestamp': self.timestamp,
            'config': self.config,
            # parameters for recording only
            'summary': self.valid_result_dict[self.timestamp.valid_epoch],
        }
        self.logger.debug(checkpoint)
        return checkpoint

    def save_checkpoint(self):
        serial_idx = self.timestamp.valid_epoch
        serial_of_soft_link = self.best_valid_timestamp.valid_epoch
        serialized_save(
            self.accelerator.unwrap_model(self.model),
            self.optimizer,
            self._get_checkpoint(),
            serial=serial_idx,
            serial_of_soft_link=serial_of_soft_link,
            path_without_extension=self.saved_model_filename,
            tag='epoch',
            extension_name='pth',
            max_save=self.max_save,
        )

    def save_generated_text(self, generated_corpus: List[str], is_valid: bool = False):
        r"""Store the generated text by our model into `self.saved_text_filename`."""
        # path_to_save = self.saved_model_filename + '_epoch-' + str(self.timestamp.valid_epoch)
        # saved_text_filename = os.path.join(path_to_save, 'generation.txt')
        # os.makedirs(path_to_save, exist_ok=True)
        if not is_valid:
            saved_text_filename = self.saved_text_filename
            with open(saved_text_filename, 'w') as fout:
                for text in generated_corpus:
                    fout.write(text + '\n')

    def resume_checkpoint(self, resume_dir: str):
        r"""Load training information.

        Args:
            resume_dir: the checkpoint file (specific by `model_path`).
        """
        # check
        self.logger.info("Resuming checkpoint from {}...".format(resume_dir))
        resume_checkpoint = os.path.join(resume_dir, 'textbox_configuration.pt')
        resume_optimizer = os.path.join(resume_dir, 'optimizer.pt')
        if os.path.isfile(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
        else:
            self.logger.warning('Checkpoint file "{}" not found. Resuming stopped.'.format(resume_dir))
            return
        if checkpoint['config']['optimizer'].lower() != self.config['optimizer']:
            self.logger.warning(
                'Optimizer configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        if os.path.isfile(resume_optimizer):
            optim_state_dict = torch.load(resume_optimizer, map_location=self.device)
            self.optimizer.load_state_dict(optim_state_dict)
        else:
            self.logger.warning('Checkpoint file "{}" not found. Resuming stopped.'.format(resume_dir))
            return

        # load start epoch and early stopping
        self.start_epoch = checkpoint['epoch'] + 1  # start from the next step
        self._summary_tracker.axes = checkpoint['timestamp']
        self.stopping_count = checkpoint['stopping_count']
        self._summary_tracker.best_valid_score = checkpoint['best_valid_score']
        self._summary_tracker.best_valid_timestamp = self._summary_tracker.axes
        self.valid_result_dict = {self._summary_tracker.axes.valid_epoch: checkpoint['summary']}

        if checkpoint['config']['seed']:
            init_seed(checkpoint['config']['seed'], checkpoint['config']['reproducibility'])
            set_seed(checkpoint['config']['seed'])

        # load architecture params from checkpoint
        if checkpoint['config']['model_name'] != self.config['model_name']:
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )

        # load optimizer state from checkpoint only when optimizer type is not changed

        self.logger.info(
            'Checkpoint loaded. Resume training from epoch {} steps {}'.format(
                self.start_epoch, self._summary_tracker.axes.train_step + 1
            )
        )

    def fit(
        self,
        train_data: DataLoader,
        valid_data: Optional[DataLoader] = None,
    ) -> dict:
        r"""Train the model based on the train data.

        Args:
            train_data: The dataloader of training set.
            valid_data: (default = None) The dataloader of training set.

        Returns:
             dict: the best valid score and best valid result.
        """

        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)

        self.logger.info("====== Start training ======")
        self.accelerator.wait_for_everyone()
        for epoch_idx in range(self.start_epoch, self.epochs + 1):
            # train
            loss = self._train_epoch(train_data, epoch_idx, valid_data)['loss']
            self.train_loss_list.append(loss)

            # valid
            if valid_data:
                self.stopped |= self._valid(valid_data, 'epoch')

            if self.stopped:
                if self.stopping_steps:
                    self.logger.info(f'Early stopped at {self.stopping_count} non-best validation.')
                elif self.max_steps < 10000000000:
                    self.logger.info(f'Stopped at max_steps {self.max_steps}.')
                break

        file = self.saved_model_filename + '_best'
        if os.path.exists(file):
            self.logger.info(f'Soft link created: {file} -> {os.readlink(file)}')
        self.logger.info(
            f'====== Finished training, best validation result '
            f'at train epoch {self.best_valid_timestamp.train_epoch} ======'
        )

        self.model = self.accelerator.unwrap_model(self.model)
        self.logger.info('Best valid result: {}'.format(self.best_valid_result.as_str()))
        return self.best_valid_result.as_dict()

    @torch.no_grad()
    def evaluate(
        self,
        eval_data: DataLoader,
        load_best_model: bool = True,
        is_valid: bool = False,
    ) -> Optional[dict]:
        r"""Evaluate the model based on the `eval_data`.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            is_valid: (default = False) True if evaluate during validation

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if is_valid:
            load_best_model = False

        if load_best_model:
            checkpoint_dir = self.saved_model_filename + '_best'
            self.logger.info('Loading model structure and parameters from {} ...'.format(checkpoint_dir))
            self.accelerator.wait_for_everyone()
            unwrap_model = self.accelerator.unwrap_model(self.model)
            unwrap_model.from_pretrained(checkpoint_dir)
            unwrap_model.tokenizer.from_pretrained(checkpoint_dir)

        if not is_valid:
            self.model = self.accelerator.prepare(self.model)

        self.model.eval()

        if self.config['dataset'] == 'multiwoz':
            self.evaluator.evaluators[0].load_data('valid' if is_valid else 'test')
            turn_domains = self.evaluator.evaluators[0].turn_domains

        # generate
        generate_corpus = []
        eval_tqdm = tqdm(eval_data, desc="generating", dynamic_ncols=True) if not self.disable_tqdm else eval_data
        state = torch.default_generator.get_state()
        for i, batch_data in enumerate(eval_tqdm):
            if self.config['dataset'] != 'multiwoz':
                generated = self.accelerator.unwrap_model(self.model).generate(batch_data, self.accelerator)
            else:
                batch_size = batch_data['source_ids'].size(0)
                idx_mask = torch.zeros(batch_size).to(self.device).bool()
                idx_mask[::3] = 1
                bs_batch = {}
                bs_batch['source_ids'] = batch_data['source_ids'][idx_mask]
                bs_batch['source_mask'] = batch_data['source_mask'][idx_mask]
                asrs_batch = {}
                asrs_batch['source_ids'] = batch_data['source_ids'][~idx_mask]
                asrs_batch['source_mask'] = batch_data['source_mask'][~idx_mask]
                bs_outputs = self.accelerator.unwrap_model(self.model).generate(bs_batch, self.accelerator)

                batch_size //= 3
                db_texts = [
                    self.evaluator.evaluators[0].span_db(bs, td)
                    for bs, td in zip(bs_outputs, turn_domains[i * batch_size:(i + 1) * batch_size])
                ]
                db_ids = torch.tensor(self.model.tokenizer.convert_tokens_to_ids(db_texts)).long()
                db_ids = db_ids.repeat_interleave(2).to(self.device)
                db_idx = torch.eq(asrs_batch['source_ids'], self.model.tokenizer.convert_tokens_to_ids('[db_nores]'))
                asrs_batch['source_ids'][db_idx] = db_ids
                asrs_outputs = self.accelerator.unwrap_model(self.model).generate(asrs_batch, self.accelerator)
                generated = sum([[bs, aspn, rs]
                                 for bs, aspn, rs in zip(bs_outputs, asrs_outputs[::2], asrs_outputs[1::2])], [])

            generate_corpus.extend(generated)
        torch.default_generator.set_state(state)
        corpus_len = len(eval_data.dataset.target_text)
        reference_dataset = eval_data.dataset
        generate_corpus = generate_corpus[:corpus_len]

        if self.post_processing == 'paraphrase':
            for i, gen in enumerate(generate_corpus):
                if gen.find('[SEP]') >= 0:
                    gen = gen.split('[SEP]')[1].strip()
                else:
                    last = max(gen.rfind('('), gen.rfind(')'))
                    last = gen.find(' ', last)
                    gen = gen[last:].strip()
                generate_corpus[i] = gen

        if self.accelerator.is_local_main_process:
            self.save_generated_text(generate_corpus, is_valid)

        result = self.evaluator.evaluate(generate_corpus, reference_dataset)

        return result
