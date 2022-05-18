# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2022/05/10, 2021/10/11, 2021/4/12, 2020/12/2, 2020/11/27, 2020/12/3, 2020/12/26
# @Author : Hu Yiwen, Tang Tianyi, Lai Xu, Jinhao Jiang, Xiaoxuan Hu, Tianyi Tang, Jinhao Jiang
# @Email  : huyiwen@ruc.edu.cn, tsui_lai@163.com, jiangjinhao@std.uestc.edu.cn, huxiaoxuan@ruc.edu.cn, steventang@ruc.edu.cn, jiangjinhao@std.uestc.edu.cn

r"""
textbox.trainer.trainer
################################
"""

import logging
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import collections

from tqdm import tqdm

from torch.utils.data import DataLoader
from time import time
from logging import getLogger

from textbox.module.Optimizer.optim import InverseSquareRootOptim, CosineOptim, LinearOptim, ConstantOptim
from textbox.evaluator import BaseEvaluator, evaluator_list, kb2text_evaluator
from textbox.utils import ensure_dir, Timer, TensorboardWriter


class EpochTracker:
    """
    Track validating metrics results and training/validating loss.
    """
    is_train = {True: "training", False: "validating"}

    def __init__(self, config, epoch_idx: int, train: bool = True):
        """
        Args:
            config: config
            mode: Set the metrics tracker in training/validating mode
        """
        self._DDP = config['DDP']
        self._metrics_factors = config['metrics_factors']
        self._writer = TensorboardWriter.get_writer()
        self._logger = getLogger()
        self.epoch_idx = epoch_idx
        self.mode_tag = self.is_train[train]

        self._total_loss = 0.
        self._total_num = 0
        self._loss = None

        self._metrics_results = None
        self._score = None

    def append_loss(self, loss: float):
        _check_nan(loss)
        self._total_loss += loss
        self._total_num += 1
        self._writer.add_scalar("Loss/" + self.mode_tag, loss)

    def set_result(self, results: dict):
        self._metrics_results = results
        for metric, result in self._metrics_results.items():
            self._writer.add_scalar("Metrics/" + metric, result)
        self._metrics_results['loss'] = self.loss

    def _calc_score(self) -> float:
        score = 0.
        for metric, result in self._metrics_results.items():
            score += result * self._metrics_factors[metric]
        self._writer.add_scalar("Metrics/mixed", score)
        return score

    @property
    def score(self):
        """get mixed score by calculating weighted sum of results"""
        if self.mode_tag == self.is_train[True]:
            self._logger.warning('Use "loss" to get training loss.')
            return self.loss

        if not self._score:
            self._score = self._calc_score()
        return self._score

    def _calc_loss(self) -> float:
        """Reduce loss if DDP enabled"""
        _loss = self._total_loss / self._total_num
        if self._DDP:
            _loss = torch.tensor(_loss).to("cuda")
            torch.distributed.all_reduce(_loss, op=torch.distributed.ReduceOp.SUM)
            _loss /= torch.distributed.get_world_size()
            _loss = _loss.item()
        return _loss

    @property
    def loss(self):
        """get total loss (loss will be reduced on the first call)
        Returns:
            float: loss
        """

        if not self._loss:
            self._loss = self._calc_loss()
        return self._loss

    @property
    def perplexity(self):
        return np.exp(self.loss)

    def info(self, time_duration: float, extra_info=""):
        """Output loss with epoch and time information."""
        output = f"Epoch {self.epoch_idx}, {extra_info} {self.mode_tag} "
        output += f"[time: {time_duration:.2f}s, {self.mode_tag}_loss: {self.loss:.4f}"
        if self.mode_tag == self.is_train[False]:
            output += ", {} ppl: {}".format(self.mode_tag, self.perplexity)
        output += "]"
        # flush
        self._logger.info(output)


def _check_nan(loss):
    if torch.isnan(loss):
        raise ValueError('Training loss is nan')


class AbstractTrainer:
    r"""Trainer Class is used to manage the training and evaluation processes of text generation system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()

    def fit(self, train_data):
        r"""Train the model based on the train data.
        """

        raise NotImplementedError('Method fit() should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method evaluate() should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in text generation systems.
    This class defines common functions for training and evaluation processes of most text generation system models,
    including fit(), evalute(), resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most text generation system models, If the training process of the model
    is to simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.
    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.DDP = config['DDP']
        self.filename = config['filename']
        self.learner = config['learner'].lower()
        self.schedule = config['schedule'].lower()
        self.init_lr = config['init_lr']
        self.warmup_steps = config['warmup_steps']
        self.max_steps = config['max_steps']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.embedding_size = config['embedding_size']
        self.warmup_steps = config['warmup_steps']
        self.grad_clip = config['grad_clip']

        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = self.config['filename'] + '.pth'
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.generated_text_dir = config['generated_text_dir']
        ensure_dir(self.generated_text_dir)
        saved_text_file = self.config['filename'] + '.txt'
        self.saved_text_file = os.path.join(self.generated_text_dir, saved_text_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = 100000000
        self.best_valid_result = None
        self.train_loss_list = list()
        self.optimizer = self._build_optimizer()

        self.metrics = self._process_metrics("metrics")
        self.valid_metrics_factors = config['valid_metrics_factors']
        self.valid_metrics = self._process_metrics("valid_metrics")
        self.evaluator = BaseEvaluator(config, self.metrics)

        self.is_logger = (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP
        self.item_tensor = None
        self.tot_item_num = None
        self.iid_field = config['ITEM_ID_FIELD']

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """

        def _get_base_optimizer(name: str):
            if name == 'adam':
                _optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif name == 'sgd':
                _optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)
            elif name == 'adagrad':
                _optim = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
            elif name == 'rmsprop':
                _optim = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
            else:
                self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
                _optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            return _optim

        def _get_schedule(name: str, base_optim):
            assert isinstance(self.init_lr, float), "Specify initial learning rate (init_lr)"
            assert isinstance(self.warmup_steps, int), "Specify warmup steps (warmup_steps)"
            if name == 'inverse':
                _optim = InverseSquareRootOptim(base_optim, self.init_lr, self.learning_rate, self.warmup_steps)
            elif name == 'cosine':
                assert isinstance(self.max_steps, int), "Specify max steps (max_steps)"
                _optim = CosineOptim(base_optim, self.init_lr, self.learning_rate, self.warmup_steps, self.max_steps)
            elif name == 'linear':
                assert isinstance(self.max_steps, int), "Specify max steps (max_steps)"
                _optim = LinearOptim(base_optim, self.init_lr, self.learning_rate, self.warmup_steps, self.max_steps)
            elif name == 'constant':
                _optim = ConstantOptim(base_optim, self.init_lr, self.learning_rate, self.warmup_steps)
            else:
                self.logger.info("Learning rate scheduling disabled.")
                _optim = base_optim
            return _optim

        optimizer = _get_base_optimizer(self.learner)
        optimizer = _get_schedule(self.schedule, optimizer)
        return optimizer

    def _process_metrics(self, metrics_type: str) -> set:
        r"""check the correctness of the setting"""
        metrics = self.config[metrics_type]
        if not isinstance(metrics, (str, list)):
            raise TypeError(f'Evaluator(s) of {metrics_type} must be a string or list.')
        if isinstance(metrics, str):
            if metrics[0] == '[' and metrics[-1] == ']':
                metrics = metrics[1:-1]
            metrics = metrics.split(",")

        metrics = map(lambda x: x.strip().lower(), metrics)

        if metrics_type == 'valid_metrics':
            metrics = list(metrics)
            self.valid_metrics_factors = self.valid_metrics_factors or list()
            if len(self.valid_metrics_factors) < len(metrics):
                self.valid_metrics_factors += [1] * (len(metrics) - len(self.valid_metrics_factors))
                self.logger.warning(
                    '"valid_metrics" got a different size with its factors (valid_metrics_factors). Filled with 1s.'
                )
            self.valid_metrics_factors = dict(zip(metrics, self.valid_metrics_factors))
            if self.valid_metrics_factors.get('loss') and self.valid_metrics_factors['loss'] > 0:
                self.valid_metrics_factors['loss'] *= -1
            self.config['metrics_factors'] = self.valid_metrics_factors

            msg = "Validating with mixed metric: "
            msg += " + ".join([f"({factor}) * {metric}" for metric, factor in self.valid_metrics_factors.items()])
            self.logger.info(msg)

        metrics = set(metrics)

        if len(metrics - evaluator_list) != 0:
            self.logger.warning(
                f"Evaluator(s) of {metrics_type} " + ", ".join(metrics - evaluator_list) +
                " are ignored because not in supported evaluators list (" + ", ".join(evaluator_list) + ")."
            )
            metrics -= metrics - evaluator_list

        return metrics

    def _train_epoch(self, train_data, epoch_idx):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            !
        """
        self.model.train()
        tracker = EpochTracker(self.config, epoch_idx, train=True)
        train_tqdm = tqdm(train_data, desc="training") if self.is_logger else train_data
        for data in train_tqdm:
            self.optimizer.zero_grad()
            loss = self.model(data, epoch_idx=epoch_idx)
            tracker.append_loss(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return tracker

    def _valid_epoch(self, valid_data, epoch_idx):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            !
        """
        self.model.eval()
        tracker = EpochTracker(self.config, epoch_idx, train=False)
        valid_tqdm = tqdm(valid_data, desc="validating") if self.is_logger else valid_data
        for data in valid_tqdm:
            losses = self.model(data)
            tracker.append_loss(losses)

        valid_results = self.evaluate(valid_data, load_best_model=False, metrics=self.valid_metrics)
        tracker.set_result(valid_results)

        return tracker

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        _state_dict = self.model.state_dict()
        if self.DDP and torch.distributed.get_rank() == 0:
            _new_dict = collections.OrderedDict()
            for key, val in _state_dict["state_dict"].items():
                changed_key = key[7:] if key[0:7] == 'module.' else key
                _new_dict[changed_key] = val
            _state_dict = _new_dict

        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': _state_dict,
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, self.saved_model_file)
        self._saved_once = True
        self.logger.info('Saving current: {}'.format(self.saved_model_file))

    def _save_generated_text(self, generated_corpus):
        r"""Store the generated text by our model.

        Args:
            corpus (list of string list):
        """
        with open(self.saved_text_file, 'w') as fout:
            for tokens in generated_corpus:
                fout.write(' '.join(tokens) + '\n')

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.logger.info("Resuming checkpoint from {}...".format(resume_file))
        if os.path.isfile(resume_file):
            checkpoint = torch.load(resume_file)
        else:
            self.logger.warning('Checkpoint file "{}" not found. Resuming stopped.'.format(resume_file))
            return

        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        if self.DDP:
            saved_dict = collections.OrderedDict()
            for state_dict_key, state_dict_val in checkpoint['state_dict'].items():
                changed_key = 'module.' + state_dict_key
                saved_dict[changed_key] = state_dict_val
            checkpoint['state_dict'] = saved_dict
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): the best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        self._saved_once = False
        best_epoch = -1

        self.logger.info("====== Start training ======")
        with TensorboardWriter(self.config):
            for epoch_idx in range(self.start_epoch, self.epochs):
                # train
                with Timer() as train_timer:
                    train_tracker = self._train_epoch(train_data, epoch_idx)
                self.train_loss_list.append(train_tracker.loss)
                train_tracker.info(train_timer.duration)

                # validation
                if self.eval_step <= 0 or not valid_data or (epoch_idx + 1) % self.eval_step != 0:
                    continue

                with torch.no_grad(), Timer() as valid_timer:
                    valid_tracker = self._valid_epoch(valid_data, epoch_idx)
                valid_tracker.info(valid_timer.duration)

                if self.early_stopping(valid_tracker, epoch_idx):
                    best_epoch = epoch_idx - self.cur_step * self.eval_step
                    break

        if not self._saved_once and saved:
            self._save_checkpoint(max(best_epoch, self.epochs - 1))
        self.logger.info('====== Finished training, best eval result in epoch {} ======'.format(best_epoch))
        return self.best_valid_score, self.best_valid_result

    def early_stopping(self, valid_tracker, saved_id):
        r""" validation-based early stopping

        Args:
            valid_tracker (EpochTracker): contain valid information (loss, perplexity and metrics score)
            saved_id (int): epoch id if saved

        Returns:
            - bool: whether to stop
        """

        stop_flag = False
        if self.best_valid_score == 100000000:
            self.best_valid_score = 0

        if valid_tracker.score > self.best_valid_score:
            self.cur_step = 0
            self.best_valid_score = valid_tracker.loss
            self.best_valid_result = valid_tracker.perplexity
            self._save_checkpoint(saved_id)
        else:
            self.cur_step += 1
            stop_flag = self.cur_step > self.stopping_step

        return stop_flag

    def _evaluate_nll_test(self, eval_data):
        r"""Calculate the negative log-likelihood of the eval_data.

        Args:
            eval_data (AbstractDataLoader): the eval data.

        Returns:
            Float: NLL_test of the eval data.
        """

        total_loss = 0
        for epoch_idx, eval_batch in enumerate(eval_data):
            nll_test = self.model.calculate_nll_test(eval_batch, epoch_idx)
            total_loss += float(nll_test)
        return total_loss / len(eval_data)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, _eval=True, metrics=None):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (AbstractDataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            _eval (bool, optional): Whether to evaluate. Default is True. False to preview generation only.
            metrics: force specify metrics

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            if not os.path.isfile(checkpoint_file):
                self.logger.error(f'Failed to evaluate model: "{checkpoint_file}" not found.')
                return
            self.logger.info('Loading model structure and parameters from {} ...'.format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()

        if not _eval:
            generate_sentence = self.model.generate(next(eval_data), eval_data)
            self.logger.info('Generation Preview: ' + ' '.join(generate_sentence[0]))
            return

        generate_corpus = []
        eval_tqdm = tqdm(eval_data, desc="evaluation") if self.is_logger else eval_data
        for batch_data in eval_tqdm:
            generated = self.model.generate(batch_data, eval_data)
            assert len(generated) == len(batch_data['target_text']), "Generated corpus has a mismatched batch size!"
            generate_corpus.extend(generated)
        self._save_generated_text(generate_corpus)

        reference_corpus = eval_data.get_reference()
        result = self.evaluator.evaluate(generate_corpus, reference_corpus, metrics=metrics)
        if "nll_test" in (metrics or self.metrics) and self.config['task_type'].lower() == "unconditional":
            result['nll_test'] = self._evaluate_nll_test(eval_data)

        return result

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = range(1, 1 + len(self.train_loss_list))
        plt.plot(epochs, self.train_loss_list)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(self.filename)
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


class GANTrainer(Trainer):
    r"""GANTrainer is designed for GAN, which is a generative adversarial net method.
    """

    def __init__(self, config, model):
        super(GANTrainer, self).__init__(config, model)

        self.optimizer = None
        self.g_optimizer = self._build_module_optimizer(self.model.generator)
        self.d_optimizer = self._build_module_optimizer(self.model.discriminator)

        self.g_pretraining_epochs = config['g_pretraining_epochs']
        self.d_pretraining_epochs = config['d_pretraining_epochs']
        self.d_sample_num = config['d_sample_num']
        self.d_sample_training_epochs = config['d_sample_training_epochs']
        self.adversarail_training_epochs = config['adversarail_training_epochs']
        self.adversarail_d_epochs = config['adversarail_d_epochs']

        self.g_pretraining_loss_dict = dict()
        self.d_pretraining_loss_dict = dict()
        self.max_length = config['seq_len'] + 2
        self.padding_token_idx = model.padding_token_idx

    def _build_module_optimizer(self, module):
        r"""Init the Module Optimizer

        Args:
            module (torch.nn.Mudule): Mudule class of torch.nn needed optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(module.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(module.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(module.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(module.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(module.parameters(), lr=self.learning_rate)

        return optimizer

    def _optimize_step(self, losses, total_loss, model, opt):
        r"""The opt uses the cliped losses to conduct an optimize step to optimize model
        and sum up losses to the total_loss.

        Args:
            losses (torch.Tensor or tuple): The loss to be backward.
            total_loss (Float): Total loss in an epoch.
            model (torch.nn.Mudule): The model to be optimized.
            opt (torch.optim): The optimizer of the model.

        Returns:
            torch.Tensor or tuple: Total loss in an epoch, shape: [].
        """
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        _check_nan(loss)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        opt.step()
        return total_loss

    def _save_checkpoint(self, epoch):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict()
        }
        torch.save(state, self.saved_model_file)

    def _add_pad(self, data):
        r"""Pad the data to the max length of corpus.

        Args:
            data (torch.Tensor): The data to be padded, shape: [batch_size, max_batch_length].

        Returns:
            torch.Tensor: The padded data, shape: [batch_size, max_length].
        """
        batch_size = data.shape[0]
        padded_data = torch.full((batch_size, self.max_length),
                                 self.padding_token_idx,
                                 dtype=torch.long,
                                 device=self.device)
        padded_data[:, :data.shape[1]] = data
        return padded_data

    def _get_real_data(self, train_data):
        r"""Get the target text index of the corpus train_datas.

        Args:
            train_data (DataLoader): the train data.

        Returns:
            torch.Tensor: The target text index, shape: [batch_size, max_batch_length].
        """
        real_datas = []
        for corpus in train_data:
            real_data = corpus['target_idx']
            real_data = self._add_pad(real_data)
            real_datas.append(real_data)

        real_datas = torch.cat(real_datas, dim=0)
        return real_datas

    def _g_train_epoch(self, train_data, epoch_idx):
        r"""Train the generator module in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.generator.train()
        total_loss = None

        for batch_idx, data in enumerate(train_data):
            losses = self.model.calculate_g_train_loss(data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)
        total_loss = [l / len(train_data)
                      for l in total_loss] if isinstance(total_loss, tuple) else total_loss / len(train_data)
        total_loss = tuple(total_loss) if isinstance(total_loss, list) else total_loss
        return total_loss

    def _d_train_epoch(self, train_data, epoch_idx):
        r"""Train the discriminator module in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.discriminator.train()
        total_loss = None
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        fake_data = self.model.sample(self.d_sample_num)
        fake_dataloader = DataLoader(fake_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)

        for _ in range(self.d_sample_training_epochs):  # d_epoch
            for real_data, fake_data in zip(real_dataloader, fake_dataloader):
                losses = self.model.calculate_d_train_loss(real_data, fake_data, epoch_idx=epoch_idx)
                total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)

        return total_loss / min(len(real_dataloader), len(fake_dataloader)) / self.d_sample_training_epochs

    def _adversarial_train_epoch(self, train_data, epoch_idx):
        r"""Adversarial training in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.generator.train()
        total_loss = None
        losses = self.model.calculate_g_adversarial_loss(epoch_idx=epoch_idx)
        total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)

        for epoch_idx in range(self.adversarail_d_epochs):
            self._d_train_epoch(train_data, epoch_idx=epoch_idx)

        return total_loss

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        # generator pretraining
        if verbose:
            self.logger.info("Start generator pretraining...")
        for epoch_idx in range(self.g_pretraining_epochs):
            training_start_time = time()
            train_loss = self._g_train_epoch(train_data, epoch_idx)
            self.g_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss,
                                                 "generator pre")
            if verbose:
                self.logger.info(train_loss_output)
        if verbose:
            self.logger.info("End generator pretraining...")

        # discriminator pretraining
        if verbose:
            self.logger.info("Start discriminator pretraining...")
        for epoch_idx in range(self.d_pretraining_epochs):
            training_start_time = time()
            train_loss = self._d_train_epoch(train_data, epoch_idx)
            self.d_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss,
                                                 "discriminator pre")
            if verbose:
                self.logger.info(train_loss_output)
        if verbose:
            self.logger.info("End discriminator pretraining...")

        # adversarial training
        if verbose:
            self.logger.info("Start adversarial training...")
        for epoch_idx in range(self.adversarail_training_epochs):
            training_start_time = time()
            train_loss = self._adversarial_train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
        if verbose:
            self.logger.info("End adversarial pretraining...")

        self._save_checkpoint(self.adversarail_training_epochs)
        return -1, None


class TextGANTrainer(GANTrainer):
    r"""TextGANTrainer is designed for TextGAN.
    """

    def __init__(self, config, model):
        super(TextGANTrainer, self).__init__(config, model)
        self.adversarail_g_epochs = config['adversarail_g_epochs']

    def _d_train_epoch(self, train_data, epoch_idx):
        self.model.discriminator.train()
        total_loss = None
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)

        for _ in range(self.d_sample_training_epochs):
            for idx, real_data in enumerate(real_dataloader):
                fake_data, z = self.model.sample()
                losses = self.model.calculate_d_train_loss(real_data, fake_data, z, epoch_idx=epoch_idx)
                total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)
                if (idx * self.model.batch_size >= self.d_sample_num):
                    break

        return total_loss / min(
            len(real_dataloader), self.d_sample_num // self.model.batch_size
        ) / self.d_sample_training_epochs

    def _adversarial_train_epoch(self, train_data, epoch_idx):
        self.model.generator.train()
        total_loss = None
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)

        for idx, real_data in enumerate(real_dataloader):
            if (idx == self.adversarail_g_epochs):
                break
            losses = self.model.calculate_g_adversarial_loss(real_data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)

        for epoch_idx in range(self.adversarail_d_epochs):
            self._d_train_epoch(train_data, epoch_idx=epoch_idx)

        return total_loss / min(len(real_dataloader), self.adversarail_g_epochs)


class RankGANTrainer(GANTrainer):
    r"""RankGANTrainer is designed for RankGAN.
    """

    def __init__(self, config, model):
        super(RankGANTrainer, self).__init__(config, model)

    def _d_train_epoch(self, train_data, epoch_idx):
        r"""Train the discriminator module in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.discriminator.train()
        total_loss = None
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        fake_data = self.model.sample(self.d_sample_num)
        fake_dataloader = DataLoader(fake_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)

        ref_index = np.random.randint(0, real_data.shape[0], size=self.model.ref_size)
        ref_data = real_data[ref_index]  # ref_size * l

        for _ in range(self.d_sample_training_epochs):
            for real_data, fake_data in zip(real_dataloader, fake_dataloader):
                losses = self.model.calculate_d_train_loss(real_data, fake_data, ref_data, epoch_idx=epoch_idx)
                total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)

        return total_loss / min(len(real_dataloader), len(fake_dataloader)) / self.d_sample_training_epochs

    def _adversarial_train_epoch(self, train_data, epoch_idx):
        r"""Adversarial training in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.generator.train()
        total_loss = None
        real_data = self._get_real_data(train_data)
        ref_index = np.random.randint(0, real_data.shape[0], size=self.model.ref_size)
        ref_data = real_data[ref_index]  # ref_size * l

        losses = self.model.calculate_g_adversarial_loss(ref_data, epoch_idx=epoch_idx)
        total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)

        d_loss = 0
        for epoch_idx in range(self.adversarail_d_epochs):
            d_loss += self._d_train_epoch(train_data, epoch_idx=epoch_idx)
        d_loss = d_loss / self.adversarail_d_epochs

        return total_loss


class Seq2SeqTrainer(Trainer):
    r"""Seq2SeqTrainer is designed for seq2seq testing, which is a typically used setting.
    """

    def __init__(self, config, model):
        super(Seq2SeqTrainer, self).__init__(config, model)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, eval=True):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if not eval:
            print(self.model.generate(eval_data.__next__(), eval_data))
            return

        generate_corpus = []
        eval_data = tqdm(eval_data) if self.is_logger else eval_data
        for batch_data in eval_data:
            generate_corpus.extend(self.model.generate(batch_data, eval_data))
        self._save_generated_text(generate_corpus)
        reference_corpus = eval_data.get_reference()
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)

        return result


class MaskGANTrainer(GANTrainer):
    r""" Trainer specifically designed for MaskGAN training process.
    """

    def __init__(self, config, model):
        super(MaskGANTrainer, self).__init__(config, model)
        self.max_length = config["seq_len"]
        self.eos_token_idx = model.eos_token_idx
        self.adversarail_c_epochs = config['adversarail_c_epochs']
        self.g_mask_pretraining_epochs = config['g_mask_pretraining_epochs']
        self.g_lr = config['gen_learning_rate']
        self.d_lr = config['dis_learning_rate']
        self.c_lr = config['critic_learning_rate']
        self.g_optimizer = self._build_module_optimizer_(self.model.generator, self.g_lr)
        self.d_optimizer = self._build_module_optimizer_(self.model.discriminator, self.d_lr)
        self.c_optimizer = self._build_module_optimizer_(self.model.discriminator.critic_fc_linear, self.c_lr)
        self.pre_lm_weight = config["pre_lm_weight"]
        self.pretrain_lm_epochs = config["pretrain_lm_epochs"]
        self.checkp = config['checkp']

    def _build_module_optimizer_(self, module, lr):
        r""" Init the Module Optimizer with specified learning rate
        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(module.parameters(), lr)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(module.parameters(), lr)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(module.parameters(), lr)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(module.parameters(), lr)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(module.parameters(), lr)

        return optimizer

    def _optimize_step(self, losses, total_loss, model, opt, retain_graph=False):
        r""" Add retain_graph option
        """
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        _check_nan(loss)

        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        opt.step()
        return total_loss

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses, train_info=""):
        r""" Specified for maskgan output
        """
        train_loss_output = "%straining [time: %.2fs, " % (train_info, e_time - s_time)
        if isinstance(losses, dict):
            for key, loss in losses.items():
                train_loss_output += '%s: %.4f, ' % (key, loss)
            train_loss_output = train_loss_output[:-2]
        else:
            train_loss_output += "train loss: %.4f" % losses
        return train_loss_output + ']'

    def pretrain_lm(self, train_data, valid_data, verbose):
        r""" Pretrain rnn-based Language Model with teacher forcing mechanism
        """

        def lm_forward(data):
            r""" One iteration of LM forward
            """
            input = data[:, :-1]  # bs * self.max_len - 1
            target = data[:, 1:]
            bs, seq_len = target.size()
            lengths = torch.tensor([seq_len] * bs)
            target_present = torch.ones_like(input).byte()
            device = target.device
            lengths = lengths.cuda(device)

            # pretaining
            encoder_outputs = pre_train_lm(input, lengths, target, target_present, pretrain=True)
            logit = pre_train_lm.vocab_linear(encoder_outputs)
            logit = logit.permute([0, 2, 1])
            lossf = torch.nn.CrossEntropyLoss()
            loss = lossf(logit, target)
            return loss

        pre_train_lm = self.model.generator
        lm_opt = self._build_module_optimizer_(pre_train_lm, lr=0.001)
        for epoch in range(self.pretrain_lm_epochs):
            total_loss = None
            real_data = self._get_real_data(train_data)  # bs * self.max_len
            real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
            for batch_idx, data in enumerate(real_dataloader):
                loss = lm_forward(data)
                total_loss = self._optimize_step(loss, total_loss, pre_train_lm, lm_opt)

            total_loss = total_loss / len(real_dataloader)
            if verbose:
                self.logger.info(
                    "Epoch {}/{} of LM pretraining loss: {} ".format(epoch + 1, self.pretrain_lm_epochs, total_loss)
                )

            ppl = 0.0
            if (epoch + 1) % 1 == 0:
                pre_train_lm.eval()
                validate_data = self._get_real_data(valid_data)  # bs * self.max_len
                validate_dataloader = DataLoader(
                    validate_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True
                )
                ppl = 0.0
                for batch_idx, data in enumerate(validate_dataloader):
                    cross_entropy_loss = lm_forward(data)
                    ppl += math.exp(cross_entropy_loss.item())
                ppl = ppl / len(validate_dataloader)
                pre_train_lm.train()
                if verbose:
                    self.logger.info(
                        "Epoch {}/{} of LM pretraining PPL: {}...".format(epoch + 1, self.pretrain_lm_epochs, ppl)
                    )
                if ppl < 110:
                    state_dict = {
                        'embedder': pre_train_lm.embedder,
                        'encoder': pre_train_lm.encoder.encoder,
                        'vocab_linear': pre_train_lm.vocab_linear
                    }
                    self.pre_lm_weight = "saved/pretrain_lm_weight" + str(epoch + 1) + ".pkl"
                    torch.save(state_dict, self.pre_lm_weight)
                    if verbose:
                        self.logger.info("End LM pretraining. PPL: {}".format(ppl))
                        self.logger.info("Weigth saved in {}".format(self.pre_lm_weight))
                    return pre_train_lm, ppl

    def _g_train_epoch(self, train_data, epoch_idx):
        self.model.generator.train()
        total_loss = None
        real_data = self._get_real_data(train_data)  # bs * self.max_len
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        for batch_idx, data in enumerate(real_dataloader):
            loss = self.model.calculate_g_train_loss(data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(loss, total_loss, self.model.generator, self.g_optimizer)
        total_loss = total_loss / len(real_dataloader)
        return total_loss

    def _get_validate_ppl(self, validate_data, epoch_idx):
        self.model.generator.eval()
        ppl = 0.0
        validate_data = self._get_real_data(validate_data)  # bs * self.max_len
        validate_dataloader = DataLoader(validate_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        for batch_idx, data in enumerate(validate_dataloader):
            loss = self.model.calculate_g_train_loss(data, epoch_idx=epoch_idx, validate=True)
            ppl += math.exp(loss.item())
        ppl = ppl / len(validate_dataloader)
        self.model.generator.train()
        return ppl

    def _d_train_epoch(self, train_data, epoch_idx):
        self.model.discriminator.train()
        total_loss = None
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        for batch_idx, data in enumerate(real_dataloader):
            losses = self.model.calculate_d_train_loss(data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)

        return total_loss / len(real_dataloader)

    def _adversarial_train_epoch(self, train_data, epoch_idx):
        r""" Specified for MaskGAN adversarial training
        """
        dis_total_loss = None
        gen_total_loss = None
        critic_total_loss = None
        g_num = 0.0
        d_num = 0.0
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)

        dis_train_data = copy.deepcopy(real_dataloader)
        gen_train_data = copy.deepcopy(real_dataloader)
        c_train_data = copy.deepcopy(real_dataloader)

        dis_train_data = iter(dis_train_data)
        gen_train_data = iter(gen_train_data)
        _ = next(dis_train_data)  # have one offset

        for g_x in gen_train_data:
            g_num += 1
            for _ in range(3):
                d_num += 1
                try:
                    d_x = next(dis_train_data)
                except StopIteration:
                    del dis_train_data
                    dis_train_data = copy.deepcopy(real_dataloader)
                    dis_train_data = iter(dis_train_data)
                    d_x = next(dis_train_data)
                losses = self.model.calculate_d_train_loss(d_x, epoch_idx=_)
                dis_total_loss = self._optimize_step(losses, dis_total_loss, self.model.discriminator, self.d_optimizer)

            gen_losses, critic_losses = self.model.calculate_g_adversarial_loss(g_x, epoch_idx=g_num)
            gen_total_loss = self._optimize_step(gen_losses, gen_total_loss, self.model.generator, self.g_optimizer)
            critic_total_loss = self._optimize_step(
                critic_losses, critic_total_loss, self.model.discriminator.critic_fc_linear, self.c_optimizer
            )
        return {
            "dis_loss": dis_total_loss / d_num,
            "gen_loss": gen_total_loss / g_num,
            "critic_loss": critic_total_loss / g_num
        }

    def _evaluate_nll_test(self, eval_data):
        total_loss = 0
        real_data = self._get_real_data(eval_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        for batch_idx, data in enumerate(real_dataloader):
            nll_test = self.model.calculate_nll_test(data, batch_idx)
            total_loss += float(nll_test)
        return total_loss / len(eval_data)

    def _add_eos(self, data, length):
        batch_size, pad_seq_len = data.size()
        padded_data = torch.full((batch_size, self.max_length),
                                 self.eos_token_idx,
                                 dtype=torch.long,
                                 device=self.device)
        for i in range(batch_size):
            l = int(length[i].cpu().data)
            if l == self.max_length + 2:
                padded_data[i, :] = data[i, 1:l - 1]
            else:
                padded_data[i, 0:l - 1] = data[i, 1:l]
        return padded_data

    def _get_real_data(self, train_data):
        real_datas = []
        for corpus in train_data:
            real_data = corpus['target_idx']  # bs*batch_max_seq_len
            length = corpus['target_length']
            real_data = self._add_eos(real_data, length)
            real_datas.append(real_data)

        real_datas = torch.cat(real_datas, dim=0)
        return real_datas

    def _save_checkpoint(self, epoch, postfix=None):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'g_opt': self.g_optimizer.state_dict(),
            'd_opt': self.d_optimizer.state_dict(),
            'c_opt': self.c_optimizer.state_dict()
        }
        if postfix is not None:
            path = self.saved_model_file + "_" + str(epoch) + "_" + postfix
            torch.save(state, path)
            return path
        else:
            torch.save(state, self.saved_model_file)

    def _load_generated_text(self):
        r""" Load the generated text by our model to log.
        """
        with open(self.saved_text_file, 'r') as fin:
            samples = []
            for i in range(5):
                text = fin.readline()
                samples.append(text)
            return samples

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        # generator pretraining
        if self.checkp is not None:
            checkpoint = torch.load(self.checkp)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.d_optimizer.load_state_dict(checkpoint["d_opt"])
            self.g_optimizer.load_state_dict(checkpoint["g_opt"])
            epoch_check = checkpoint['epoch']
            if verbose:
                self.logger.info("Load checkpoint file from: {}".format(self.checkp))
        else:
            if self.pre_lm_weight is None:
                if verbose:
                    self.logger.info("Start LM pretraining...")
                pretrain_lm, ppl = self.pretrain_lm(train_data, valid_data, verbose)

                pretrain_lm = torch.load(self.pre_lm_weight)
                embedder = pretrain_lm['embedder'].state_dict()
                lstm = pretrain_lm['encoder'].state_dict()
                vocab_linear = pretrain_lm['vocab_linear'].state_dict()

                self.model.generator.embedder.load_state_dict(embedder)
                self.model.generator.encoder.encoder.load_state_dict(lstm)
                self.model.generator.decoder.decoder.load_state_dict(lstm)
                self.model.generator.vocab_linear.load_state_dict(vocab_linear)
                self.model.discriminator.encoder.encoder.load_state_dict(lstm)
                self.model.discriminator.decoder.decoder.load_state_dict(lstm)
                if verbose:
                    self.logger.info("Load pretrained LM weight")
            else:
                pretrain_lm = torch.load(self.pre_lm_weight)
                embedder = pretrain_lm['embedder'].state_dict()
                lstm = pretrain_lm['encoder'].state_dict()
                vocab_linear = pretrain_lm['vocab_linear'].state_dict()

                self.model.generator.embedder.load_state_dict(embedder)
                self.model.generator.encoder.encoder.load_state_dict(lstm)
                self.model.generator.decoder.decoder.load_state_dict(lstm)
                self.model.generator.vocab_linear.load_state_dict(vocab_linear)
                self.model.discriminator.encoder.encoder.load_state_dict(lstm)
                self.model.discriminator.decoder.decoder.load_state_dict(lstm)
                if verbose:
                    self.logger.info("Load pretrained LM weight from: {}".format(self.pre_lm_weight))

        if verbose:
            self.logger.info("Start generator mask pretraining...")
        for epoch_idx in range(self.g_mask_pretraining_epochs):
            training_start_time = time()
            train_loss = self._g_train_epoch(train_data, epoch_idx)
            self.g_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss,
                                                 "generator pre")
            if verbose:
                self.logger.info(train_loss_output)

            ppl = self._get_validate_ppl(valid_data, epoch_idx)
            if verbose:
                self.logger.info(
                    "Epoch {}/{} of mask pretraining PPL: {}...".format(
                        epoch_idx + 1, self.g_mask_pretraining_epochs, ppl
                    )
                )
            if ppl <= 90:
                if verbose:
                    path = self._save_checkpoint(epoch_idx + 1, postfix="pretrain_gen")
                    self.logger.info(">>>> [Pretrain Gen] PPL: {} save weight in {}".format(ppl, path))
                    self.logger.info("End generator mask pretraining...")
                    break
            if (epoch_idx) % 10 == 0:
                self.logger.info(">>>> [Pretrain Gen] Save pretrain gen check in epoch %d ..." % (epoch_idx + 1))
                path = self._save_checkpoint(epoch_idx + 1, postfix="pretrain_gen")

                self.model.eval()
                test_result = self.evaluate(valid_data, model_file=path)
                self.model.train()
                sample = self._load_generated_text()
                tmp = "\n"
                for i, s in enumerate(sample):
                    tmp += str(i)
                    tmp += ": "
                    tmp += s.strip()
                    tmp += "\n"
                self.logger.info('>>>> [Pretrain Gen] test result: {}'.format(test_result))
                self.logger.info('>>>> [Pretrain Gen] test result samples: {}'.format(tmp))

        # discriminator pretraining
        if verbose:
            self.logger.info("Start discriminator pretraining...")
        for epoch_idx in range(self.d_pretraining_epochs):
            training_start_time = time()
            train_loss = self._d_train_epoch(train_data, epoch_idx)
            self.d_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss,
                                                 "discriminator pre")
            if verbose:
                self.logger.info(train_loss_output)
        if verbose:
            self.logger.info("End discriminator pretraining...")

        # adversarial training
        if verbose:
            self.logger.info("Start adversarial training...")
        for epoch_idx in range(self.adversarail_training_epochs):
            training_start_time = time()
            train_loss = self._adversarial_train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            if (epoch_idx + 1) % 10 == 0:
                path = self._save_checkpoint((epoch_idx + 1), postfix="adv_train")
                self.model.eval()
                test_result = self.evaluate(valid_data, model_file=path)
                self.model.train()

                sample = self._load_generated_text()
                tmp = "\n"
                for i, s in enumerate(sample):
                    tmp += str(i)
                    tmp += ": "
                    tmp += s.strip()
                    tmp += "\n"
                self.logger.info('>>>>>> [Adv] test result: {}'.format(test_result))
                self.logger.info('>>>>>> [Adv] test result samples: {}'.format(tmp))

        if verbose:
            self.logger.info("End adversarial pretraining...")

        self._save_checkpoint(self.adversarail_training_epochs)
        return -1, None


class LeakGANTrainer(GANTrainer):
    r"""Specified for leakgan trainer
    """

    def __init__(self, config, model):
        super(LeakGANTrainer, self).__init__(config, model)
        self.interleaved_pretrain_epoch = config['interleaved_pretrain_epoch']
        self.adversarail_g_epochs = config['adversarail_g_epochs']
        gen_lr = config['generator_lr']  # 0.001
        dis_lr = config['discriminator_lr']  # 0.00005
        self.g_optimizer = self._build_module_optimizer_(self.model.generator, gen_lr)  # (manager_opt, worker_opt)
        self.d_optimizer = self._build_module_optimizer_(self.model.discriminator, dis_lr)
        self.iters_num = config['iter_num']
        self.eos_token_idx = model.eos_token_idx

    def _build_module_optimizer_(self, module, learing_rate):
        r"""Specified for leakgan
        """
        multi_flag = False
        if module._get_name() == 'LeakGANGenerator':
            manager_params, worker_params = module.split_params()
            multi_flag = True

        if self.learner.lower() == 'adam':
            if multi_flag:
                manager_opt = optim.Adam(manager_params, lr=learing_rate)
                worker_opt = optim.Adam(worker_params, lr=learing_rate)
            else:
                optimizer = optim.Adam(module.parameters(), lr=learing_rate)
        elif self.learner.lower() == 'sgd':
            if multi_flag:
                manager_opt = optim.SGD(manager_params, lr=learing_rate)
                worker_opt = optim.SGD(worker_params, lr=learing_rate)
            else:
                optimizer = optim.SGD(module.parameters(), lr=learing_rate)
        elif self.learner.lower() == 'adagrad':
            if multi_flag:
                manager_opt = optim.Adagrad(manager_params, lr=learing_rate)
                worker_opt = optim.Adagrad(worker_params, lr=learing_rate)
            else:
                optimizer = optim.Adagrad(module.parameters(), lr=learing_rate)
        elif self.learner.lower() == 'rmsprop':
            if multi_flag:
                manager_opt = optim.RMSprop(manager_params, lr=learing_rate)
                worker_opt = optim.RMSprop(worker_params, lr=learing_rate)
            else:
                optimizer = optim.RMSprop(module.parameters(), lr=learing_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            if multi_flag:
                manager_opt = optim.Adam(manager_params, lr=learing_rate)
                worker_opt = optim.Adam(worker_params, lr=learing_rate)
            else:
                optimizer = optim.Adam(module.parameters(), lr=learing_rate)
        if multi_flag:
            return (manager_opt, worker_opt)
        else:
            return optimizer

    def _optimize_step(self, losses, total_loss, model, opt):
        r"""Specified for leakgan optimize
        """
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        _check_nan(loss)

        if isinstance(losses, tuple):
            for i, (o, loss) in enumerate(zip(opt, losses)):
                o.zero_grad()
                loss.backward(retain_graph=True if i < len(opt) - 1 else False)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

            for o in opt:
                o.step()
        else:
            opt.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()

        return total_loss

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses, train_info=""):
        r"""Specified for leakgan output format
        """
        train_loss_output = "%straining [time: %.2fs, " % (train_info, e_time - s_time)
        if isinstance(losses, dict):
            for key, loss in losses.items():
                train_loss_output += '%s: %.4f, ' % (key, loss)
            train_loss_output = train_loss_output[:-2]
        else:
            train_loss_output += "train loss: %.4f" % losses
        return train_loss_output + ']'

    def _add_eos(self, data, length):
        batch_size = data.shape[0]
        padded_data = torch.full((batch_size, self.max_length),
                                 self.eos_token_idx,
                                 dtype=torch.long,
                                 device=self.device)
        for i in range(batch_size):
            len = length[i].cpu().data
            padded_data[i, :len] = data[i, :len]
        return padded_data

    def _get_real_data(self, train_data):
        r"""Specified for leakgan which use eos_idx pad not pad_idx
        """
        real_datas = []
        for corpus in train_data:
            real_data = corpus['target_idx']
            length = corpus['target_length']
            real_data = self._add_eos(real_data, length)
            real_datas.append(real_data)

        real_datas = torch.cat(real_datas, dim=0)
        return real_datas

    def _adversarial_train_epoch(self, train_data, epoch_idx):
        r"""Specified for leakgan adversarial training
        """
        self.model.generator.train()
        total_g_loss = None
        total_d_loss = 0
        total_d_acc = 0
        adv_mana_loss = 0
        adv_work_loss = 0
        adv_d_loss = 0
        for e in range(self.adversarail_g_epochs):
            losses = self.model.calculate_g_adversarial_loss(epoch_idx=e)
            total_g_loss = self._optimize_step(losses, total_g_loss, self.model.generator, self.g_optimizer)
        adv_mana_loss, adv_work_loss = total_g_loss
        adv_mana_loss = adv_mana_loss / self.adversarail_g_epochs
        adv_work_loss = adv_work_loss / self.adversarail_g_epochs

        for e in range(self.adversarail_d_epochs):
            loss_dict = self._d_train_epoch(train_data, epoch_idx=epoch_idx)
            total_d_loss = total_d_loss + loss_dict['total_loss']
            total_d_acc = total_d_acc + loss_dict['train_acc']
        adv_d_loss = total_d_loss / self.adversarail_d_epochs
        adv_c_loss = total_d_acc / self.adversarail_d_epochs
        return {"mana_loss": adv_mana_loss, "work_loss": adv_work_loss, "dis_loss": adv_d_loss, "train_acc": adv_c_loss}

    def _g_train_epoch(self, train_data, epoch_idx):
        total_loss = None
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        for batch_idx, data in enumerate(real_dataloader):
            # interaction = interaction.to(self.device)
            losses = self.model.calculate_g_train_loss(data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)
        total_loss = [l / len(real_dataloader)
                      for l in total_loss] if isinstance(total_loss, tuple) else total_loss / len(train_data)
        mana_loss, work_loss = total_loss
        return {"mana_loss": mana_loss, "work_loss": work_loss}

    def _d_train_epoch(self, train_data, epoch_idx):
        total_loss = None
        total_acc = 0
        real_data = self._get_real_data(train_data)
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)
        # not need sample self.d_sample_num numbers becauese only train discriminator 5 batch
        d_sample_num = (self.d_sample_training_epochs + 1) * self.model.batch_size
        fake_data = self.model.sample(d_sample_num)

        fake_dataloader = DataLoader(fake_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)

        idx = 0
        for real_data, fake_data in zip(real_dataloader, fake_dataloader):
            # self.model.discriminator.eval() # pretraining not use dropout
            if idx == self.d_sample_training_epochs:
                break
            losses, acc = self.model.calculate_d_train_loss(real_data, fake_data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)
            total_acc = total_acc + acc
            idx += 1

        total_loss = total_loss / self.d_sample_training_epochs
        total_acc = total_acc / self.d_sample_training_epochs

        return {"total_loss": total_loss, "train_acc": total_acc}

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        # pretraining
        if verbose:
            self.logger.info(">> Start pretraining")
        # generator pretraining
        for epoch_idx in range(self.g_pretraining_epochs):  # 80
            if verbose:
                self.logger.info(
                    ">>>> [Pretrain Gen] Start %d / %d epochs generator pretraining" %
                    (epoch_idx + 1, self.g_pretraining_epochs)
                )
            training_start_time = time()
            train_loss = self._g_train_epoch(train_data, epoch_idx)
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx + 1, training_start_time, training_end_time, train_loss,
                                                 "generator pre")
            train_loss_output = ">>>> " + train_loss_output
            if verbose:
                self.logger.info(train_loss_output)

        # discriminator pretraining
        for epoch_idx in range(self.d_pretraining_epochs):  # 5
            if verbose:
                self.logger.info(
                    ">>>> [Pretrain Dis]Start %d / %d epochs discriminator pretraining..." %
                    (epoch_idx + 1, self.d_pretraining_epochs)
                )
            training_start_time = time()
            train_loss = self._d_train_epoch(train_data, epoch_idx)
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss,
                                                 "discriminator pre")
            train_loss_output = ">>>> " + train_loss_output
            if verbose:
                self.logger.info(train_loss_output)
        if verbose:
            self.logger.info(">> End pretraining")

        # adversarial training
        if verbose:
            self.logger.info(">> Start adversarial training")
        for epoch in range(int(self.iters_num / self.adversarail_training_epochs)):
            if verbose:
                self.logger.info(">>>> [Adv] Start epoch %d / 10 interleaved adversarial training" % (epoch + 1))

            for epoch_idx in range(self.adversarail_training_epochs):
                if verbose:
                    self.logger.info(
                        ">>>>>> [Adv] Start epoch %d / %d adversarial training" %
                        (epoch_idx + 1, self.adversarail_training_epochs)
                    )
                training_start_time = time()
                train_loss = self._adversarial_train_epoch(train_data, epoch_idx)
                # self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output((epoch_idx + 1), training_start_time, training_end_time,
                                                     train_loss,
                                                     train_info="adv ")
                train_loss_output = ">>>>>> " + train_loss_output
                if verbose:
                    self.logger.info(train_loss_output)

            # gen pretrain
            for epoch_idx in range(5):
                if verbose:
                    self.logger.info(">>>>>> [Adv] Start epoch %d / 5 pretrain generator" % (epoch_idx + 1))
                training_start_time = time()
                train_loss = self._g_train_epoch(train_data, epoch_idx)
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output((epoch_idx + 1), training_start_time, training_end_time,
                                                     train_loss,
                                                     "adv generator pre")
                train_loss_output = ">>>>>> " + train_loss_output
                if verbose:
                    self.logger.info(train_loss_output)

            # dis pretrain
            for epoch_idx in range(5):  # d_steps
                if verbose:
                    self.logger.info(">>>>>> [Adv] Start epoch %d / 5 pretrain discriminator" % (epoch_idx + 1))
                training_start_time = time()
                train_loss = self._d_train_epoch(train_data, epoch_idx)
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output((epoch_idx + 1), training_start_time, training_end_time,
                                                     train_loss,
                                                     "adv discriminator pre")
                train_loss_output = ">>>>>> " + train_loss_output
                if verbose:
                    self.logger.info(train_loss_output)

        self._save_checkpoint(self.adversarail_training_epochs)
        return -1, None
