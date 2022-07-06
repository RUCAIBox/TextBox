import os
import torch
import torch.optim as optim
import numpy as np
import collections

from tqdm import tqdm

from torch.utils.data import DataLoader
from time import time
from logging import getLogger

from textbox.module.optimizer import InverseSquareRootOptim, CosineOptim, LinearOptim, ConstantOptim
from textbox.evaluator import BaseEvaluator, evaluator_list
from textbox.utils import ensure_dir, early_stopping


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of text generation system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.
        """

        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method [next] should be implemented.')


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
        self.logger = getLogger()
        self.optimizer = config['optimizer'].lower()
        self.scheduler = config['scheduler'].lower() if config['scheduler'] else None
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
        self.checkpoint_dir = config['checkpoint_dir']
        self.grad_clip = config['grad_clip']
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
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        self.metrics = config["metrics"]
        self._check_metrics()
        self.evaluator = BaseEvaluator(config, self.metrics)

        self.is_logger = (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP
        self.item_tensor = None
        self.tot_item_num = None
        self.iid_field = config['ITEM_ID_FIELD']

    def _check_metrics(self):
        r"""check the correct of the setting"""
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                if self.metrics[0] == '[':
                    self.metrics = self.metrics[1:]
                if self.metrics[-1] == ']':
                    self.metrics = self.metrics[:-1]
                self.metrics = self.metrics.strip().split(",")
            self.metrics = [metric.lower() for metric in self.metrics]
            for metric in self.metrics:
                if metric not in evaluator_list:
                    raise ValueError(
                        "evaluator {} can't be found. ".format(metric) + "(evaluator should be in [" +
                        ", ".join(evaluator_list) + "])"
                    )
        else:
            raise TypeError('evaluator must be a string or list')

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
                self.logger.info("Using none scheduler")
                _optim = base_optim
            return _optim

        optimizer = _get_base_optimizer(self.optimizer)
        optimizer = _get_schedule(self.scheduler, optimizer)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        total_loss = None

        pbar = train_data
        if self.is_logger:
            pbar = tqdm(pbar)

        for data in pbar:
            self.optimizer.zero_grad()
            losses = self.model(data, epoch_idx=epoch_idx)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        train_loss = total_loss / len(train_data)
        return train_loss

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        with torch.no_grad():
            self.model.eval()
            total_loss = None
            for batch_idx, data in enumerate(valid_data):
                losses = self.model(data)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                self._check_nan(loss)
            valid_loss = total_loss / len(valid_data)
            ppl = np.exp(valid_loss)
        return valid_loss, ppl

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.DDP:
            if (torch.distributed.get_rank() == 0):
                saved_dict = collections.OrderedDict()
                for key, val in state.items():
                    if (key == 'state_dict'):
                        for state_dict_key, state_dict_val in val.items():
                            if (state_dict_key[0:7] == 'module.'):
                                changed_key = state_dict_key[7:]
                            else:
                                changed_key = state_dict_key
                            saved_dict[changed_key] = state_dict_val
                        state[key] = saved_dict
                torch.save(state, self.saved_model_file)
        else:
            torch.save(state, self.saved_model_file)

    def _save_generated_text(self, generated_corpus):
        r"""Store the generated text by our model.

        Args:
            corpus (list of string list):
        """
        with open(self.saved_text_file, 'w') as fin:
            for tokens in generated_corpus:
                fin.write(' '.join(tokens) + '\n')

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            if self.is_logger:
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
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        if self.is_logger:
            self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses, train_info=""):
        train_loss_output = "epoch %d %straining [time: %.2fs, " % (epoch_idx, train_info, e_time - s_time)
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                train_loss_output += 'train_loss%d: %.4f, ' % (idx + 1, loss)
            train_loss_output = train_loss_output[:-2]
        else:
            train_loss_output += "train loss: %.4f" % losses
        return train_loss_output + ']'

    def reduce_loss(self, loss):
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss /= torch.distributed.get_world_size()
        return loss

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if self.start_epoch >= self.epochs or self.epochs <= 0:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            training_end_time = time()
            train_loss = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            if self.DDP:
                train_loss = self.reduce_loss(torch.tensor(train_loss).to("cuda")).item()
            self.train_loss_dict[epoch_idx] = train_loss
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                if self.is_logger:
                    self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current: %s' % self.saved_model_file
                    if verbose:
                        if self.is_logger:
                            self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                with torch.no_grad():
                    valid_score, valid_result = self._valid_epoch(valid_data)
                # valid_loss, ppl
                if self.DDP:
                    valid_score = self.reduce_loss(torch.tensor(valid_score).to("cuda")).item()
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step, max_step=self.stopping_step, bigger=False
                )
                # better model are supposed to provide smaller perplexity and loss
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_loss: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid ppl: {}'.format(valid_result)
                if verbose:
                    if self.is_logger:
                        self.logger.info(valid_score_output)
                        self.logger.info(valid_result_output)
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            if self.is_logger:
                                self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        if self.is_logger:
                            self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result


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
        with torch.no_grad():
            for batch_data in tqdm(eval_data):
                generate_corpus.extend(self.model.generate(batch_data, eval_data))
        self._save_generated_text(generate_corpus)
        reference_corpus = eval_data.get_reference()
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)

        return result
