# @Time   : 2020/11/5
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

r"""
textbox.trainer.trainer
################################
"""

import os
import itertools
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from time import time
from logging import getLogger

from textbox.evaluator import NgramEvaluator
from textbox.data.corpus import Corpus
from textbox.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    DataLoaderType, EvaluatorType


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
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
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evalute(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """
    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = 100000000
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.evaluator = NgramEvaluator(config)
        # self.eval_type = config['eval_type']
        # if self.eval_type == EvaluatorType.INDIVIDUAL:
        #     self.evaluator = LossEvaluator(config)
        # else:
        #     self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.iid_field = config['ITEM_ID_FIELD']

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
        for batch_idx, data in enumerate(train_data):
            # interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss(data, epoch_idx=epoch_idx)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
        return total_loss / len(train_data)

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        self.model.eval()
        total_loss = None
        valid_len = float(len(valid_data))
        for batch_idx, data in enumerate(valid_data):
            # interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = self.model.calculate_loss(data)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
        self.optimizer.zero_grad()
        valid_loss = total_loss / valid_len
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
        torch.save(state, self.saved_model_file)

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
            self.logger.warning('Architecture configuration given in config file is different from that of checkpoint. '
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses, train_info=""):
        train_loss_output = "epoch %d " + train_info + "training [time: %.2fs, " % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                train_loss_output += 'train_loss%d: %.4f, ' % (idx + 1, loss)
            train_loss_output = train_loss_output[:-2]
        else:
            train_loss_output += "train loss: %.4f" % losses
        return train_loss_output + ']'

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
        # if hasattr(self.model, 'train_preparation'):
        #     self.model.train_preparation(train_data=train_data, valid_data=valid_data)
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current: %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                # valid_loss, ppl
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=False)
                # better model are supposed to provide smaller perplexity and loss
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_loss: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid ppl: {}'.format(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None):
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
        generate_corpus = self.model.generate(eval_data)
        reference_corpus = eval_data.get_reference()
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)

        return result

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


class UnconditionalTrainer(Trainer):
    r"""UnconditionalTrainer is designed for RNN, which is a knowledge-aware recommendation method.
    """

    def __init__(self, config, model):
        super(UnconditionalTrainer, self).__init__(config, model)


class GANTrainer(Trainer):
    r"""GANTrainer is designed for GAN, which is a generative adversarial net method.
    """

    def __init__(self, config, model):
        super(GANTrainer, self).__init__(config, model)

        self.optimizer = None
        self.g_optimizer = self._build_module_optimizer(self.model.generator)
        self.d_optimizer = self._build_module_optimizer(self.model.discriminator)

        self.grad_clip = config['grad_clip']
        self.g_pretraining_epochs = config['g_pretraining_epochs']
        self.d_pretraining_epochs = config['d_pretraining_epochs']
        self.d_sample_num = config['d_sample_num']
        self.d_sample_training_epochs = config['d_sample_training_epochs']
        self.adversarail_training_epochs = config['adversarail_training_epochs']
        self.adversarail_d_epochs = config['adversarail_d_epochs']

        self.g_pretraining_loss_dict = dict()
        self.d_pretraining_loss_dict = dict()
    
    def _build_module_optimizer(self, module):
        r"""Init the Module Optimizer

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
        opt.zero_grad()
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        self._check_nan(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        opt.step()
        return total_loss

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
            # interaction = interaction.to(self.device)
            losses = self.model.calculate_g_train_loss(data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)
        
        return total_loss / len(train_data)
    
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
        fake_data = self.model.sample(self.d_sample_num) # a dataloader of fake samples generated by generator

        for _ in range(self.d_sample_training_epochs):
            for corpus, fake_data in zip(train_data, fake_data):
                # interaction = interaction.to(self.device)
                real_data = corpus['target_text']
                losses = self.model.calculate_d_train_loss(real_data, fake_data, epoch_idx=epoch_idx)
                total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)

        return total_loss / len(train_data) / self.d_sample_training_epochs
    
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
        # generator pretraining
        if verbose:
            self.logger.info("Start generator pretraining...")
        for epoch_idx in range(self.g_pretraining_epochs):
            training_start_time = time()
            train_loss = self._g_train_epoch(train_data, epoch_idx)
            self.g_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss, "generator pre")
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
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss, "discriminator pre")
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
