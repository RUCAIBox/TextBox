# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/12/2, 2020/11/27, 2020/12/3, 2020/12/15
# @Author : Jinhao Jiang, Xiaoxuan Hu, Tianyi Tang, Jinhao Jiang
# @Email  : jiangjinhao@std.uestc.edu.cn, huxiaoxuan@ruc.edu.cn, steventang@ruc.edu.cn, jiangjinhao@std.uestc.edu.cn

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
import copy

from torch.utils.data import DataLoader
from time import time
from logging import getLogger

from textbox.module.Optimizer.optim import ScheduledOptim
from textbox.evaluator import NgramEvaluator, TranslationEvaluator, SummarizationEvaluator
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
        self.embedding_size = config['embedding_size']
        self.warmup_steps = config['warmup_steps']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.generated_text_dir = config['generated_text_dir']
        ensure_dir(self.generated_text_dir)
        saved_text_file = '{}-{}.txt'.format(self.config['model'], get_local_time())
        self.saved_text_file = os.path.join(self.generated_text_dir, saved_text_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = 100000000
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.task_type = config['task_type'].lower()
        if self.task_type == "translation":
            self.evaluator = TranslationEvaluator(config)
        elif self.task_type == "summarization":
            self.evaluator = SummarizationEvaluator(config)
        else:
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
        elif self.learner.lower() == 'schedule':
            optimizer = ScheduledOptim(optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                                       self.learning_rate, self.embedding_size, self.warmup_steps)
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
        self.model.eval()
        total_loss = None
        for batch_idx, data in enumerate(valid_data):
            # interaction = interaction.to(self.device)
            # self.optimizer.zero_grad()
            losses = self.model.calculate_loss(data)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
        # self.optimizer.zero_grad()
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
        train_loss_output = "epoch %d %straining [time: %.2fs, " % (epoch_idx, train_info, e_time - s_time)
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
            self._save_checkpoint(epoch_idx)
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
                with torch.no_grad():
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

    def _evaluate_nll_test(self, eval_data):
        r"""Calculate the negative log-likelihood of the eval_data.

        Args:
            eval_data (DataLoader): the eval data.

        Returns:
            Float: NLL_test of the eval data.
        """

        total_loss = 0
        for epoch_idx, eval_batch in enumerate(eval_data):
            nll_test = self.model.calculate_nll_test(eval_batch, epoch_idx)
            total_loss += float(nll_test)
        return total_loss / len(eval_data)

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
        with torch.no_grad():
            generate_corpus = self.model.generate(eval_data)
        self._save_generated_text(generate_corpus)
        reference_corpus = eval_data.get_reference()
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)
        result['nll_test'] = self._evaluate_nll_test(eval_data)

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
    r"""UnconditionalTrainer is designed for RNN, which is a typical unconditional generator.
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
        self.max_length = config['max_seq_length'] + 2
        self.pad_idx = model.pad_idx

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
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        self._check_nan(loss)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        opt.step()
        return total_loss

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
            'state_dict': self.model.state_dict()
        }
        torch.save(state, self.saved_model_file)

    def _add_pad(self, data):
        batch_size = data.shape[0]
        padded_data = torch.full((batch_size, self.max_length), self.pad_idx, dtype=torch.long, device=self.device)
        padded_data[:, : data.shape[1]] = data
        return padded_data

    def _get_real_data(self, train_data):
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
            # interaction = interaction.to(self.device)
            losses = self.model.calculate_g_train_loss(data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)
        total_loss = [l / len(train_data) for l in total_loss] if isinstance(total_loss, tuple) else total_loss / len(
            train_data)
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

        for _ in range(self.d_sample_training_epochs):
            for real_data in real_dataloader:
                fake_data, z = self.model.sample()
                losses = self.model.calculate_d_train_loss(real_data, fake_data, z, epoch_idx=epoch_idx)
                total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)

        return total_loss / len(real_dataloader) / self.d_sample_training_epochs

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
        real_dataloader = DataLoader(real_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True)

        for real_data in real_dataloader:
            losses = self.model.calculate_g_adversarial_loss(real_data, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)

        for epoch_idx in range(self.adversarail_d_epochs):
            self._d_train_epoch(train_data, epoch_idx=epoch_idx)

        return total_loss / len(real_dataloader)


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

        # d_output = "d_loss: %f" % (d_loss)
        # self.logger.info(d_output)
        return total_loss


class ConditionalTrainer(Trainer):
    r"""TranslationTrainer is designed for seq2seq testing, which is a typically used setting.
    """

    def __init__(self, config, model):
        super(ConditionalTrainer, self).__init__(config, model)

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
        self._save_generated_text(generate_corpus)
        reference_corpus = eval_data.get_reference()
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)

        return result


class MaskGANTrainer(GANTrainer):
    r""" Trainer specifically designed for MaskGAN training process.

    """

    def __init__(self, config, model):
        super(MaskGANTrainer, self).__init__(config, model)
        self.adversarail_c_epochs = config['adversarail_c_epochs']  # !!

    def _optimize_step(self, losses, total_loss, model, opt, retain_graph=False):
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        self._check_nan(loss)

        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
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
            losses = self.model.calculate_g_train_loss(data, is_advtrain=False, epoch_idx=epoch_idx)
            total_loss = self._optimize_step(losses, total_loss, self.model.generator, self.g_optimizer)
        total_loss = total_loss / len(train_data)
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

        for batch_idx, data in enumerate(train_data):
            losses = self.model.calculate_d_train_loss(data, epoch_idx=epoch_idx, is_advtrain=False)
            total_loss = self._optimize_step(losses, total_loss, self.model.discriminator, self.d_optimizer)

        return total_loss / len(train_data)

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
        dis_total_loss = None
        gen_total_loss = None
        critic_total_loss = None
        g_num = 0.0
        d_num = 0.0
        dis_train_data = copy.deepcopy(train_data)
        gen_train_data = copy.deepcopy(train_data)
        _ = next(dis_train_data)
        for g_x in gen_train_data:
            g_num += 1
            # self.model.discriminator.train()
            for _ in range(self.adversarail_d_epochs):
                d_num += 1
                try:
                    d_x = next(dis_train_data)
                except StopIteration:
                    del dis_train_data
                    dis_train_data = copy.deepcopy(train_data)
                    d_x = next(dis_train_data)
                losses = self.model.calculate_d_train_loss(d_x, epoch_idx=_, is_advtrain=True)
                dis_total_loss = self._optimize_step(losses, dis_total_loss, self.model.discriminator, self.d_optimizer)

            # self.model.generator.train()
            gen_losses, critic_losses = self.model.calculate_g_adversarial_loss(g_x, epoch_idx=g_num)
            gen_total_loss = self._optimize_step(gen_losses, gen_total_loss, self.model.generator, self.g_optimizer,
                                                 retain_graph=True)
            critic_total_loss = self._optimize_step(critic_losses, critic_total_loss, self.model.discriminator,
                                                    self.d_optimizer)
        return (dis_total_loss / d_num, gen_total_loss / g_num, critic_total_loss / g_num)


class LeakGANTrainer(GANTrainer):
    def __init__(self, config, model):
        super(LeakGANTrainer, self).__init__(config, model)
        self.interleaved_pretrain_epoch = config['interleaved_pretrain_epoch']
        self.adversarail_g_epochs = config['adversarail_g_epochs']
        self.g_optimizer = self._build_module_optimizer(self.model.generator)  # (manager_opt, worker_opt)
        self.d_optimizer = self._build_module_optimizer(self.model.discriminator)
        self.iters_num = config['iter_num']
        self.end_idx = model.end_idx

    def _build_module_optimizer(self, module):
        # specified for leakgan
        gen_lr = 0.001
        dis_lr = 0.00005
        multi_flag = False
        if module._get_name() == 'LeakGANGenerator':
            manager_params, worker_params = module.split_params()
            multi_flag = True

        if self.learner.lower() == 'adam':
            if multi_flag:
                manager_opt = optim.Adam(manager_params, lr=gen_lr)
                worker_opt = optim.Adam(worker_params, lr=gen_lr)
            else:
                optimizer = optim.Adam(module.parameters(), lr=dis_lr)
        elif self.learner.lower() == 'sgd':
            if multi_flag:
                manager_opt = optim.SGD(manager_params, lr=gen_lr)
                worker_opt = optim.SGD(worker_params, lr=gen_lr)
            else:
                optimizer = optim.SGD(module.parameters(), lr=dis_lr)
        elif self.learner.lower() == 'adagrad':
            if multi_flag:
                manager_opt = optim.Adagrad(manager_params, lr=gen_lr)
                worker_opt = optim.Adagrad(worker_params, lr=gen_lr)
            else:
                optimizer = optim.Adagrad(module.parameters(), lr=dis_lr)
        elif self.learner.lower() == 'rmsprop':
            if multi_flag:
                manager_opt = optim.RMSprop(manager_params, lr=gen_lr)
                worker_opt = optim.RMSprop(worker_params, lr=gen_lr)
            else:
                optimizer = optim.RMSprop(module.parameters(), lr=dis_lr)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            if multi_flag:
                manager_opt = optim.Adam(manager_params, lr=gen_lr)
                worker_opt = optim.Adam(worker_params, lr=gen_lr)
            else:
                optimizer = optim.Adam(module.parameters(), lr=dis_lr)
        if multi_flag:
            return (manager_opt, worker_opt)
        else:
            return optimizer

    def _optimize_step(self, losses, total_loss, model, opt):
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        self._check_nan(loss)

        if isinstance(losses, tuple):
            for i, (o, loss) in enumerate(zip(opt, losses)):
                o.zero_grad()
                loss.backward(retain_graph=True if i < len(opt) - 1 else False)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                o.step()
        else:
            opt.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()

        return total_loss

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses, train_info=""):
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
        padded_data = torch.full((batch_size, self.max_length), self.end_idx, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            len = length[i].cpu().data
            padded_data[i, :len] = data[i, :len]
        return padded_data

    def _get_real_data(self, train_data):
        # use eos_idx pad not pad_idx
        real_datas = []
        for corpus in train_data:
            real_data = corpus['target_idx']
            length = corpus['target_length']
            real_data = self._add_eos(real_data, length)
            real_datas.append(real_data)

        real_datas = torch.cat(real_datas, dim=0)
        return real_datas

    def _adversarial_train_epoch(self, train_data, epoch_idx):
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
        total_loss = [l / len(real_dataloader) for l in total_loss] if isinstance(total_loss,
                                                                                  tuple) else total_loss / len(
            train_data)
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

    def _save_checkpoint(self, epoch, postfix=None):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict()
        }
        if postfix is not None:
            path = self.saved_model_file + "_" + str(epoch) + "_" + postfix
            torch.save(state, path)
            return path
        else:
            torch.save(state, self.saved_model_file)

    def _load_generated_text(self):
        r"""Load the generated text by our model to log.

        """
        with open(self.saved_text_file, 'r') as fin:
            samples = []
            for i in range(5):
                text = fin.readline()
                samples.append(text)
            return samples

    def fit(self, train_data, valid_data=None, verbose=True, saved=True):
        # pretraining
        if verbose:
            self.logger.info(">> Start pretraining")
        # generator pretraining
        for epoch_idx in range(self.g_pretraining_epochs):  # 80
            if verbose:
                self.logger.info(">>>> [Pretrain Gen] Start %d / %d epochs generator pretraining" % (
                    epoch_idx + 1, self.g_pretraining_epochs))
            training_start_time = time()
            train_loss = self._g_train_epoch(train_data, epoch_idx)
            # self.g_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx + 1, training_start_time, training_end_time, train_loss,
                                                 "generator pre")
            train_loss_output = ">>>> " + train_loss_output
            if verbose:
                self.logger.info(train_loss_output)

            if epoch_idx % 15 == 0:
                self.logger.info(">>>> [Pretrain Gen] Save pretrain gen in epoch %d ..." % (epoch_idx + 1))
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
        for epoch_idx in range(self.d_pretraining_epochs):  # 5
            if verbose:
                self.logger.info(">>>> [Pretrain Dis]Start %d / %d epochs discriminator pretraining..." % (
                epoch_idx + 1, self.d_pretraining_epochs))
            training_start_time = time()
            train_loss = self._d_train_epoch(train_data, epoch_idx)
            # self.d_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss,
            #                                                                         tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss,
                                                 "discriminator pre")
            train_loss_output = ">>>> " + train_loss_output
            if verbose:
                self.logger.info(train_loss_output)

            if epoch_idx + 1 == self.d_pretraining_epochs:
                self.logger.info(">>>> [Pretrain Gen] Save pretrain dis in epoch %d ..." % (epoch_idx + 1))
                path = self._save_checkpoint(epoch_idx + 1, postfix="pretrain_dis")

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
                    self.logger.info(">>>>>> [Adv] Start epoch %d / %d adversarial training" % (
                    epoch_idx + 1, self.adversarail_training_epochs))
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
                if (epoch_idx + 1) % 5 == 0:
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

            # gen pretrain
            for epoch_idx in range(5):
                if verbose:
                    self.logger.info(">>>>>> [Adv] Start epoch %d / 5 pretrain generator" % (epoch_idx + 1))
                training_start_time = time()
                train_loss = self._g_train_epoch(train_data, epoch_idx)
                # self.g_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output((epoch_idx + 1), training_start_time, training_end_time,
                                                     train_loss,
                                                     "adv generator pre")
                train_loss_output = ">>>>>> " + train_loss_output
                if verbose:
                    self.logger.info(train_loss_output)

                if (epoch_idx + 1) % 5 == 0:
                    path = self._save_checkpoint((epoch_idx + 1), postfix="adv_pretrain")
                    self.model.eval()
                    test_result = self.evaluate(valid_data, model_file=path)
                    self.model.train()
                    self.logger.info('>>>>>> [Adv] test result: {}'.format(test_result))

            # dis pretrain
            for epoch_idx in range(5):  # d_steps
                if verbose:
                    self.logger.info(">>>>>> [Adv] Start epoch %d / 5 pretrain discriminator" % (epoch_idx + 1))
                training_start_time = time()
                train_loss = self._d_train_epoch(train_data, epoch_idx)
                # self.d_pretraining_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss,
                #                                                                         tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output((epoch_idx + 1), training_start_time, training_end_time,
                                                     train_loss,
                                                     "adv discriminator pre")
                train_loss_output = ">>>>>> " + train_loss_output
                if verbose:
                    self.logger.info(train_loss_output)

            if (epoch + 1) % 3 == 0:
                path = self._save_checkpoint((epoch + 1), postfix="adv and pre train")
                test_result = self.evaluate(valid_data, model_file=path)
                sample = self._load_generated_text()
                tmp = "\n"
                for i, s in enumerate(sample):
                    tmp += str(i)
                    tmp += ": "
                    tmp += "\n"
                self.logger.info('>>>> [Adv] Adv and pretrain test result: {}'.format(test_result))
                self.logger.info('>>>> [Adv] Adv and pretrain test result samples: {}'.format(tmp))

        self._save_checkpoint(self.adversarail_training_epochs)
        return -1, None
