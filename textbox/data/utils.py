from torch.utils.data import DataLoader
from textbox.data.denoising_dataset import DenoisingCollate
from ..data.unilm_dataset import UnilmCollate
from textbox.data.abstract_dataset import AbstractDataset, AbstractCollate
from accelerate.logging import get_logger

collate_options = {'disabled': AbstractCollate, 'denoising': DenoisingCollate, 'unilm': UnilmCollate}


def data_preparation(config, tokenizer):
    collate_name = config['pretrain_task']
    if config['model_name'] == 'unilm':
        collate_name = 'unilm'
    collate_fn = collate_options.get(collate_name, AbstractCollate)
    logger = get_logger(__name__)
    logger.info(f'Pretrain type: {collate_fn.get_type()}')

    if config['dataset'] == 'multiwoz':
        assert config['eval_batch_size'] % 3 == 0

    dataloaders = []
    if config['do_train']:
        train_dataset = AbstractDataset(config, 'train')
        train_dataset.tokenize(tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['train_batch_size'],
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn(config, tokenizer, 'train')
        )
        dataloaders.append(train_dataloader)
    else:
        dataloaders.append(None)

    if config['do_valid']:
        valid_dataset = AbstractDataset(config, 'valid')
        valid_dataset.tokenize(tokenizer)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config['eval_batch_size'],
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn(config, tokenizer, 'valid')
        )
        dataloaders.append(valid_dataloader)
    else:
        dataloaders.append(None)

    if config['do_test']:
        test_dataset = AbstractDataset(config, 'test')
        test_dataset.tokenize(tokenizer)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['eval_batch_size'],
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn(config, tokenizer, 'test')
        )
        dataloaders.append(test_dataloader)
    else:
        dataloaders.append(None)

    return dataloaders
