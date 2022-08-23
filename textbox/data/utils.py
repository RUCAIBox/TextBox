from torch.utils.data import DataLoader
from textbox.data.denoising_dataset import TextInfillingCollate, DenoisingCollate
from textbox.data.abstract_dataset import AbstractDataset, AbstractCollate
from logging import getLogger

collate_options = {
    'disabled': AbstractCollate,
    'denoising': DenoisingCollate,
    'text_infilling': TextInfillingCollate
}

def data_preparation(config, tokenizer):
    train_dataset = AbstractDataset(config, 'train')
    valid_dataset = AbstractDataset(config, 'valid')
    test_dataset = AbstractDataset(config, 'test')

    train_dataset.tokenize(tokenizer)
    valid_dataset.tokenize(tokenizer)
    test_dataset.tokenize(tokenizer)

    collate_name = config['pretrain_task']
    collate_fn = collate_options.get(collate_name, AbstractCollate)
    logger = getLogger(__name__)
    logger.info(f'Pretrain type: {collate_fn.get_type()}')

    train_dataloader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, pin_memory=True, collate_fn=collate_fn(config, tokenizer, 'train'))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['eval_batch_size'], shuffle=False, pin_memory=True, collate_fn=collate_fn(config, tokenizer, 'valid'))
    test_dataloader = DataLoader(test_dataset, batch_size=config['eval_batch_size'], shuffle=False, pin_memory=True, collate_fn=collate_fn(config, tokenizer, 'test'))
    return train_dataloader, valid_dataloader, test_dataloader

