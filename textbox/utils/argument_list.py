"""Classification of hyperparameters.
Parameters start with underscore are internal variables and might be modified by program."""

general_parameters = [
    'gpu_id',
    'use_gpu',  # GPU
    'device',
    '_is_local_main_process',  # internal variables
    'seed',
    'reproducibility',  # reproducibility
    'config_files',
    'cmd',  # input
    'filename',
    'saved_dir',
    'saved',
    'state',
    'email',  # output
    'quick_test',  # partial experiment
    'space',
    'algo',
    '_hyper_tuning',  # hyper tuning
    'multi_seed',  # multiple random seed
    'romanian_postprocessing',
    'wandb'
]

training_parameters = [
    'do_train',
    'do_valid',  # partial experiment
    'optimizer',
    'adafactor_kwargs',
    'optimizer_kwargs',
    'scheduler',
    'scheduler_kwargs',  # optimizer
    'valid_steps',
    'valid_strategy',  # validation
    'max_save',  # checkpoint saving
    'stopping_steps',  # early stopping
    'epochs',
    'max_steps',
    'learning_rate',
    'train_batch_size',
    'grad_clip',
    'weight_decay',  # common parameters
    'accumulation_steps',  # accelerator
    'disable_tqdm',  # tqdm
    'pretrain_task',  # pretraining
    'resume_training'
]

evaluation_parameters = [
    'do_test',  # partial experiment
    'lower_evaluation',
    'multiref_strategy',
    'bleu_max_ngrams',
    'bleu_type',
    'beam_size',
    'smoothing_function',
    'corpus_bleu',
    'rouge_max_ngrams',
    'rouge_type',
    'meteor_type',
    'chrf_type',
    'distinct_max_ngrams',
    'inter_distinct',
    'unique_max_ngrams',
    'self_bleu_max_ngrams',
    'tgt_lang',
    'decoding_strategy',
    'metrics',
    'n_grams',
    'eval_batch_size',
    'corpus_meteor'
]

model_parameters = [
    'model',
    'model_name',  # model name
    'model_path',
    'config_path',
    'config_kwargs',
    'tokenizer_path',
    'tokenizer_kwargs',
    'generation_kwargs',  # hf
    'efficient_kwargs',
    'efficient_methods',
    'efficient_unfreeze_model',  # efficient methods
    'label_smoothing',
]

dataset_parameters = [
    'dataset',
    'data_path',  # dataset name
    'src_lang',
    'tgt_lang',  # dataset language
    'src_vocab_size',
    'tgt_vocab_size',  # vocab
    'src_len',
    'tgt_len',
    'truncate',  # dataset maximal length
    'tokenize_strategy',  # tokenize strategy
]

efficient_kwargs_dict = {
    'lora': {
        'lora_r': 4,
        'lora_dropout': 0.1,
        'lora_alpha': 32
    },
    'prefix-tuning': {
        'prefix_length': 100,
        'prefix_dropout': 0.1,
        'prefix_mid_dim': 512
    },
    'p-tuning-v2': {
        'prefix_length': 100,
        'prefix_dropout': 0.1,
        'prefix_mid_dim': 512
    },
    'adapter': {
        'adapter_mid_dim': 64
    },
    'prompt-tuning': {
        'prompt_length': 100
    },
    'bitfit': {}
}
