"""Classification of hyperparameters.
Parameters start with underscore are internal parameters and might be modified by program."""

general_parameters = [
    'gpu_id', 'use_gpu', 'seed', 'reproducibility', 'state', 'data_path', 'checkpoint_dir', 'generated_text_dir',
    'email', 'filename', 'DDP', 'logdir', 'quick_test', '_is_local_main_process', 'device', 'config_files', 'space',
    'multi_seed', 'algo', '_hyper_tuning'
]

training_parameters = [
    'do_train', 'do_valid',  # partial experiment
    'optimizer', 'adafactor_kwargs', 'optimizer_kwargs', 'scheduler', 'scheduler_kwargs',  # optimizer
    'valid_intervals', 'valid_strategy',
    'max_save',  # checkpoint saving
    'stopping_steps',  # early stopping
    'epochs', 'max_steps', 'learning_rate', 'train_batch_size', 'grad_clip', 'weight_decay',  # common parameters
    'accumulation_steps',  # accelerator
]

evaluation_parameters = [
    'do_test', 'lower_evaluation', 'multiref_strategy', 'bleu_max_ngrams', 'bleu_type', 'beam_size',
    'smoothing_function', 'corpus_bleu', 'rouge_max_ngrams', 'rouge_type', 'meteor_type', 'chrf_type',
    'distinct_max_ngrams', 'inter_distinct', 'unique_max_ngrams', 'self_bleu_max_ngrams', 'tgt_lang',
    'decoding_strategy', 'metrics', 'n_grams', 'eval_batch_size', 
]

model_parameters = [
    'model', 'model_name', 'model_path', 'config_kwargs', 'truncate', 'tokenizer_kwargs', 'generation_kwargs',
    'efficient_kwargs', 'efficient_methods', 'efficient_unfreeze_model'
]

dataset_parameters = [
    'dataset',
]
