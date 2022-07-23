general_parameters = [
    'gpu_id', 'use_gpu', 'seed', 'reproducibility', 'state', 'data_path', 'checkpoint_dir', 'generated_text_dir',
    'email', 'filename', 'DDP', 'logdir', 'quick_test', 'is_local_main_process', 'device',
]

training_parameters = [
    'epochs', 'train_batch_size', 'optimizer', 'learning_rate', 'eval_step', 'eval_epoch', 'stopping_step', 'grad_clip',
    'g_pretraining_epochs', 'd_pretraining_epochs', 'd_sample_num', 'd_sample_training_epochs',
    'adversarial_training_epochs', 'adversarial_g_epochs', 'adversarial_d_epochs', 'scheduler', 'init_lr',
    'warmup_steps', 'max_steps', 'weight_decay', 'adafactor_kwargs', 'optimizer_kwargs', 'accumulation_steps',
    'max_save', 'stopping_steps'
]

evaluation_parameters = [
    'lower_evaluation', 'multiref_strategy', 'bleu_max_ngrams', 'bleu_type', 'beam_size', 'smoothing_function',
    'corpus_bleu', 'rouge_max_ngrams', 'rouge_type', 'meteor_type', 'chrf_type', 'distinct_max_ngrams',
    'inter_distinct', 'unique_max_ngrams', 'self_bleu_max_ngrams', 'tgt_lang', 'decoding_strategy', 'metrics',
    'n_grams', 'eval_batch_size',
]

model_parameters = [
    'model', 'model_name', 'model_path', 'config_kwargs', 'truncate', 'tokenizer_kwargs', 'generation_kwargs',
    'efficient_kwargs', 'efficient_methods', 'efficient_kwargs', 'efficient_unfreeze_model'
]

dataset_parameters = [
    'dataset',
]
