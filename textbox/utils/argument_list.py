general_arguments = {
    'gpu_id', 'use_gpu', 'DDP', 'seed', 'reproducibility', 'state', 'data_path', 'checkpoint_dir', 'generated_text_dir',
    'device', 'filename', 'dataset', 'quick_test', 'config_kwargs'
}

training_arguments = {
    'epochs', 'train_batch_size', 'optimizer', 'learning_rate', 'eval_step', 'eval_epoch', 'stopping_step', 'grad_clip',
    'g_pretraining_epochs', 'd_pretraining_epochs', 'd_sample_num', 'd_sample_training_epochs',
    'adversarial_training_epochs', 'adversarial_g_epochs', 'adversarial_d_epochs', 'schedule', 'init_lr',
    'warmup_steps', 'max_steps', 'valid_metrics'
}

evaluation_arguments = {
    'beam_size', 'decoding_strategy', 'metrics', 'n_grams', 'eval_batch_size', 'generation_kwargs'
}

model_arguments = {
    'model', 'model_path', 'tokenizer_kwargs', 'tokenizer_path', 'tokenize_strategy'
}
